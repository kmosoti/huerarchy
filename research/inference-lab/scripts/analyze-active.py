"""Add source-bound plain-language findings and censored recovery plots to a completed study."""
import argparse
from collections import defaultdict
import csv
import gzip
from html import escape
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_lab.active.study import verify
from research_lab.common import file_hash, write_json


def query_coverage(root, blocks):
    """Post-study description of public teaching inputs; no learner or oracle execution."""
    records = []
    for block in blocks:
        if block["status"] != "complete" or block["kind"] != "core":
            continue
        directory = Path(root) / block["path"]
        runs = {r["run_id"]: r for r in map(json.loads, (directory / "runs.jsonl").read_text().splitlines())
                if r["family"] == "relay" and r["role"] == "primary"}
        with gzip.open(directory / "worlds.json.gz", "rt") as stream:
            worlds = json.load(stream)
        counts = defaultdict(lambda: [0, 0, set()])
        with gzip.open(directory / "trace.jsonl.gz", "rt") as stream:
            for line in stream:
                observation = json.loads(line)
                run = runs.get(observation["run_id"])
                if run is None:
                    continue
                question = observation["feedback"]["question_index"]
                large = len(worlds[run["task"]]["view"]["questions"][question]["entities"]) == 4
                for period in ("all", "first32") if observation["sequence"] <= 32 else ("all",):
                    value = counts[run["run_id"], period]
                    value[0] += 1; value[1] += large; value[2].add(question)
        assert {run_id for run_id, period in counts if period == "all"} == set(runs)
        for (run_id, period), (answers, large, distinct) in sorted(counts.items()):
            run = runs[run_id]
            assert answers == (run["answers"] if period == "all" else min(32, run["answers"]))
            pool = worlds[run["task"]]["view"]["questions"]
            records.append({"block": block["block"], "run_id": run_id, "regime": run["task"].split("/")[1],
                            "variant": run["variant"], "noise": run["noise"], "seed": run["seed"], "period": period,
                            "answers": answers, "four_object_fraction": large / answers,
                            "pool_four_object_fraction": sum(len(s["entities"]) == 4 for s in pool) / len(pool),
                            "distinct_query_fraction": len(distinct) / len(pool)})
    groups = defaultdict(list)
    for row in records:
        groups[row["regime"], row["variant"], row["period"]].append(row)
    aggregates = []
    for (regime, variant, period), rows in sorted(groups.items()):
        aggregates.append({"regime": regime, "variant": variant, "period": period,
                           "runs": len(rows), "worlds": len({r["block"] for r in rows}),
                           **{key: sum(r[key] for r in rows) / len(rows) for key in
                              ("four_object_fraction", "pool_four_object_fraction", "distinct_query_fraction")}})
    return records, aggregates


def analyze(root):
    root = Path(root)
    manifest = verify(root)
    summary = json.loads((root / "summary.json").read_text())
    rows = summary["aggregates"]
    index = {(r["role"], r["family"], r["regime"], r["variant"]): r for r in rows}
    findings = []
    if summary["partial_blocks"]:
        findings.append(f"{summary['partial_blocks']} blocks were incomplete and are excluded from paired comparisons. "
                        "Completion can depend on task difficulty, so the remaining sample may be biased toward faster worlds; partial traces are retained.")
    for regime, label in (("abc", "successive new rules"), ("aba", "a returning rule")):
        a = index.get(("primary", "relay", regime, "epig-change"))
        b = index.get(("primary", "relay", regime, "broad-upfront"))
        if a and b:
            findings.append(f"On relays with {label}, the new EPIG/change learner recovered on {a['recovered']}/{a['transitions']} changes, "
                            f"versus {b['recovered']}/{b['transitions']} for broad-upfront. Mean capped recovery cost was "
                            f"{a['recovery_cost']:.1f} versus {b['recovery_cost']:.1f} answers; failures retain their available horizon.")
    a = index.get(("primary", "coin", "diagnostic", "epig-change"))
    b = index.get(("primary", "coin", "diagnostic", "broad-upfront"))
    if a and b and a["post32_brier"] is not None:
        findings.append(f"After the first 32 answers on random-coin tasks, the new learner's probability error (Brier) was "
                        f"{a['post32_brier']:.3f}, versus {b['post32_brier']:.3f}; 0.250 is optimal for a fair coin. "
                        f"It assigned at least 90% confidence to {a['post32_confident_coverage']:.1%} of predictions, versus {b['post32_confident_coverage']:.1%}.")
    findings.append(f"The library produced {summary['concept_records']} concept records across repeated independent runs; "
                    f"{summary['promoted_concept_records']} passed the two-task predictive gate. These are records, not globally distinct concepts.")
    for family in ("pair", "relay"):
        a = index.get(("target", family, "transfer", "library-validated"))
        b = index.get(("target", family, "transfer", "library-none"))
        if a and b:
            findings.append(f"On new {family} tasks, the validated library changed mean accuracy during learning from {b['accuracy']:.1%} to {a['accuracy']:.1%}. "
                            f"Total capped answers to success, including acquisition, were {a['total_initial_cost']:.1f} versus {b['total_initial_cost']:.1f}; "
                            f"total search costs were {a['total_search_contexts']:.0f} versus {b['total_search_contexts']:.0f} program–scene evaluations.")
    recoveries = defaultdict(list)
    noise_groups, library_stages = defaultdict(list), defaultdict(lambda: {"libraries": 0, "candidate_records": 0, "promoted_records": 0, "target_runs": 0, "target_runs_using_imports": 0})
    for block in manifest["blocks"]:
        if block["status"] != "complete":
            continue
        for line in (root / block["path"] / "concepts.jsonl").read_text().splitlines():
            library = json.loads(line)["library"]
            stage = library_stages[library["kind"]]
            stage["libraries"] += 1
            stage["candidate_records"] += len(library["concepts"])
            stage["promoted_records"] += sum(c["status"] == "promoted" for c in library["concepts"])
        for line in (root / block["path"] / "runs.jsonl").read_text().splitlines():
            r = json.loads(line)
            regime = r["task"].split("/")[1]
            noise_groups[(r["role"], r["family"], regime, r["variant"], r["noise"])].append((block["block"], r))
            if r["role"] == "target":
                stage = library_stages[r["variant"].removeprefix("library-")]
                stage["target_runs"] += 1
                stage["target_runs_using_imports"] += bool(r["imported_concepts"])
            if block["kind"] == "core" and r["family"] == "relay" and regime in {"abc", "aba"}:
                recoveries[(regime, r["variant"])].extend((p["delay"], p["horizon"]) for p in r["phases"][1:])
    noise_rows = []
    for (role, family, regime, variant, noise), items in sorted(noise_groups.items()):
        row = dict(role=role, family=family, regime=regime, variant=variant, noise=noise,
                   runs=len(items), worlds=len({block for block, _ in items}))
        for metric in ("accuracy", "final_accuracy", "brier", "log_loss", "post32_brier", "post32_confident_coverage", "cpu_seconds", "query_cpu_seconds", "search_contexts"):
            values = [r[metric] for _, r in items if r[metric] is not None]
            row[metric] = sum(values) / len(values) if values else None
        initial = [r["phases"][0]["capped_cost"] for _, r in items if r["phases"]]
        transitions = [phase for _, r in items for phase in r["phases"][1:]]
        row.update(initial_capped_answers=sum(initial) / len(initial) if initial else None,
                   recovery_capped_answers=sum(p["capped_cost"] for p in transitions) / len(transitions) if transitions else None,
                   recovered=sum(p["delay"] is not None for p in transitions), transitions=len(transitions),
                   budget_stops=sum(r["budget_exhausted"] for _, r in items))
        noise_rows.append(row)
    with (root / "noise-breakdown.csv").open("w", newline="") as stream:
        if noise_rows:
            writer = csv.DictWriter(stream, fieldnames=list(noise_rows[0]))
            writer.writeheader(); writer.writerows(noise_rows)
    coverage_records, coverage = query_coverage(root, manifest["blocks"])
    with (root / "relay-query-coverage.csv").open("w", newline="") as stream:
        if coverage_records:
            writer = csv.DictWriter(stream, fieldnames=list(coverage_records[0]))
            writer.writeheader(); writer.writerows(coverage_records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.hashsalt"] = "active-recovery"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    curves = []
    for ax, regime in zip(axes, ("abc", "aba")):
        for variant in ("broad-upfront", "broad-window32", "epig-change", "epig-change-archive", "gate32", "gate-sequential"):
            observations = recoveries.get((regime, variant), [])
            if not observations:
                continue
            times = sorted({delay if delay is not None else horizon for delay, horizon in observations})
            x, y, survival = [0], [0.], 1.
            for t in times:
                at_risk = sum((d if d is not None else h) >= t for d, h in observations)
                events = sum(d == t for d, _ in observations)
                survival *= 1 - events / at_risk
                x.append(t); y.append(1 - survival)
            ax.step(x, y, where="post", label=variant)
            curves.append({"regime": regime, "variant": variant, "responses": x, "estimated_recovered_fraction": y,
                           "transitions": len(observations), "recoveries": sum(d is not None for d, _ in observations)})
        ax.set(title="New rules" if regime == "abc" else "Returning rule", xlabel="Answers after a rule change", ylim=(0, 1), ylabel="Estimated fraction recovered")
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(root / f"recovery.{ext}", dpi=150, metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)
    analysis = {"findings": findings, "recovery_curves": curves, "library_stages": dict(library_stages),
                "noise_breakdown": noise_rows,
                "exploratory_relay_query_coverage": coverage,
                "query_coverage_scope": "Post-hoc descriptive diagnostic added after inspecting the complete core results. It measures public teaching-graph size and distinct questions for every relay condition. Four objects is the largest permitted teaching graph, including broken chains; it is not a label of a successful three-link path. These associations do not establish why a policy generalizes better or worse. No experiment settings were changed.",
                "diagnostic_scope": "All completed conditions, with no test-selected filtering. Library imports describe the final most-probable program, not every provisional candidate. Noise-shift tasks use their declared changing noise schedule instead of the nominal noise column.",
                "curve_method": "Kaplan–Meier descriptive curves. Unrecovered transitions are censored at the next change/end; phase durations vary. Censoring independence is an assumption, so read curves with observed recovery fractions and capped costs.",
                "target_assessments": summary["target_assessments"]}
    write_json(root / "analysis.json", analysis)
    document = (root / "REPORT.html").read_text()
    document = re.sub(r'<section id="active-findings">.*?</section>', '', document, flags=re.S)
    section = '<section id="active-findings"><h2>What the study found</h2><ul>' + ''.join(f'<li>{escape(f)}</li>' for f in findings) + '</ul><p><img src="recovery.png" alt="Recovery with censored failures"><br><a href="recovery.svg">Recovery figure (SVG)</a> · <a href="analysis.json">Analysis data</a></p><p>' + escape(analysis['curve_method']) + '</p></section>'
    section = section.replace('<h2>What the study found</h2>', '<h2>What the study found</h2><p>Accuracy averages held-out balanced accuracy over the full teaching trace. The final_accuracy field in the CSV and JSON averages the final 20 answers. Success requires at least 90% accuracy for ten consecutive answers; failed attempts retain their available horizon in capped costs.</p>', 1)
    from research_lab.active.report import table
    stages = table(["Library", "Repeated libraries", "Candidate records", "Promoted records", "Targets whose final program uses a concept"],
                   [[kind, s["libraries"], s["candidate_records"], s["promoted_records"], f"{s['target_runs_using_imports']}/{s['target_runs']}"] for kind, s in sorted(library_stages.items())])
    section = section.removesuffix('</section>') + '<h3>Where concept reuse succeeds or stops</h3>' + stages + '<p>Counts include repeated world/noise/seed runs. Only the validated condition has a predictive promotion gate. <a href="noise-breakdown.csv">All conditions separated by feedback noise (CSV)</a>.</p></section>'
    coverage_table = table(["Regime", "Learner", "Four-object questions in first 32 answers", "Distinct teaching pool visited in first 32 answers"],
                           [[r["regime"], r["variant"], f"{r['four_object_fraction']:.1%}", f"{r['distinct_query_fraction']:.1%}"] for r in coverage
                            if r["period"] == "first32" and r["variant"] in {"broad-upfront", "eig-change", "epig-change"}])
    section = section.removesuffix('</section>') + '<h3>Exploratory question-selection diagnostic</h3><p>' + escape(analysis['query_coverage_scope']) + ' <a href="relay-query-coverage.csv">Per-run coverage measurements (CSV)</a>; aggregate measurements are in analysis.json.</p>' + coverage_table + '</section>'
    document = document.replace('<h2>Measured comparisons</h2>', section + '<h2>Measured comparisons</h2>', 1)
    (root / "REPORT.html").write_text(document)
    for name in ("analysis.json", "noise-breakdown.csv", "relay-query-coverage.csv", "recovery.png", "recovery.svg", "REPORT.html"):
        manifest["outputs"][name] = file_hash(root / name)
    manifest.setdefault("postprocessing_sources", {})[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    write_json(root / "manifest.json", manifest)
    print("\n".join(findings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    analyze(parser.parse_args().run)
