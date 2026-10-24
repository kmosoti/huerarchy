from collections import defaultdict
import csv
import gzip
from html import escape
import json
from pathlib import Path

import numpy as np

from ..common import file_hash, json_lines, write_json
from ..learning.report import curve_svg, table
from .experiment import verify


def comparisons(records):
    index = {(r["task"], r["noise"], r["seed"], r["variant"]): r for r in records}
    rng = np.random.default_rng(27182)
    rows = []
    for family in ("mechanism", "pair", "relay"):
        for changing in (False, True):
            for candidate, baseline in (("adaptive", "small-reset"), ("adaptive-library", "adaptive"),
                                        ("library-progress", "adaptive-library"), ("library-random", "adaptive-library")):
                pairs = defaultdict(list)
                for r in records:
                    if r["diagnostic"] or (r["family"], r["changing"], r["variant"]) != (family, changing, candidate):
                        continue
                    other = index.get((r["task"], r["noise"], r["seed"], baseline))
                    if other:
                        pairs[r["curriculum"]].append(r["balanced_accuracy_auc"] - other["balanced_accuracy_auc"])
                if pairs:
                    values = np.array([np.mean(d) for d in pairs.values()])
                    bootstrap = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
                    rows.append({"family": family, "changing": changing, "candidate": candidate, "baseline": baseline,
                                 "curricula": len(values), "difference": float(values.mean()),
                                 "low": float(np.percentile(bootstrap, 2.5)), "high": float(np.percentile(bootstrap, 97.5))})
    return rows


def render(directory):
    directory = Path(directory)
    manifest = verify(directory)
    config = json.loads((directory / "config.json").read_text())
    records = list(json_lines(directory / "runs.jsonl"))
    if len(records) != manifest["task_runs"] or not all(r["checkpoint_verified"] for r in records):
        raise ValueError("discovery run/checkpoint counts disagree")
    groups = defaultdict(list)
    for r in records:
        groups[(r["family"], r["changing"], r["variant"])].append(r)
    aggregates = []
    fields = ("balanced_accuracy_auc", "final_balanced_accuracy", "expected_brier", "confident_coverage",
              "growth_promotions", "confirmed_resets", "uncertainty_admissions", "alarms", "proposals_returned",
              "search_program_context_evaluations", "unique_questions", "available_concepts", "imported_concepts", "new_concepts", "wall_seconds", "cpu_seconds")
    for (family, changing, variant), items in sorted(groups.items()):
        row = {"family": family, "changing": changing, "variant": variant, "runs": len(items)}
        row.update({field: float(np.mean([r[field] for r in items])) for field in fields})
        row["budget_exhausted_runs"] = sum(r["budget_exhausted"] for r in items)
        aggregates.append(row)
    paired = comparisons(records)
    concepts = list(json_lines(directory / "concepts.jsonl"))
    findings = []
    for family in ("mechanism", "pair", "relay"):
        for changing in (False, True):
            rows = [a for a in aggregates if a["family"] == family and a["changing"] == changing]
            if rows:
                winner = max(rows, key=lambda a: a["balanced_accuracy_auc"])
                findings.append(f"{'Changing' if changing else 'Stable'} {family}: {winner['variant']} had the highest mean trajectory accuracy "
                                f"({winner['balanced_accuracy_auc']:.1%}) among the tested configurations. This is descriptive, not a guarantee of superiority.")
    summary = {"schema": 1, "task_runs": manifest["task_runs"], "teaching_turns": manifest["teaching_turns"],
               "rust_proofs": manifest["rust_proofs"], "checkpoints_verified": len(records),
               "concept_records": len(concepts), "aggregates": aggregates, "paired_comparisons": paired,
               "findings": findings, "config": config, "elapsed_seconds": manifest["elapsed_seconds"],
               "cpu_seconds": manifest["cpu_seconds"], "peak_rss_bytes": manifest["peak_rss_bytes"]}
    write_json(directory / "summary.json", summary)
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    sections = ["<h1>Growing explanations in three discovery worlds</h1>",
                f"<p>{manifest['task_runs']:,} task runs; {manifest['teaching_turns']:,} teaching responses; "
                f"{manifest['rust_proofs']} Python/Rust proof audits. Every final checkpoint reproduced predictions and its next question.</p>",
                "<p>The learner constructs small rule programs from supplied building blocks. It does not invent an unrestricted language, "
                "acquire natural language, or update neural weights. This is a controlled comparison of existing research ideas.</p>",
                "<h2>Measured findings</h2><ul>" + "".join(f"<li>{escape(f)}</li>" for f in findings) + "</ul>",
                "<h2>The worlds</h2>",
                table(["World", "Task", "Unseen evaluation"], [
                    ["Mechanism", "Conjunctions, alternatives, exactly-one conditions, and majority rules", "Unqueried input combinations"],
                    ["Pair workshop", "Different devices share a pattern involving contacts and matching attributes", "New objects and larger scenes"],
                    ["Relay workshop", "Signals pass through compatible connections; cycles and broken edges are present", "Paths with 4–7 links, versus 1–3 during teaching"]]),
                "<p>Each curriculum contains one mechanism task, three pair tasks, and a relay task. Object names are random. "
                "The measurement schema is shared and versioned; new object names do not imply a new meaning for the measurements. "
                "Scenes and allowable tests are public; targets, clean labels, feedback noise, and change boundaries remain evaluator-owned. "
                "This is an API/data-flow boundary between trusted local components, not a security sandbox.</p>",
                "<h2>Learning conditions</h2>",
                table(["Condition", "Behavior"], [
                    ["small-reset", "Single-clause starting language; immediate Page–Hinkley resets"],
                    ["broad-upfront", "Rich productions available immediately; bounded candidates refreshed directly from observed fit"],
                    ["adaptive", "New programs compete with a reset and uncertainty on fresh feedback"],
                    ["adaptive-library", "Adaptive growth plus persistent relational helpers"],
                    ["library-random", "Same growth and library with random questions"],
                    ["library-progress", "Same growth and library with loss-improvement-driven question groups"]]),
                "<p>The broad-upfront condition is a bounded search baseline, not exhaustive optimal search over every richer program. "
                "Its direct candidate admission and the adaptive fresh-response gate differ; comparisons assess these complete strategies. "
                "Information-seeking questions are used unless random/progress is specified.</p>",
                f"<p>Noise levels {escape(str(config['noise_rates']))}; seeds {escape(str(config['seeds']))}. The learner assumes {config['assumed_noise']:.0%} noise. "
                f"Changing curricula switch mechanisms after responses {escape(str(config['changes']))}. "
                f"New explanations face {config['confirmation']} fresh responses. Improvement must exceed added program complexity plus "
                f"{config['promotion_margin']:.3f} log-score units, and must improve raw prediction as well. "
                "This margin is a heuristic acceptance rule, not a calibrated significance test.</p>",
                "<h2>Prediction and adaptation</h2><p>Accuracy is held-out balanced accuracy averaged over the teaching trajectory. "
                "Final accuracy averages the last 20 responses. Expected Brier scores include irreducible randomness in stochastic diagnostics. "
                "All methods pay for earlier curriculum lessons, including acquiring reusable concepts. Per-run records show cumulative label and search costs.</p>"]
    sections.append(table(["World", "Changes", "Condition", "Accuracy", "Final", "Brier", "Growth/task", "Used concepts", "Budget stops"], [
        [a["family"], "yes" if a["changing"] else "no", a["variant"], f"{a['balanced_accuracy_auc']:.3f}",
         f"{a['final_balanced_accuracy']:.3f}", f"{a['expected_brier']:.3f}", f"{a['growth_promotions']:.2f}",
         f"{a['imported_concepts']:.2f}", a["budget_exhausted_runs"]] for a in aggregates if a["family"] in {"mechanism", "pair", "relay"}]))
    sections.append("<h2>Paired comparisons</h2><p>Differences in mean trajectory balanced accuracy. Noise levels, seed repeats, and pair tasks "
                    "are averaged within each curriculum; percentile bootstrap intervals resample curricula (2,000 draws). "
                    "There are few independent curricula, especially after separating stable and changing cases. These are exploratory intervals.</p>")
    sections.append(table(["World", "Changes", "Comparison", "Difference", "95% interval", "Curricula"], [
        [p["family"], p["changing"], p["candidate"] + " minus " + p["baseline"], f"{p['difference']:+.4f}",
         f"[{p['low']:+.4f}, {p['high']:+.4f}]", p["curricula"]] for p in paired]))
    sections.append("<h2>Concepts and explanations</h2><p>A helper is accepted only if it occurs at least three times across two tasks and "
                    "reduces corpus description size after paying for its definition. That is a semantics-preserving shortcut relative to the source programs; "
                    "it does not establish that those programs describe reality. Definitions are immutable and versioned. "
                    "Later prediction performance tests whether the shortcut is useful. Concept counts alone are not a success metric.</p>")
    example_concepts = []
    seen = set()
    for c in concepts:
        key = json.dumps(c["definition"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            example_concepts.append(c)
    from .programs import Program
    for c in example_concepts[:8]:
        sections.append(f"<p>{escape(c['name'])}: {c['uses']} uses across {len(c['source_tasks'])} tasks; "
                        f"{c['description_savings_nodes']} nodes saved.</p><pre>{escape(Program.from_dict(c['definition']).text())}</pre>")
    if not concepts:
        sections.append("<p>No helper met the acceptance conditions in this run. This is an experimental outcome, not evidence of successful concept discovery.</p>")
    examples = list(json_lines(directory / "explanations.jsonl"))
    for example in examples[:6]:
        sections.append(f"<details><summary>{escape(example['run_id'])}</summary><pre>{escape(example['readable_program'])}</pre>"
                        f"<p>Conditional prediction: {example['audit']['outcome']}; candidate mass {example['posterior_mass']:.3f}. "
                        "The full source-fact dictionary and Rust proof are in explanations.jsonl.</p></details>")
    sections.append("<h2>Learning curves</h2>")
    lookup = {r["run_id"]: r for r in records}
    sums, counts = {}, defaultdict(int)
    trace_count = 0
    with gzip.open(directory / "trace.jsonl.gz", "rt") as stream:
        for line in stream:
            trace = json.loads(line)
            record = lookup[trace["run_id"]]
            trace_count += 1
            if record["diagnostic"]:
                continue
            key = (record["family"], record["changing"], record["variant"])
            if key not in sums:
                sums[key] = np.zeros(config["steps"])
            sums[key][trace["sequence"] - 1] += trace["balanced_accuracy"]
            counts[key] += trace["sequence"] == 1
    if trace_count != manifest["teaching_turns"]:
        raise ValueError("discovery trace count mismatch")
    for family in ("mechanism", "pair", "relay"):
        for changing in (False, True):
            curves = {key[2]: values / counts[key] for key, values in sums.items() if key[:2] == (family, changing)}
            if curves:
                sections.append(f"<h3>{family.capitalize()}, {'changing' if changing else 'stable'}</h3>" + curve_svg(curves, config["steps"], config["changes"] if changing else []))
    sections.append("<h2>Failure controls</h2><p>Coin outputs are independent random events. Distraction mixes random outputs with a learnable rule. "
                    "Unsupported four-bit parity needs eight conjunction clauses, beyond the four-clause bound. "
                    "These controls expose confidence errors, unnecessary growth, and a representational ceiling. They are excluded from primary comparisons.</p>")
    sections.append(table(["Diagnostic", "Condition", "Accuracy", "Brier", "Confidence coverage", "Growth/task", "Uncertainty/task"], [
        [a["family"], a["variant"], f"{a['balanced_accuracy_auc']:.3f}", f"{a['expected_brier']:.3f}",
         f"{a['confident_coverage']:.3f}", f"{a['growth_promotions']:.2f}", f"{a['uncertainty_admissions']:.2f}"]
        for a in aggregates if a["family"] not in {"mechanism", "pair", "relay"}]))
    sections.append(f"<h2>Resources and reproducibility</h2><p>Teaching/evaluation loop: {manifest['elapsed_seconds']:.1f}s wall, "
                    f"{manifest['cpu_seconds']:.1f}s Python CPU; peak process RSS {manifest['peak_rss_bytes']/2**20:.1f} MiB. "
                    "CPU/RSS exclude native proof child processes. Timings include storage, checkpoint verification, and evaluation; "
                    "search context counters separately charge candidate predictions on public training scenes. No energy sensor is available.</p>")
    if manifest.get("resource_scope"):
        sections.append(f"<p>{escape(manifest['resource_scope'])}. {manifest.get('workers', 1)} independent curriculum workers. "
                        "Immutable syntax/type caches may be warm across tasks; these wall times are not isolated cold-kernel comparisons.</p>")
    research = json.loads((directory / "research.json").read_text())
    sections.append("<h2>Research and limits</h2><ul>" + "".join(
        f'<li><a href="{escape(s["url"])}">{escape(s["title"])}</a>: {escape(s["use"])} {escape(s["limit"])}</li>' for s in research["sources"]) + "</ul>")
    sections.append("<p>Reproduce with <code>python -m research_lab discovery run --config config.json --output NEW_DIRECTORY</code>.</p><p>" + " · ".join(
        f'<a href="{name}">{name}</a>' for name in ("config.json", "summary.json", "summary.csv", "runs.jsonl", "trace.jsonl.gz", "events.jsonl.gz", "concepts.jsonl", "explanations.jsonl", "research.json", "worlds.json.gz", "memory.sqlite", "manifest.json")) + "</p>")
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Discovery worlds and hypothesis expansion</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#172033}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ddd;padding:6px;text-align:left}th{background:#edf2f7}h2{margin-top:2rem}svg{max-width:100%;font:12px system-ui}pre{white-space:pre-wrap;background:#f3f5f7;padding:1rem}</style>' + "".join(sections) + "</html>"
    (directory / "REPORT.html").write_text(html)
    for name in ("summary.json", "summary.csv", "REPORT.html"):
        manifest["outputs"][name] = file_hash(directory / name)
    manifest["render_sources"] = {"research_lab/discovery/report.py": file_hash(Path(__file__))}
    write_json(directory / "manifest.json", manifest)
    return directory / "REPORT.html"
