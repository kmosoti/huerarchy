from __future__ import annotations

from collections import defaultdict
import csv
import gzip
from html import escape
import json
from pathlib import Path

import numpy as np

from ..common import file_hash, write_json


def average(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{escape(str(h))}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'


def render(root):
    from .study import verify
    root = Path(root)
    manifest = verify(root)
    records, concept_records = [], []
    for b in manifest["blocks"]:
        if b["status"] != "complete":
            continue
        for line in (root / b["path"] / "runs.jsonl").read_text().splitlines():
            records.append({**json.loads(line), "block": b["block"], "kind": b["kind"], "block_path": b["path"]})
        concept_records.extend(json.loads(line) for line in (root / b["path"] / "concepts.jsonl").read_text().splitlines())
    groups = defaultdict(list)
    for r in records:
        r["regime"] = r["task"].split("/")[1]
        r["initial_cost"] = r["phases"][0]["capped_cost"] if r["phases"] else None
        r["recovery_cost"] = average([p["capped_cost"] for p in r["phases"][1:]])
        r["total_initial_cost"] = r["acquisition_before"] + r["initial_cost"] if r["initial_cost"] is not None else None
        r["total_search_contexts"] = r["search_before"] + r["search_contexts"]
        groups[(r["role"], r["family"], r["regime"], r["variant"])].append(r)
    aggregates = []
    for (role, family, regime, variant), items in sorted(groups.items()):
        row = {"role": role, "family": family, "regime": regime, "variant": variant, "runs": len(items), "worlds": len({r["block"] for r in items})}
        for key in ("accuracy", "final_accuracy", "brier", "log_loss", "post32_brier", "post32_confident_coverage", "initial_cost", "recovery_cost", "cpu_seconds", "query_cpu_seconds", "search_contexts", "total_search_contexts", "total_initial_cost", "acquisition_before", "cpu_before"):
            row[key] = average([r[key] for r in items])
        transitions = [p for r in items for p in r["phases"][1:]]
        row.update(recovered=sum(p["delay"] is not None for p in transitions), transitions=len(transitions),
                   initial_successes=sum(bool(r["phases"]) and r["phases"][0]["delay"] is not None for r in items),
                   budget_exhausted=sum(r["budget_exhausted"] for r in items),
                   imported_programs=sum(bool(r["imported_concepts"]) for r in items))
        aggregates.append(row)
    contrasts = [("epig-window", "eig-window"), ("epig-change", "eig-change"), ("eig-change", "eig-window"),
                 ("epig-change", "epig-window"), ("epig-change", "broad-upfront"), ("epig-change", "broad-window32"),
                 ("eig-window", "random-window"), ("gate8", "gate32"), ("gate-sequential", "gate32"),
                 ("epig-change-archive", "epig-change"), ("library-exact", "library-none"),
                 ("library-typed", "library-exact"), ("library-validated", "library-typed"), ("library-validated", "library-none")]
    paired = []
    rng = np.random.default_rng(20261024)
    for role, family, regime in sorted({k[:3] for k in groups}):
        for candidate, baseline in contrasts:
            a, b = groups.get((role, family, regime, candidate)), groups.get((role, family, regime, baseline))
            if not a or not b:
                continue
            for metric in ("accuracy", "brier", "log_loss", "initial_cost", "recovery_cost", "search_contexts", "total_search_contexts", "total_initial_cost", "cpu_seconds"):
                aa = {block: average([r[metric] for r in a if r["block"] == block]) for block in {r["block"] for r in a}}
                bb = {block: average([r[metric] for r in b if r["block"] == block]) for block in {r["block"] for r in b}}
                blocks = sorted(k for k in aa.keys() & bb.keys() if aa[k] is not None and bb[k] is not None)
                if not blocks:
                    continue
                differences = np.array([aa[k] - bb[k] for k in blocks])
                bootstrap = rng.choice(differences, (5000, len(blocks)), replace=True).mean(axis=1)
                paired.append({"role": role, "family": family, "regime": regime, "candidate": candidate, "baseline": baseline,
                    "metric": metric, "worlds": len(blocks), "difference": float(differences.mean()),
                    "low": float(np.quantile(bootstrap, .025)), "high": float(np.quantile(bootstrap, .975)),
                    "baseline_mean": average(list(bb.values())), "exploratory": len(blocks) < 8})
    summary = {"schema": 1, "complete_blocks": sum(b["status"] == "complete" for b in manifest["blocks"]),
        "partial_blocks": sum(b["status"] != "complete" for b in manifest["blocks"]),
        "task_runs": len(records), "acquired_answers": sum(r["acquired_answers"] for r in records),
        "processed_answers": sum(r["answers"] for r in records), "checkpoints": sum(r["checkpoint_verified"] for r in records),
        "cpu_seconds": sum(b["cpu_seconds"] for b in manifest["blocks"]), "elapsed_seconds": manifest["elapsed_seconds"],
        "largest_process_rss_bytes": max([b["peak_process_rss_bytes"] for b in manifest["blocks"]] or [0]),
        "aggregates": aggregates, "paired_comparisons": paired,
        "concept_records": sum(len(r["library"]["concepts"]) for r in concept_records),
        "promoted_concept_records": sum(c["status"] == "promoted" for r in concept_records for c in r["library"]["concepts"]),
        "scope": "Bounded symbolic hypotheses; no neural learning. Intervals resample whole world blocks; all contrasts reported, no test-selected winner."}
    by_key = {(a["role"], a["family"], a["regime"], a["variant"]): a for a in aggregates}
    targets = []
    stable_differences = [p["difference"] for p in paired if p["candidate"] == "epig-change" and p["baseline"] == "broad-upfront" and p["metric"] == "accuracy" and p["regime"] == "stable"]
    stable_ok = bool(stable_differences) and min(stable_differences) >= -.02
    for p in paired:
        if p["family"] == "relay" and p["candidate"] == "epig-change" and p["baseline"] == "broad-upfront" and p["metric"] == "recovery_cost" and p["regime"] in {"abc", "aba"}:
            reduction = -p["difference"] / p["baseline_mean"] if p["baseline_mean"] else 0
            targets.append({"target": "recovery", "regime": p["regime"], "relative_cost_reduction": reduction,
                            "stable_accuracy_guardrail": stable_ok, "met_in_sample": reduction >= .2 and stable_ok})
    for a in aggregates:
        if a["family"] == "coin" and a["post32_brier"] is not None:
            targets.append({"target": "random-calibration", "variant": a["variant"], "brier": a["post32_brier"],
                "confident_coverage": a["post32_confident_coverage"],
                "met_in_sample": a["post32_brier"] <= .27 and a["post32_confident_coverage"] <= .1})
    for family in ("pair", "relay"):
        a = by_key.get(("target", family, "transfer", "library-validated"))
        b = by_key.get(("target", family, "transfer", "library-none"))
        if a and b:
            labels = 1 - a["total_initial_cost"] / b["total_initial_cost"]
            search = 1 - a["total_search_contexts"] / b["total_search_contexts"]
            reliable = a["accuracy"] >= b["accuracy"] - .02 and a["brier"] <= b["brier"] + .01 and a["log_loss"] <= b["log_loss"] + .02
            targets.append({"target": "validated-transfer", "family": family, "total_label_reduction": labels,
                "total_search_reduction": search, "reliability_guardrail": reliable,
                "met_in_sample": max(labels, search) >= .2 and reliable})
    summary["target_assessments"] = targets
    write_json(root / "summary.json", summary)
    with (root / "summary.csv").open("w", newline="") as stream:
        if aggregates:
            writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
            writer.writeheader(); writer.writerows(aggregates)
    def fmt(x):
        return "—" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
    html = ["<!doctype html><meta charset='utf-8'><title>Active symbolic learning</title><style>body{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 24px;color:#172b35}table{border-collapse:collapse;font-size:13px;margin:24px 0}td,th{padding:7px;border-bottom:1px solid #cdd9df;text-align:left}th{background:#eaf1f5}img{max-width:100%}p{max-width:1000px;line-height:1.55}</style>",
        "<h1>Active symbolic learning and concept transfer</h1>",
        f"<p>{summary['task_runs']:,} task runs; {summary['acquired_answers']:,} newly acquired answers; {summary['processed_answers']:,} processed answers including validation reuse. {summary['checkpoints']:,} checkpoints verified. {summary['partial_blocks']} partial blocks excluded from paired analysis.</p>",
        "<p>These experiments compare probability updates, question selection, changing rules, and reusable relational templates. Every symbolic proof is conditional on its proposed program. The experiments do not establish general intelligence, unrestricted concept invention, or the benefits of embeddings.</p>",
        "<p>EPIG uses only the permitted teaching-input distribution. Long evaluation paths remain outside learning and question selection. World rules and feedback noise are evaluator-owned. Candidate generation uses a fixed 10% noise ranking heuristic in every new condition; the probability updater integrates its separate noise grid. The resulting probabilities are conditional on the bounded searched bank, not an exact posterior over all programs.</p>",
        "<p>The change model retains at most 32 run-length branches and records discarded mass. Stochastic-model weight cannot distinguish all random processes from missing explanations. Capped costs retain failures at their available horizon; recovery fractions must be read alongside costs. Confidence plots use expected correctness for stochastic controls.</p>",
        "<h2>Measured comparisons</h2>"]
    html.append("<p>Research targets are assessed below on the completed sample. Passing a numerical target is not a statistical guarantee; all paired intervals and failures are retained. Transfer targets include common source and validation acquisition costs, including shadow search. Reliability guardrails allow at most 0.02 accuracy loss, 0.01 Brier increase, and 0.02 nats log-loss increase.</p>")
    html.append(table(["Target", "Condition", "Met in this sample", "Measurements"], [[t["target"], t.get("variant", t.get("regime", t.get("family", ""))), t["met_in_sample"], json.dumps({k: v for k, v in t.items() if k not in {"target", "variant", "regime", "family", "met_in_sample"}}, sort_keys=True)] for t in targets]))
    selected = [a for a in aggregates if a["role"] in {"primary", "target"}]
    html.append(table(["Role", "World", "Regime", "Variant", "Independent worlds", "Accuracy", "Brier", "Initial cost", "Recovery cost", "Recovered", "CPU s/task", "Budget stops"],
        [[a["role"], a["family"], a["regime"], a["variant"], a["worlds"], fmt(a["accuracy"]), fmt(a["brier"]), fmt(a["initial_cost"]), fmt(a["recovery_cost"]),
          f"{a['recovered']}/{a['transitions']}", fmt(a["cpu_seconds"]), a["budget_exhausted"]] for a in selected]))
    html.append("<h2>Paired effects</h2><p>Differences are candidate minus baseline. Lower error, cost, and CPU are better. Intervals are descriptive cluster-bootstrap intervals, with 5,000 resamples; fewer than eight independent worlds is exploratory. Multiple reported contrasts are not a familywise significance claim.</p>")
    html.append(table(["World", "Regime", "Candidate / baseline", "Metric", "Difference", "95% interval", "Worlds"],
        [[p["family"], p["regime"], p["candidate"] + " / " + p["baseline"], p["metric"], fmt(p["difference"]), f"[{fmt(p['low'])}, {fmt(p['high'])}]", p["worlds"]]
         for p in paired if p["metric"] in {"accuracy", "initial_cost", "recovery_cost", "brier"}]))
    html.append(f"<h2>Concept evidence</h2><p>{summary['concept_records']} concept records across repeated runs; {summary['promoted_concept_records']} passed the two-task predictive gate. Counts are not counts of globally distinct discoveries. Exact and typed compression conditions use provisional definitions. The validated condition requires positive fresh gain in each of two later tasks and total gain above ln(20). Source programs can be wrong.</p>")
    html.append("<p>Three source tasks and two validation tasks are charged to each transfer condition. Validation shadows reuse already-acquired answers; their computation is included. For conservative comparison, the reported prior CPU charge includes the shared validation work. Source/validation questions are uniform and shared; target questions are active. Each template has at most eight typed instantiations per schema. Template search uses no hidden predicate-name alignment.</p>")
    examples = [r for r in records if r["role"] == "target" and r["imported_concepts"]][:6]
    for r in examples:
        html.append(f"<p>{escape(r['run_id'])}; final accuracy {fmt(r['final_accuracy'])}</p><pre>{escape(r['program_text'])}</pre>")
    html.append("<h2>Research basis and limits</h2><p><a href='https://proceedings.mlr.press/v206/bickfordsmith23a.html'>EPIG</a> motivates prediction-oriented questions; <a href='https://arxiv.org/abs/0710.3742'>Bayesian change-point detection</a> motivates uncertain regime duration; <a href='https://www.pldi21.org/poster_pldi.355.html'>DreamCoder</a> motivates reusable structure; <a href='https://ojs.aaai.org/index.php/AAAI/article/view/28925'>noisy MDL program learning</a> motivates noise-aware evidence. This is a controlled synthesis, not a reproduction or new-algorithm claim. Description savings are an atom-count proxy, not measured compressed bits. All hypotheses use supplied positive-rule productions.</p>")
    html.append("<h2>Figures</h2><div id='figures'></div><p><a href='summary.json'>Full numeric results</a> · <a href='summary.csv'>CSV</a> · <a href='selection.json'>Timing-based sample sizes</a> · <a href='config.json'>Configuration</a> · <a href='manifest.json'>Evidence manifest</a></p>")
    html.append(f"<p>Worker CPU: {summary['cpu_seconds']:.1f}s; largest process RSS: {summary['largest_process_rss_bytes'] / 2**20:.1f} MiB. RSS is not summed simultaneous memory. Energy was not measured. Old experiment artifacts are retained separately.</p>")
    (root / "REPORT.html").write_text("\n".join(html))
    figures = plot(root, records, manifest)
    content = (root / "REPORT.html").read_text().replace("<div id='figures'></div>", ''.join(f"<p><img src='{name}.png' alt='{name}'><br><a href='{name}.svg'>Download SVG</a></p>" for name in figures))
    (root / "REPORT.html").write_text(content)
    for name in ("summary.json", "summary.csv", "REPORT.html", *[f"{f}.{ext}" for f in figures for ext in ("png", "svg")]):
        manifest["outputs"][name] = file_hash(root / name)
    write_json(root / "manifest.json", manifest)
    return root / "REPORT.html"


def plot(root, records, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "active-study", "font.size": 9})
    index = {r["run_id"]: r for r in records}
    curves, reliability = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: np.zeros((3, 10)))
    for b in manifest["blocks"]:
        if b["status"] != "complete":
            continue
        with gzip.open(root / b["path"] / "trace.jsonl.gz", "rt") as stream:
            for line in stream:
                t = json.loads(line)
                r = index[t["run_id"]]
                if r["role"] == "primary" and r["family"] in {"mechanism", "pair", "relay"} and r["variant"] in {"broad-upfront", "eig-window", "epig-window", "eig-change", "epig-change"}:
                    curves[(r["family"], r["regime"], r["variant"])][t["sequence"]].append(t["balanced_accuracy"])
                if r["role"] == "primary" and r["family"] == "coin" and t["sequence"] > 32:
                    reliability[r["variant"]] += np.array([t["confidence_counts"], t["confidence_sum"], t["correct_sum"]])
    names = []
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(root / f"{name}.png", dpi=140)
        fig.savefig(root / f"{name}.svg", metadata={"Date": None})
        plt.close(fig); names.append(name)
    if curves:
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
        for i, family in enumerate(("mechanism", "pair", "relay")):
            for j, regime in enumerate(("stable", "abc", "aba")):
                ax = axes[i, j]
                for variant in ("broad-upfront", "eig-window", "epig-window", "eig-change", "epig-change"):
                    data = curves.get((family, regime, variant), {})
                    if data:
                        x = sorted(data); ax.plot(x, [np.mean(data[t]) for t in x], label=variant, linewidth=1.3)
                ax.set(title=f"{family} / {regime}", ylim=(0, 1), xlabel="Teaching answers", ylabel="Held-out balanced accuracy")
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        save(fig, "learning-curves")
    if reliability:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([.5, 1], [.5, 1], "k--", label="Calibrated")
        for variant in ("broad-upfront", "eig-window", "epig-change", "gate-sequential"):
            if variant in reliability:
                a = reliability[variant]; valid = a[0] > 0
                ax.plot(a[1, valid] / a[0, valid], a[2, valid] / a[0, valid], "o-", label=variant)
        ax.set(xlabel="Reported confidence", ylabel="Expected correctness", title="Random-coin control after 32 answers", xlim=(.49, 1.01), ylim=(.45, 1.01))
        ax.legend(); ax.grid(alpha=.2); save(fig, "confidence")
    targets = [r for r in records if r["role"] == "target"]
    if targets:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        variants = ["library-none", "library-exact", "library-typed", "library-validated"]
        for ax, field, label in zip(axes, ("initial_cost", "search_contexts"), ("Capped answers to threshold", "Program–scene search evaluations")):
            ax.bar([v.replace("library-", "") for v in variants], [average([r[field] for r in targets if r["variant"] == v]) or 0 for v in variants])
            ax.set(ylabel=label, title="Transfer targets"); ax.tick_params(axis="x", rotation=20)
        save(fig, "concept-transfer")
    return names
