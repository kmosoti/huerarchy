from __future__ import annotations

from collections import defaultdict
import csv
import gzip
from html import escape
import json
from pathlib import Path

import numpy as np

from ..common import file_hash, json_lines, write_json
from .experiment import verify_manifest


def paired_comparisons(records):
    index = {(r["world"], r["noise"], r["seed"], r["learner"], r["policy"]): r for r in records}
    results = []
    rng = np.random.default_rng(1789)
    for kind in ("stable", "changing"):
        for learner in ("full-history", "window", "drift-reset"):
            by_world = defaultdict(list)
            for record in records:
                if record["world_kind"] != kind or record["learner"] != learner or record["policy"] != "information-gain":
                    continue
                partner = index.get((record["world"], record["noise"], record["seed"], learner, "random"))
                if partner:
                    by_world[record["world"]].append(record["balanced_accuracy_auc"] - partner["balanced_accuracy_auc"])
            if not by_world:
                continue
            differences = np.array([np.mean(v) for v in by_world.values()])
            bootstrap = differences[rng.integers(len(differences), size=(2000, len(differences)))].mean(axis=1)
            results.append({"world_kind": kind, "learner": learner, "worlds": len(differences),
                            "accuracy_auc_difference": float(np.mean(differences)),
                            "bootstrap_95_low": float(np.percentile(bootstrap, 2.5)),
                            "bootstrap_95_high": float(np.percentile(bootstrap, 97.5))})
    return results


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"


def curve_svg(curves, steps, changes):
    width, height, left, top = 800, 300, 50, 20
    plot_w, plot_h = 730, 220
    colors = ["#2563eb", "#d97706", "#16a34a", "#9333ea", "#dc2626", "#0891b2", "#475569", "#be185d"]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Held-out balanced accuracy learning curves">']
    for value in (0, 0.25, 0.5, 0.75, 1):
        y = top + (1 - value) * plot_h
        parts.append(f'<path d="M{left},{y}h{plot_w}" stroke="#ddd"/><text x="5" y="{y + 4}">{value:.2f}</text>')
    for change in changes:
        x = left + change / steps * plot_w
        parts.append(f'<path d="M{x},{top}v{plot_h}" stroke="#999" stroke-dasharray="4 4"/>')
    for index, (name, values) in enumerate(curves.items()):
        points = " ".join(f"{left + (i + 1) / steps * plot_w:.2f},{top + (1 - v) * plot_h:.2f}" for i, v in enumerate(values))
        color = colors[index % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        parts.append(f'<text x="{left + (index % 2) * 370}" y="{260 + (index // 2) * 17}" fill="{color}">{escape(name)}</text>')
    parts.append(f'<text x="{width - 120}" y="{top + plot_h + 15}">{steps} questions</text></svg>')
    return "".join(parts)


def render(directory):
    directory = Path(directory)
    manifest = verify_manifest(directory)
    config = json.loads((directory / "config.json").read_text())
    records = list(json_lines(directory / "runs.jsonl"))
    if len(records) != manifest["runs"] or not all(r["checkpoint_verified"] for r in records):
        raise ValueError("run count or checkpoint verification mismatch")
    groups = defaultdict(list)
    for record in records:
        groups[(record["world_kind"], record["noise"], record["learner"], record["policy"])].append(record)
    aggregates = []
    for (kind, noise, learner, policy), items in sorted(groups.items()):
        completed = [recovery["confirmed_recovery_delay"] for r in items for recovery in r["recovery"]
                     if recovery["confirmed_recovery_delay"] is not None]
        opportunities = sum(len(r["recovery"]) for r in items)
        row = {"world_kind": kind, "noise": noise, "learner": learner, "policy": policy, "runs": len(items)}
        for field in ("balanced_accuracy_auc", "final_balanced_accuracy", "heldout_brier", "heldout_log_loss",
                      "confident_coverage", "observed_prequential_log_loss", "unique_questions", "cpu_seconds", "wall_seconds"):
            row[field] = float(np.mean([r[field] for r in items]))
        row.update(alarms_per_run=float(np.mean([len(r["alarms"]) for r in items])),
                   recovered=len(completed), recovery_opportunities=opportunities,
                   recovery_median_queries=float(np.median(completed)) if completed else None)
        covered = sum(r["confident_coverage"] for r in items)
        row["confident_error_rate"] = (sum((r["confident_error_rate"] or 0) * r["confident_coverage"]
                                          for r in items) / covered) if covered else None
        aggregates.append(row)
    pairs = paired_comparisons([r for r in records if r["world_kind"] != "diagnostic"])
    findings = []
    for kind in ("stable", "changing"):
        for learner, policy in (("full-history", "information-gain"), ("drift-reset", "information-gain"),
                                ("window", "information-gain")):
            selected = [r for r in records if (r["world_kind"], r["learner"], r["policy"]) == (kind, learner, policy)]
            if selected:
                findings.append(f"{kind.capitalize()} worlds: {learner} with informative questions averaged "
                                f"{np.mean([r['balanced_accuracy_auc'] for r in selected]):.1%} held-out balanced accuracy "
                                "over the teaching trajectory, averaging the configured noise levels and seeds.")
    reset_stable = [r for r in records if r["world_kind"] == "stable" and r["learner"] == "drift-reset" and r["policy"] == "information-gain"]
    if reset_stable:
        findings.append(f"The informative drift-reset learner averaged {np.mean([len(r['alarms']) for r in reset_stable]):.2f} "
                        "false change alarms per stable run. Adaptation has a measurable retention cost.")
    summary = {"schema": 1, "runs": manifest["runs"], "teaching_turns": manifest["teaching_turns"],
               "config": config, "aggregates": aggregates, "curiosity_paired_comparisons": pairs,
               "elapsed_seconds": manifest["elapsed_seconds"], "cpu_seconds": manifest["cpu_seconds"],
               "peak_rss_bytes": manifest["peak_rss_bytes"], "energy_joules": None,
               "checkpoint_verified_runs": sum(r["checkpoint_verified"] for r in records),
               "proof_checks": manifest["proof_checks"], "findings": findings}
    write_json(directory / "summary.json", summary)
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    sections = ["<h1>Online rule learning and curiosity</h1>",
                f"<p>{manifest['runs']:,} runs; {manifest['teaching_turns']:,} teaching turns; "
                f"{manifest['proof_checks']} conditional reasoner checks. All final component checkpoints reproduced predictions and the next question.</p>",
                "<p>This is a structured-world experiment. A supplied vocabulary and conjunction hypothesis family are its inductive biases. "
                "It does not demonstrate language acquisition, concept discovery, neural training, or human-like intrinsic motivation.</p>",
                "<h2>Measured findings</h2><ul>" + "".join(f"<li>{escape(f)}</li>" for f in findings) + "</ul>",
                "<h2>What was tested</h2>",
                f"<p>Each world has {config['dimensions']} binary attributes. Hidden targets have one to three signed conditions; "
                f"changing worlds switch rules after questions {escape(str(config['changes']))}. Feedback flip rates are "
                f"{escape(str(config['noise_rates']))}; every learner assumes {config['assumed_noise']:.0%} noise. "
                "The clean target, actual flip events, and phase boundaries remain evaluator-only. "
                "Question and evaluation pools are disjoint random halves, resampled to contain both outcomes in each phase.</p>",
                "<p>All variants receive the same label budget. Corresponding runs share world, split, and Bernoulli flip schedule. "
                "Repeated questions receive fresh noisy responses. Memory-only is an exact example lookup with Beta(1,1) smoothing; "
                "unseen examples remain at probability 0.5. Frozen keeps its original rule prior. "
                "Information gain uses the learner's posterior and includes random exploration.</p>",
                "<h2>Rule learning and adaptation</h2>",
                "<p>Accuracy AUC is mean post-update held-out balanced accuracy over all turns; final accuracy averages the last 20 turns. "
                "Brier scores assess latent clean outcomes (lower is better). Teacher log loss is scored before the answer is learned, "
                "against the noisy response. Those are different quantities.</p>"]
    primary = [a for a in aggregates if a["world_kind"] != "diagnostic"]
    sections.append(table(["World", "Noise", "Learner / policy", "Accuracy AUC", "Final accuracy", "Brier", "Alarms/run", "Recovery"], [
        [a["world_kind"], f"{a['noise']:.0%}", a["learner"] + " / " + a["policy"], f"{a['balanced_accuracy_auc']:.3f}",
         f"{a['final_balanced_accuracy']:.3f}", f"{a['heldout_brier']:.3f}", f"{a['alarms_per_run']:.2f}",
         f"{a['recovered']}/{a['recovery_opportunities']}" + (f"; median {a['recovery_median_queries']:g} questions" if a["recovery_median_queries"] is not None else "")]
        for a in primary]))
    sections.append("<p>Recovery requires ten consecutive turns at held-out balanced accuracy ≥0.90. "
                    "Delay ends when that window is confirmed. Unrecovered transitions are right-censored at the next change or run end; "
                    "the displayed median includes recovered transitions only. In stable worlds, every detector alarm is a false change alarm.</p>")
    sections.append("<h2>Does question selection help?</h2><p>Paired information-gain minus random differences in accuracy AUC. "
                    "Seed and noise repeats are averaged within each world; percentile bootstrap intervals resample worlds (2,000 draws). "
                    "Few worlds and synthetic rules limit generalization. Intervals crossing zero do not establish an improvement.</p>")
    sections.append(table(["World", "Learner", "Worlds", "Difference", "95% interval"], [
        [p["world_kind"], p["learner"], p["worlds"], f"{p['accuracy_auc_difference']:+.4f}",
         f"[{p['bootstrap_95_low']:+.4f}, {p['bootstrap_95_high']:+.4f}]"] for p in pairs]))
    # Stream raw traces: retain curve sums, never load every teaching turn into memory.
    run_lookup = {r["run_id"]: r for r in records}
    sums, counts = {}, defaultdict(int)
    selected = {("full-history", "random"), ("window", "random"), ("drift-reset", "random"), ("drift-reset", "information-gain")}
    with gzip.open(directory / "trace.jsonl.gz", "rt") as stream:
        trace_count = 0
        for line in stream:
            trace = json.loads(line)
            record = run_lookup[trace["run_id"]]
            trace_count += 1
            if record["world_kind"] == "diagnostic" or (record["learner"], record["policy"]) not in selected:
                continue
            key = (record["world_kind"], record["noise"], record["learner"] + " / " + record["policy"])
            if key not in sums:
                sums[key] = np.zeros(config["steps"])
            sums[key][trace["sequence"] - 1] += trace["balanced_accuracy"]
            if trace["sequence"] == 1:
                counts[key] += 1
    if trace_count != manifest["teaching_turns"]:
        raise ValueError("trace count mismatch")
    sections.append("<h2>Learning curves</h2>")
    for kind in ("stable", "changing"):
        for noise in config["noise_rates"]:
            curves = {key[2]: values / counts[key] for key, values in sums.items() if key[:2] == (kind, noise)}
            if curves:
                sections.append(f"<h3>{kind.capitalize()}, {noise:.0%} feedback errors</h3>" + curve_svg(curves, config["steps"], config["changes"] if kind == "changing" else []))
    diagnostic = [a for a in aggregates if a["world_kind"] == "diagnostic"]
    if diagnostic:
        sections.append("<h2>Outside the hypothesis family</h2><p>The separate XOR world cannot be expressed by one conjunction. "
                        "It is excluded from primary comparisons. Poor accuracy, excessive confidence, or repeated alarms here are failure evidence; "
                        "resetting cannot add a missing rule family.</p>")
        sections.append(table(["Noise", "Learner / policy", "Accuracy AUC", "Brier", "Alarms/run", "Confident coverage", "Error when confident"], [
            [f"{a['noise']:.0%}", a["learner"] + " / " + a["policy"], f"{a['balanced_accuracy_auc']:.3f}",
             f"{a['heldout_brier']:.3f}", f"{a['alarms_per_run']:.2f}", f"{a['confident_coverage']:.3f}",
             f"{a['confident_error_rate']:.3f}" if a["confident_error_rate"] is not None else "—"] for a in diagnostic]))
    sections.extend(["<h2>Resources and reproduction</h2>",
                     f"<p>Experiment wall time {manifest['elapsed_seconds']:.1f}s; process CPU time {manifest['cpu_seconds']:.1f}s; "
                     f"controller peak RSS {manifest['peak_rss_bytes'] / 2**20:.1f} MiB. RSS covers the entire run; "
                     "CPU time excludes Rust child processes. No energy measurement is available. "
                     "Per-run timings include teaching, scoring, event storage, and occasional proof checks; learner and policy timings are separate in raw results.</p>",
                     "<p>Observations and checkpoints are in memory.sqlite; evaluator metrics are separate. "
                     "Conditional proof explanations never certify learned hypotheses. "
                     "All components are single-process local implementations; the language bridge is a later plugin.</p>",
                     "<p>Reproduce with <code>python -m research_lab learning run --config config.json --output NEW_DIRECTORY</code>.</p>",
                     "<p>" + " · ".join(f'<a href="{name}">{name}</a>' for name in (
                         "config.json", "worlds.json", "summary.json", "summary.csv", "runs.jsonl", "trace.jsonl.gz", "memory.sqlite", "explanations.jsonl", "manifest.json")) + "</p>"])
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Online learning experiment</title><style>body{font:16px/1.5 system-ui;margin:2rem auto;padding:0 1rem;max-width:1200px;color:#172033}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border:1px solid #ddd;text-align:left}thead{background:#edf2f7}svg{max-width:100%;font:12px system-ui}h2{margin-top:2rem}code{overflow-wrap:anywhere}</style>' + "".join(sections) + "</html>"
    (directory / "REPORT.html").write_text(html)
    for name in ("summary.json", "summary.csv", "REPORT.html"):
        manifest["outputs"][name] = file_hash(directory / name)
    # Rendering may be repeated without rerunning the experiment. Preserve original
    # experiment source hashes and identify the renderer that produced these outputs.
    manifest["render_sources"] = {"research_lab/learning/report.py": file_hash(Path(__file__))}
    write_json(directory / "manifest.json", manifest)
    return directory / "REPORT.html"
