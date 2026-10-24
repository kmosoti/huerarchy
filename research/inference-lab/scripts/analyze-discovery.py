"""Postprocess verified traces into learning/recovery costs and failure summaries."""
import argparse
from collections import defaultdict
import gzip
from html import escape
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from research_lab.common import file_hash, json_lines, write_json
from research_lab.discovery.experiment import verify
from research_lab.learning.report import table


def delay(values, begin, end, consecutive=10):
    for offset in range(begin, end - consecutive + 1):
        if min(values[offset:offset + consecutive]) >= 0.9:
            return offset + consecutive - begin
    return None


def analyze(directory):
    directory = Path(directory)
    manifest = verify(directory)
    summary = json.loads((directory / "summary.json").read_text())
    config = summary["config"]
    records = {r["run_id"]: r for r in json_lines(directory / "runs.jsonl")}
    details = []
    def finish(run_id, values, uncertain, alarm_episodes, alarm_positive_responses):
        if run_id is None:
            return
        record = records[run_id]
        if record["alarms"] != alarm_positive_responses:
            raise ValueError("alarm trace and per-task counter disagree")
        changes = config["changes"] if record["changing"] else []
        first_end = changes[0] if changes else config["steps"]
        initial = delay(values, 0, first_end)
        recovery = [{"after_response": boundary, "delay": delay(values, boundary, end),
                     "censoring_horizon": end - boundary}
                    for boundary, end in zip(changes, changes[1:] + [config["steps"]])]
        details.append({"run_id": run_id, "initial_learning_delay": initial,
                        "total_acquisition_labels_at_initial_success": record["cumulative_acquisition_labels"] - config["steps"] + initial if initial is not None else None,
                        "initial_censoring_horizon": first_end, "recovery": recovery,
                        "uncertainty_fraction": uncertain / len(values), "alarm_episodes": alarm_episodes,
                        "alarm_positive_responses": alarm_positive_responses})
    current, values, uncertain = None, [], 0
    alarm_episodes = alarm_positive_responses = 0
    previous_alarm = False
    with gzip.open(directory / "trace.jsonl.gz", "rt") as stream:
        for line in stream:
            trace = json.loads(line)
            if trace["run_id"] != current:
                finish(current, values, uncertain, alarm_episodes, alarm_positive_responses)
                current, values, uncertain = trace["run_id"], [], 0
                alarm_episodes = alarm_positive_responses = 0
                previous_alarm = False
            values.append(trace["balanced_accuracy"])
            uncertain += trace["uncertain"]
            alarm_episodes += bool(trace["alarm"]) and not previous_alarm
            alarm_positive_responses += bool(trace["alarm"])
            previous_alarm = bool(trace["alarm"])
    finish(current, values, uncertain, alarm_episodes, alarm_positive_responses)
    if len(details) != len(records):
        raise ValueError("noncontiguous or incomplete task traces")
    groups = defaultdict(list)
    for d in details:
        record = records[d["run_id"]]
        groups[(record["family"], record["changing"], record["variant"])].append((record, d))
    aggregates = []
    for (family, changing, variant), items in sorted(groups.items()):
        initial = [d["initial_learning_delay"] for _, d in items if d["initial_learning_delay"] is not None]
        transitions = [t for _, d in items for t in d["recovery"]]
        recovered = [t["delay"] for t in transitions if t["delay"] is not None]
        aggregates.append({"family": family, "changing": changing, "variant": variant, "runs": len(items),
                           "initial_successes": len(initial), "initial_median_responses": float(np.median(initial)) if initial else None,
                           "recovered": len(recovered), "transitions": len(transitions),
                           "recovery_median_responses": float(np.median(recovered)) if recovered else None,
                           "mean_alarm_episodes": float(np.mean([d["alarm_episodes"] for _, d in items])),
                           "mean_alarm_positive_responses": float(np.mean([d["alarm_positive_responses"] for _, d in items])),
                           "mean_uncertainty_fraction": float(np.mean([d["uncertainty_fraction"] for _, d in items]))})
    findings = []
    paired = summary["paired_comparisons"]
    stable_relay = next((p for p in paired if (p["family"], p["changing"], p["candidate"], p["baseline"]) == ("relay", False, "adaptive", "small-reset")), None)
    if stable_relay:
        findings.append(f"On stable relay tasks, adaptive expansion improved mean trajectory balanced accuracy over the small reset learner "
                        f"by {100*stable_relay['difference']:.1f} percentage points. This supports the value of a richer explanation language in these cases.")
    transfer = next((p for p in paired if (p["family"], p["changing"], p["candidate"], p["baseline"]) == ("relay", False, "adaptive-library", "adaptive")), None)
    if transfer:
        findings.append(f"Adding the concept library changed stable-relay accuracy by {100*transfer['difference']:+.2f} percentage points "
                        f"(curriculum bootstrap interval {100*transfer['low']:+.2f} to {100*transfer['high']:+.2f}). "
                        "Definition creation and reuse therefore do not establish an accuracy advantage here.")
    primary = [a for a in summary["aggregates"] if a["family"] in {"mechanism", "pair", "relay"}]
    worlds = {(a["family"], a["changing"]) for a in primary}
    broad_wins = sum(max([a for a in primary if (a["family"], a["changing"]) == key], key=lambda a: a["balanced_accuracy_auc"])["variant"] == "broad-upfront" for key in worlds)
    findings.append(f"The broad-upfront baseline had the highest mean trajectory accuracy in {broad_wins}/{len(worlds)} primary family/change groups. "
                    "Its rich language and immediate candidate admission are a stronger package than the tested delayed-growth strategy; this is not an exhaustive-search oracle comparison.")
    relay_change = {a["variant"]: a for a in primary if a["family"] == "relay" and a["changing"]}
    if {"adaptive-library", "broad-upfront"}.issubset(relay_change):
        a, b = relay_change["adaptive-library"], relay_change["broad-upfront"]
        findings.append(f"On changing relays, adaptive expansion with reuse averaged {a['balanced_accuracy_auc']:.1%} trajectory accuracy "
                        f"and {a['final_balanced_accuracy']:.1%} final accuracy, versus {b['balanced_accuracy_auc']:.1%} and {b['final_balanced_accuracy']:.1%} for broad-upfront. "
                        "This is a substantial adaptation failure of the tested configuration. A 32-response confirmation block is long relative to an 80-response phase; "
                        "the experiment does not isolate that delay from candidate search, recent-evidence fitting, or other causes.")
    coin_growth = sum(r["growth_promotions"] for r in records.values() if r["family"] == "coin")
    coin_runs = sum(r["family"] == "coin" for r in records.values())
    findings.append(f"Purely random controls produced {coin_growth} accepted growth events across {coin_runs} runs. "
                    "Fresh-response testing and a complexity cost reduce opportunities to overfit but do not certify discoveries.")
    analysis = {"schema": 1, "success_definition": "Ten consecutive held-out balanced-accuracy values at least 0.90; delay ends when confirmed",
                "censoring": "No success before the next target change or task end is right-censored; medians include successes only",
                "findings": findings, "aggregates": aggregates, "per_run": details}
    write_json(directory / "adaptation.json", analysis)
    summary["adaptation"] = {key: value for key, value in analysis.items() if key != "per_run"}
    write_json(directory / "summary.json", summary)
    section = '<section id="adaptation-analysis"><h2>What the comparison establishes</h2><ul>' + "".join(f"<li>{escape(f)}</li>" for f in findings) + "</ul>"
    section += "<p>Helpers are mined from final most-probable source programs, which may be wrong. Compression preserves their meaning; "
    section += "later predictive evidence must establish usefulness. Source-program selection itself is a limitation to study.</p>"
    section += "<h3>Learning and recovery costs</h3><p>Success requires ten consecutive responses at held-out balanced accuracy ≥0.90. "
    section += "The delay ends when success is confirmed. Unrecovered transitions are censored at the next change or task end. "
    section += "Medians cover successes only, so they must be read alongside the success counts. An alarm episode starts when the detector flag rises from false to true. "
    section += "The raw per-task 'alarms' counter instead counts responses with a positive flag, which can remain latched during confirmation or in the broad baseline. "
    section += "These are not counts of distinct real-world changes. Episodes in unchanged worlds can indicate noise or an inadequate model.</p>"
    section += table(["World", "Changes", "Condition", "Initial successes", "Median responses", "Recoveries", "Median recovery", "Alarm episodes/task"], [
        [a["family"], a["changing"], a["variant"], f"{a['initial_successes']}/{a['runs']}",
         a["initial_median_responses"] if a["initial_median_responses"] is not None else "—",
         f"{a['recovered']}/{a['transitions']}", a["recovery_median_responses"] if a["recovery_median_responses"] is not None else "—",
         f"{a['mean_alarm_episodes']:.2f}"] for a in aggregates if a["family"] in {"mechanism", "pair", "relay"}])
    section += '<p><a href="adaptation.json">Per-run learning delays and censored recovery records</a>.</p></section>'
    html_path = directory / "REPORT.html"
    html = html_path.read_text()
    html = re.sub(r'<section id="adaptation-analysis">.*?</section>', "", html, flags=re.S)
    html = html.replace("<h2>Measured findings</h2>", section + "<h2>Measured findings</h2>", 1)
    html_path.write_text(html)
    for name in ("adaptation.json", "summary.json", "REPORT.html"):
        manifest["outputs"][name] = file_hash(directory / name)
    manifest["analysis_sources"] = {"scripts/analyze-discovery.py": file_hash(Path(__file__))}
    write_json(directory / "manifest.json", manifest)
    print("\n".join(findings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    analyze(parser.parse_args().run)
