"""Export standard scientific figures from verified raw discovery measurements."""
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research_lab.common import file_hash, json_lines, write_json
from research_lab.discovery.experiment import verify


def export(directory):
    directory = Path(directory)
    manifest = verify(directory)
    summary = json.loads((directory / "summary.json").read_text())
    config = summary["config"]
    lookup = {r["run_id"]: r for r in json_lines(directory / "runs.jsonl")}
    curves, counts = {}, defaultdict(int)
    with gzip.open(directory / "trace.jsonl.gz", "rt") as stream:
        for line in stream:
            trace = json.loads(line)
            record = lookup[trace["run_id"]]
            if record["diagnostic"]:
                continue
            key = (record["family"], record["changing"], record["variant"])
            if key not in curves:
                curves[key] = np.zeros(config["steps"])
            curves[key][trace["sequence"] - 1] += trace["balanced_accuracy"]
            counts[key] += trace["sequence"] == 1
    plt.rcParams.update({"svg.hashsalt": "discovery-figures-v1", "font.size": 10})
    images, artifacts = [], []
    def save(fig, name):
        for suffix in ("svg", "png"):
            filename = name + "." + suffix
            metadata = {"Date": None} if suffix == "svg" else {"Software": "matplotlib " + matplotlib.__version__}
            fig.savefig(directory / filename, dpi=150, bbox_inches="tight", metadata=metadata)
            artifacts.append(filename)
        plt.close(fig)
    for family in ("mechanism", "pair", "relay"):
        for changing in (False, True):
            selected = {key[2]: values / counts[key] for key, values in curves.items() if key[:2] == (family, changing)}
            if not selected:
                continue
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for variant in config["variants"]:
                if variant in selected:
                    ax.plot(np.arange(1, config["steps"] + 1), selected[variant], label=variant, linewidth=1.4)
            if changing:
                for boundary in config["changes"]:
                    ax.axvline(boundary + 0.5, color="0.45", linestyle="--", linewidth=0.8)
            ax.set(xlabel="Teaching responses", ylabel="Held-out balanced accuracy", ylim=(0, 1.02),
                   title=f"{family.capitalize()} workshop — {'changing' if changing else 'stable'}")
            ax.grid(alpha=0.2)
            ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=8)
            filename = f"curve-{family}-{'changing' if changing else 'stable'}"
            save(fig, filename)
            images.append(filename + ".svg")
    pairs = summary["paired_comparisons"]
    if pairs:
        fig, ax = plt.subplots(figsize=(11, max(4, len(pairs) * 0.34)))
        labels = []
        for index, p in enumerate(pairs):
            ax.plot([p["low"], p["high"]], [index, index], color="#2563eb", linewidth=1.2)
            ax.plot(p["difference"], index, "o", color="#2563eb", markersize=4)
            labels.append(f"{p['family']} / {'changing' if p['changing'] else 'stable'}: {p['candidate']} − {p['baseline']}")
        ax.set_yticks(range(len(pairs)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="0.4", linestyle="--", linewidth=0.8)
        ax.set(xlabel="Difference in mean held-out balanced accuracy", title="Paired comparisons with curriculum-level bootstrap intervals")
        ax.grid(axis="x", alpha=0.2)
        save(fig, "paired-comparisons")
    html_path = directory / "REPORT.html"
    document = html_path.read_text()
    inline = re.findall(r"<svg\b.*?</svg>", document, flags=re.S)
    if inline:
        if len(inline) != len(images):
            raise ValueError("figure count differs from report curves")
        replacements = iter(images)
        document = re.sub(r"<svg\b.*?</svg>", lambda _: '<img style="max-width:100%" src="' + next(replacements) + '" alt="Held-out learning curve">', document, flags=re.S)
    elif not all(f'src="{name}"' in document for name in images):
        raise ValueError("report has an unrecognized figure layout")
    document = re.sub(r'<section id="exported-figures">.*?</section>', "", document, flags=re.S)
    links = " · ".join(f'<a href="{escape(name)}">{escape(name)}</a>' for name in artifacts)
    footer = '<section id="exported-figures"><h2>Exportable figures</h2><p>Generated from verified raw measurements with Matplotlib ' + matplotlib.__version__ + '.</p><p>' + links + '</p></section>'
    document = document.replace("</html>", footer + "</html>")
    html_path.write_text(document)
    write_json(directory / "plots-manifest.json", {"matplotlib": matplotlib.__version__,
               "script_sha256": file_hash(Path(__file__)), "requirements_sha256": file_hash(ROOT / "requirements.lock"),
               "inputs": {name: file_hash(directory / name) for name in ("runs.jsonl", "trace.jsonl.gz", "summary.json")},
               "figures": {name: file_hash(directory / name) for name in artifacts}})
    for name in artifacts + ["plots-manifest.json", "REPORT.html"]:
        manifest["outputs"][name] = file_hash(directory / name)
    manifest["plot_sources"] = {"scripts/plot-discovery.py": file_hash(Path(__file__))}
    write_json(directory / "manifest.json", manifest)
    print(f"Verified and exported {len(artifacts)} figure files to {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    export(parser.parse_args().run)
