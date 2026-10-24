"""Restore a learner, policy, and concept library from verified on-disk evidence."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_lab.common import digest, json_lines
from research_lab.discovery.experiment import verify
from research_lab.discovery.learner import DiscoveryLearner
from research_lab.discovery.programs import Scene, Program
from research_lab.discovery.worlds import PublicTask
from research_lab.learning.components import SQLiteMemory
from research_lab.learning.types import create


def restore(directory, run_id):
    directory = Path(directory)
    verify(directory)
    config = json.loads((directory / "config.json").read_text())
    record = next((r for r in json_lines(directory / "runs.jsonl") if r["run_id"] == run_id), None)
    if record is None:
        raise ValueError("unknown task run")
    with gzip.open(directory / "worlds.json.gz", "rt") as stream:
        worlds = json.load(stream)
    tasks = [t for c in worlds["curricula"] for t in c["tasks"]] + worlds["diagnostics"]
    task = next(t for t in tasks if t["identifier"] == record["task"])
    public = PublicTask({name: tuple(types) for name, types in task["public"]["schema"].items()},
                        tuple(Scene.from_dict(s) for s in task["public"]["questions"]),
                        tuple(Scene.from_dict(s) for s in task["public"]["evaluation"]))
    connection = sqlite3.connect(f"file:{directory.resolve()}/memory.sqlite?mode=ro", uri=True)
    try:
        # Read only: avoid the schema-creating SQLiteMemory constructor.
        import zlib
        row = connection.execute("SELECT state FROM checkpoints WHERE run_id=?", (run_id,)).fetchone()
        state = json.loads(zlib.decompress(row[0]))
    finally:
        connection.close()
    library = create("discovery-library", config["components"]["library"])
    library.restore(state["library"])
    # Mining at the task boundary may have just added helpers. The learner used
    # the earlier immutable prefix, whereas the next task can use the whole library.
    ordered = list(library.definitions.items())
    available = None
    for count in range(len(ordered) + 1):
        prefix = dict(ordered[:count])
        fingerprint = digest({name: Program.make((clause,)).to_dict() for name, clause in prefix.items()})
        if fingerprint == state["learner"]["library_sha256"]:
            available = prefix
            break
    if available is None:
        raise ValueError("checkpoint lacks the learner's original library version")
    learner = DiscoveryLearner(public, config, record["variant"], available, initialize_broad=False)
    learner.restore(state["learner"])
    kind = "random" if record["variant"] == "library-random" else "progress" if record["variant"] == "library-progress" else "information"
    policy = create("discovery-policy", config["components"]["policy"], kind=kind, seed=0, exploration=config["exploration"])
    policy.restore(state["policy"])
    return learner, policy, library, record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    learner, policy, library, record = restore(args.run, args.run_id)
    question = policy.choose(learner.probabilities(), learner.public.questions, learner.noise)
    print(json.dumps({"run_id": record["run_id"], "sequence": learner.count,
                      "next_question": asdict(question), "predicted_probability": float(learner.probabilities()[question.index]),
                      "uncertain": learner.uncertain, "confirmation_pending": learner.pending is not None,
                      "library_used_by_saved_learner": list(learner.library),
                      "library_available_for_next_task": list(library.definitions)}, indent=2))
