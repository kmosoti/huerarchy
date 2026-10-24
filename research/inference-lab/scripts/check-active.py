"""Validate a completed study's evidence, database counts, checkpoint predictions, and local report links."""
import argparse
import gzip
from html.parser import HTMLParser
import json
from pathlib import Path
import sqlite3
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_lab.active.study import verify
from research_lab.active.experiment import learner_for, next_query, source_hashes
from research_lab.active.worlds import OracleTask
from research_lab.discovery.programs import Program, BatchEvaluator, rust_audit
from research_lab.common import file_hash, write_json


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in {"href", "src"} and value)


def check(root):
    root = Path(root)
    manifest = verify(root)
    assert source_hashes() == manifest["sources"], "current experimental sources differ from the recorded run"
    totals = {"runs": 0, "observations": 0, "checkpoints": 0, "conditional_proofs": 0}
    selected, previews = set(), []
    for block in manifest["blocks"]:
        directory = root / block["path"]
        records = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
        connection = sqlite3.connect(f"file:{directory / 'memory.sqlite'}?mode=ro", uri=True)
        try:
            assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            # A queued block can reach its deadline before its first task starts.
            assert {"events", "checkpoints"} <= tables or (not records and block["status"] == "partial")
            observations = connection.execute("SELECT count(*) FROM events WHERE kind='observation'").fetchone()[0] if "events" in tables else 0
            checkpoints = connection.execute("SELECT count(*) FROM checkpoints").fetchone()[0] if "checkpoints" in tables else 0
            assert observations == sum(r["answers"] for r in records)
            assert checkpoints == len(records)
            totals["runs"] += len(records); totals["observations"] += observations; totals["checkpoints"] += checkpoints
            totals["conditional_proofs"] += block["proofs"]
            with gzip.open(directory / "worlds.json.gz", "rt") as stream:
                worlds = json.load(stream)
            for r in records:
                variant = r["variant"]
                if variant not in {"broad-upfront", "gate-sequential", "epig-change", "epig-change-archive", "library-typed", "library-validated"} or variant in selected:
                    continue
                state = json.loads(zlib.decompress(connection.execute("SELECT state FROM checkpoints WHERE run_id=?", (r["run_id"],)).fetchone()[0]))
                task = OracleTask.from_dict(worlds[state["task"]])
                definitions = {name: Program.from_dict(value).clauses[0] for name, value in state["library"].items()}
                obj, policy = learner_for(task, state["config"], variant, 0, definitions, state["transfer"], False, state["learner"])
                obj.restore(state["learner"])
                if policy: policy.restore(state["policy"])
                best, mass = obj.best()
                fresh = BatchEvaluator(task.view.questions, task.view.schema, definitions).evaluate(best)
                np.testing.assert_array_equal(fresh, obj.engine.cache[best.identifier])
                audit = rust_audit(best, task.evaluation[0], task.view.schema, definitions)
                question, reason = next_query(obj, policy)
                previews.append({"run_id": r["run_id"], "next_question": question, "selection": reason,
                                 "program_mass": mass, "fresh_prediction_cache_matches": True,
                                 "rust_python_closure_matches": True, "conditional_prediction": audit["outcome"],
                                 "teacher_answers_acquired": 0, "library_instances": len(definitions)})
                selected.add(variant)
        finally:
            connection.close()
    parser = Links(); parser.feed((root / "REPORT.html").read_text())
    for link in parser.links:
        if not link.startswith(("https:", "http:", "#")):
            assert (root / link.split("#")[0]).exists(), link
    result = {"schema": 1, **totals, "restored_examples": previews, "local_report_links_checked": len(parser.links),
              "artifact_hashes_verified": True, "experimental_sources_match": True,
              "database_integrity_verified": True, "rust_reasoner_sha256": file_hash(ROOT / "target/release/bench")}
    write_json(root / "verification.json", result)
    manifest["outputs"]["verification.json"] = file_hash(root / "verification.json")
    manifest.setdefault("postprocessing_sources", {})[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    write_json(root / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in result.items() if k != "restored_examples"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    check(parser.parse_args().run)
