"""Bind each completed suite to its exact fixture and raw output hashes."""
import json
from pathlib import Path

from .common import DATA, RESULTS, file_hash, write_json

SUITES = {
    "exact": ["exact.jsonl", "verification.json"],
    "index-build": ["index-build.jsonl", "sqlite-object-storage.json"],
    "closure": ["closure.jsonl", "snapshot-closure.jsonl", "proof-example.json"],
    "communication": ["communication.jsonl"],
    "retrieval": ["retrieval.jsonl", "vector-storage.json", "model.json", "embedding-timing-native.jsonl"],
    "instruction-set": ["instruction-set.jsonl", "instruction-set-equivalence.json"],
    "vector-kernels": ["vector-kernels.jsonl", "vector-kernel-validation.json"],
}


def record_suites(names, directory=DATA):
    path = RESULTS / "suite-inputs.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    fixture_sha = file_hash(directory / "manifest.json")
    for name in names:
        outputs = {filename: file_hash(RESULTS / filename) for filename in SUITES[name]}
        if name == "vector-kernels":
            extra = json.loads((RESULTS / "vector-kernel-validation.json").read_text())["raw_results"]
            for filename, expected in extra.items():
                if Path(filename).name != filename or file_hash(RESULTS / filename) != expected:
                    raise RuntimeError(f"native per-query output failed verification: {filename}")
                outputs[filename] = expected
        records[name] = {"fixture_manifest_sha256": fixture_sha,
                         "outputs": outputs}
    write_json(path, records)


def verify_suites(directory=DATA):
    path = RESULTS / "suite-inputs.json"
    if not path.exists():
        raise RuntimeError("suite input manifest missing; complete the benchmark suites before rendering")
    records = json.loads(path.read_text())
    fixture_sha = file_hash(directory / "manifest.json")
    for name, filenames in SUITES.items():
        if name not in records or records[name]["fixture_manifest_sha256"] != fixture_sha:
            raise RuntimeError(f"{name} results belong to a different fixture or are missing")
        if not set(filenames).issubset(records[name]["outputs"]):
            raise RuntimeError(f"{name} is missing required output bindings")
        for filename, expected in records[name]["outputs"].items():
            if Path(filename).name != filename or expected != file_hash(RESULTS / filename):
                raise RuntimeError(f"{name} output changed or is incomplete: {filename}")
