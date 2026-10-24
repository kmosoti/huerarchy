"""Independent curriculum workers; no shared learned state or asynchronous updates."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
import multiprocessing
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from ..common import ROOT, digest, file_hash, peak_rss_bytes, write_json
from ..learning.components import SQLiteMemory


def _partition(config, output, indices, diagnostics):
    from .experiment import run
    return str(run(config, output, _indices=indices, _include_diagnostics=diagnostics, _render=False))


def run_parallel(config, output):
    from .experiment import verify
    from .report import render
    output = Path(output or ROOT / "results/discovery" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="discovery-partitions-", dir=ROOT.parent / "scratch"))
    write_json(output / "manifest.json", {"schema": 1, "status": "running", "partition_workspace": str(work)})
    start = time.perf_counter()
    jobs = [(f"curriculum-{i:03d}", [i], False) for i in range(config["curricula"])]
    if config["diagnostics"]:
        jobs.insert(0, ("diagnostics", [], True))
    with ProcessPoolExecutor(max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = {executor.submit(_partition, config, work / name, indices, diagnostics): name for name, indices, diagnostics in jobs}
        for future in as_completed(pending):
            future.result()
            print(f"discovery partition complete: {pending[future]}", flush=True)
    directories = sorted((work / name for name, _, _ in jobs), key=lambda p: p.name == "diagnostics")
    manifests = [verify(directory) for directory in directories]
    first = manifests[0]
    if any(m["sources"] != first["sources"] or m["config_sha256"] != digest(config) for m in manifests):
        raise ValueError("parallel partitions have mixed source or configuration hashes")
    for name in ("config.json", "environment.json", "research.json"):
        shutil.copyfile(directories[0] / name, output / name)
    worlds = {"schema": 1, "curricula": [], "diagnostics": []}
    for directory in directories:
        with gzip.open(directory / "worlds.json.gz", "rt") as stream:
            part = json.load(stream)
            worlds["curricula"].extend(part["curricula"])
            worlds["diagnostics"].extend(part["diagnostics"])
    worlds["curricula"].sort(key=lambda c: c["id"])
    with gzip.open(output / "worlds.json.gz", "wt") as stream:
        json.dump(worlds, stream)
    for name in ("runs.jsonl", "concepts.jsonl", "explanations.jsonl"):
        with (output / name).open("wb") as destination:
            for directory in directories:
                with (directory / name).open("rb") as source:
                    shutil.copyfileobj(source, destination)
    for name in ("trace.jsonl.gz", "events.jsonl.gz"):
        # Concatenated gzip members are a valid streaming gzip file.
        with (output / name).open("wb") as destination:
            for directory in directories:
                with (directory / name).open("rb") as source:
                    shutil.copyfileobj(source, destination)
    connection = sqlite3.connect(output / "memory.sqlite")
    SQLiteMemory(connection, "schema-only")
    connection.commit()
    try:
        for directory in directories:
            connection.execute("ATTACH DATABASE ? AS part", (str(directory / "memory.sqlite"),))
            connection.execute("INSERT INTO events SELECT * FROM part.events")
            connection.execute("INSERT INTO checkpoints SELECT * FROM part.checkpoints")
            connection.commit()
            connection.execute("DETACH DATABASE part")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("merged experience memory failed integrity check")
    finally:
        connection.close()
    write_json(output / "partition-manifests.json", {directory.name: manifest for directory, manifest in zip(directories, manifests)})
    manifest = {**first, "status": "complete", "task_runs": sum(m["task_runs"] for m in manifests),
                "teaching_turns": sum(m["teaching_turns"] for m in manifests), "rust_proofs": sum(m["rust_proofs"] for m in manifests),
                "elapsed_seconds": time.perf_counter() - start, "cpu_seconds": sum(m["cpu_seconds"] for m in manifests),
                "peak_rss_bytes": max([peak_rss_bytes(), *[m["peak_rss_bytes"] for m in manifests]]),
                "resource_scope": "CPU sums worker teaching loops; RSS is the largest individual worker/controller peak, not simultaneous aggregate RSS",
                "workers": config["workers"], "outputs": {}}
    names = ("config.json", "worlds.json.gz", "runs.jsonl", "trace.jsonl.gz", "events.jsonl.gz", "memory.sqlite",
             "concepts.jsonl", "explanations.jsonl", "research.json", "environment.json", "partition-manifests.json")
    manifest["outputs"] = {name: file_hash(output / name) for name in names}
    write_json(output / "manifest.json", manifest)
    render(output)
    shutil.rmtree(work)
    return output
