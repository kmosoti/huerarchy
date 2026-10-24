"""Measure index construction on disposable copies, preserving authoritative inputs."""
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from .common import DATA, RESULTS, emit, file_hash, write_json


def index_build_benchmark(repeats=5, directory=DATA):
    source = directory / "events.sqlite"
    original_hash = file_hash(source)
    with (RESULTS / "index-build.jsonl").open("w") as stream, tempfile.TemporaryDirectory() as temporary:
        for repetition in range(repeats):
            target = Path(temporary) / "build.sqlite"
            tick = time.perf_counter_ns()
            shutil.copyfile(source, target)
            copy_ns = time.perf_counter_ns() - tick
            db = sqlite3.connect(target)
            db.executescript("DROP INDEX events_node; DROP INDEX nodes_group; DROP TABLE summaries;")
            stages = [
                ("secondary_indexes", "CREATE INDEX events_node ON events(node,id); CREATE INDEX nodes_group ON nodes(group_id);"),
                ("fts5", "CREATE VIRTUAL TABLE summaries USING fts5(text, tokenize='unicode61'); INSERT INTO summaries(rowid,text) SELECT node+1,summary FROM nodes ORDER BY node;"),
            ]
            for variant, sql in stages:
                cpu = time.process_time()
                tick = time.perf_counter_ns()
                db.executescript(sql)
                db.commit()
                elapsed_ns = time.perf_counter_ns() - tick
                emit(stream, {"suite": "index_build", "variant": variant, "repetition": repetition,
                              "elapsed_ns": elapsed_ns, "cpu_seconds": time.process_time() - cpu,
                              "source_copy_ns_excluded": copy_ns,
                              "scope": "create/populate/commit on disposable copied DB; source copy and dropping old indexes excluded; OS cache not flushed"})
            try:
                objects = {name: size for name, size in db.execute("SELECT name,SUM(pgsize) FROM dbstat GROUP BY name")}
                write_json(RESULTS / "sqlite-object-storage.json", {"objects_bytes": objects, "source_db_bytes": source.stat().st_size,
                           "note": "dbstat allocated pages by object on rebuilt copy; excludes external event fixture and model artifacts"})
            except sqlite3.OperationalError:
                write_json(RESULTS / "sqlite-object-storage.json", {"objects_bytes": None, "reason": "dbstat unavailable"})
            db.close()
    assert file_hash(source) == original_hash, "index benchmark changed authoritative input"
    print(f"index construction: {repeats} repetitions on copies; authoritative DB unchanged", flush=True)
