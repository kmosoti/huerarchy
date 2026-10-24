from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time

from ..common import ROOT, digest, file_hash, profile_environment, write_json
from .experiment import run_core_block, run_transfer_block, verify_block, source_hashes
from .types import configuration


def choose_count(counts, cpu_per_block, seconds, workers, reserve):
    affordable = [n for n in counts if cpu_per_block * n / workers <= seconds * (1 - reserve)]
    return max(affordable) if affordable else min(counts)


def _job(kind, output, block, config, deadline):
    return (run_core_block if kind == "core" else run_transfer_block)(output, block, config, deadline)


def _phase(root, kind, count, config, deadline):
    results = []
    # Scheduling order is fixed and independent of outcomes; each block contains every paired condition.
    with ProcessPoolExecutor(max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(_job, kind, str(root / "blocks" / f"{kind}-{block:03d}"), block, config, deadline): block for block in range(count)}
        for future in as_completed(pending):
            block = pending[future]
            result = future.result()
            results.append({"path": f"blocks/{kind}-{block:03d}", "kind": kind, "block": block, **result})
            print(f"active {kind}: block {block + 1}/{count} {result['status']}; {result['tasks']} tasks; {result['wall_seconds']:.1f}s", flush=True)
    return sorted(results, key=lambda r: r["block"])


def run(config=None, output=None, smoke=False):
    c = configuration(config)
    if smoke:
        c = configuration({**c, "steps": 36, "source_steps": 48, "validation_steps": 24, "validation_warmup": 12,
                           "scene_count": 12, "noise_rates": [.1], "seeds": [0], "world_counts": [1], "workers": 1})
    root = Path(output or ROOT / "results/active" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if root.exists():
        raise ValueError("study output must be new")
    root.mkdir(parents=True)
    (root / "blocks").mkdir()
    write_json(root / "config.json", c)
    write_json(root / "environment.json", profile_environment())
    manifest = {"schema": 1, "status": "running", "smoke": smoke, "sources": source_hashes(), "config_sha256": digest(c),
                "started": datetime.now(timezone.utc).isoformat(), "outputs": {}, "blocks": []}
    write_json(root / "manifest.json", manifest)
    started = time.time()
    hard_end = started + c["study_seconds"]
    if smoke:
        selection = {"core": 1, "transfer": 1, "pilot": "smoke; no timing pilot"}
    else:
        pilot = configuration({**c, "fixture_seed": c["fixture_seed"] + 7919, "noise_rates": [.1], "seeds": [0], "workers": 1})
        (root / "pilot").mkdir()
        pilot_deadline = min(started + c["pilot_seconds"], hard_end - c["report_seconds"])
        print("active: timing pilot on separate development fixtures", flush=True)
        # Independent pilot blocks run together to keep the wall-time allocation meaningful.
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = {kind: pool.submit(_job, kind, str(root / "pilot" / kind), 0, pilot, pilot_deadline) for kind in ("core", "transfer")}
            timings = {kind: future.result() for kind, future in jobs.items()}
        write_json(root / "pilot" / "config.json", pilot)
        multiplier = len(c["noise_rates"]) * len(c["seeds"])
        selection = {"pilot": timings, "selection_uses": "elapsed CPU and completion only; no evaluation metric",
                     "reserve": c["timing_reserve"]}
        for kind in ("core", "transfer"):
            measured = timings[kind]
            estimate = measured["cpu_seconds"] * multiplier * (2 if kind == "transfer" else 1)
            if measured["status"] != "complete":
                estimate = float("inf")
            selection[kind] = choose_count(c["world_counts"], estimate, c[kind + "_seconds"], c["workers"], c["timing_reserve"])
            selection[kind + "_estimated_cpu_seconds_per_block"] = estimate if estimate != float("inf") else None
    write_json(root / "selection.json", selection)
    print(f"active: frozen study sizes: core={selection['core']}, transfer={selection['transfer']}", flush=True)
    for kind in ("core", "transfer"):
        deadline = min(time.time() + c[kind + "_seconds"], hard_end - c["report_seconds"])
        if time.time() >= deadline:
            break
        manifest["blocks"].extend(_phase(root, kind, selection[kind], c, deadline))
        write_json(root / "manifest.json", manifest)
    manifest.update(status="complete", elapsed_seconds=time.time() - started, energy_joules=None,
                    budget_seconds=c["study_seconds"], ended=datetime.now(timezone.utc).isoformat(),
                    resource_scope="CPU sums full worker blocks including restore checks; RSS is largest process peak, not concurrent aggregate")
    if source_hashes() != manifest["sources"]:
        raise ValueError("study source hashes changed")
    manifest["outputs"] = {name: file_hash(root / name) for name in ("config.json", "environment.json", "selection.json")}
    write_json(root / "manifest.json", manifest)
    from .report import render
    render(root)
    return root


def verify(root):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    if m.get("schema") != 1 or m.get("status") != "complete":
        raise ValueError("study is not complete")
    for name, sha in m["outputs"].items():
        if Path(name).name != name or file_hash(root / name) != sha:
            raise ValueError(f"changed study artifact: {name}")
    for b in m["blocks"]:
        if Path(b["path"]).parts != ("blocks", f"{b['kind']}-{b['block']:03d}"):
            raise ValueError("invalid block path")
        actual = verify_block(root / b["path"])
        if any(actual[k] != b[k] for k in ("status", "outputs", "sources", "config_sha256")):
            raise ValueError("changed block manifest")
    selection = json.loads((root / "selection.json").read_text())
    if isinstance(selection.get("pilot"), dict):
        pilot_config = json.loads((root / "pilot" / "config.json").read_text())
        for kind, expected in selection["pilot"].items():
            if kind not in {"core", "transfer"}:
                raise ValueError("invalid pilot kind")
            actual = verify_block(root / "pilot" / kind)
            if actual != expected or actual["config_sha256"] != digest(pilot_config):
                raise ValueError("pilot evidence changed")
    return m
