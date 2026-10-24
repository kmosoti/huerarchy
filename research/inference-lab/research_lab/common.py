from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODEL_REVISION = "5f1c2fd19f37ca653a3d719c8ee6cc1895f4b64f"
BITNET_REVISION = "0b341e582afbf9e1011f24744b554c96a3477eb5"
BITNET_SUBMODULE = "390c307752ab78fd8189f359d6954c9ba1be74af"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def json_lines(path):
    with open(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def emit(stream, record):
    stream.write(canonical(record).decode() + "\n")
    stream.flush()


def percentiles(values):
    import numpy as np
    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)} if values else {}


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def run_checked(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def profile_environment():
    import importlib.metadata
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    flags = next((x.split(":", 1)[1].split() for x in Path("/proc/cpuinfo").read_text().splitlines()
                  if x.startswith("flags")), [])
    cpu_model = next((x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                      if x.startswith("model name")), None)
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
    connection.close()
    rust = ROOT.parent / "scratch/inference-cargo/bin/rustc"
    env = dict(os.environ, CARGO_HOME=str(ROOT.parent / "scratch/inference-cargo"),
               RUSTUP_HOME=str(ROOT.parent / "scratch/inference-rustup"))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "python_executable": sys.executable,
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: None)(),
        "gil_note": "Runtime API reports the actual GIL state" if hasattr(sys, "_is_gil_enabled") else "API absent; conventional CPython build",
        "platform": platform.platform(), "cpu_model": cpu_model, "cpu_flags": flags,
        "visible_cpus": os.cpu_count(), "cpu_max": read("/sys/fs/cgroup/cpu.max"),
        "memory_max": read("/sys/fs/cgroup/memory.max"),
        "gpu_devices": [str(x) for x in Path("/dev").glob("nvidia*")],
        "sqlite": sqlite3.sqlite_version, "fts5": True,
        "rust": subprocess.check_output([str(rust), "--version"], env=env, text=True).strip() if rust.exists() else None,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "cbor2", "zstandard", "cmake", "ninja")},
        "energy_joules": None, "energy_reason": "No exposed RAPL/powercap energy sensor; CPU time is not energy",
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "bitnet_revision": BITNET_REVISION, "bitnet_llama_revision": BITNET_SUBMODULE,
        "model_revision": MODEL_REVISION,
    }
