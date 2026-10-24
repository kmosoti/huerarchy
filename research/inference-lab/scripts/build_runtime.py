"""Pinned upstream builds; no upstream source edits or driver changes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_lab.common import BITNET_REVISION, BITNET_SUBMODULE, MODEL_REVISION, file_hash, write_json

checkout = ROOT.parent / "scratch/BitNet"
results = ROOT / "results"
results.mkdir(exist_ok=True)


def command(args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


if not (checkout / ".git").exists():
    checkout.mkdir(parents=True, exist_ok=True)
    command(["git", "init", "-q", str(checkout)])
    command(["git", "-C", str(checkout), "remote", "add", "origin", "https://github.com/microsoft/BitNet.git"])
    command(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", BITNET_REVISION])
    command(["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"])
current = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
if current != BITNET_REVISION:
    raise SystemExit("Existing checkout has a different revision; preserve it and select a separate scratch directory")
command(["git", "-C", str(checkout), "submodule", "update", "--init", "--recursive", "--depth", "1"])
assert subprocess.check_output(["git", "-C", str(checkout / "3rdparty/llama.cpp"), "rev-parse", "HEAD"], text=True).strip() == BITNET_SUBMODULE
model = ROOT / "models/bitnet-embedding-270m-i2_s.gguf"
model.parent.mkdir(exist_ok=True)
if not model.exists():
    url = f"https://huggingface.co/microsoft/bitnet-embedding-270m/resolve/{MODEL_REVISION}/bitnet-embeddings-270m-bf16-i2_s.gguf"
    temporary = model.with_suffix(".download")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    temporary.replace(model)
assert file_hash(model) == "8ee5ae971b103cd55758934be54e5c9f7cc2b58b15890615acce8e649988c751"
metadata = {"bitnet_revision": BITNET_REVISION, "submodule_revision": BITNET_SUBMODULE,
            "model_revision": MODEL_REVISION, "model_sha256": file_hash(model), "builds": {}}
for build in ("native", "avx2"):
    directory = checkout / f"build-{build}"
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_OPENMP=OFF", "-DGGML_CCACHE=OFF",
             "-DLLAMA_BUILD_COMMON=ON", "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_EXAMPLES=ON",
             "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_UI=ON", "-DLLAMA_USE_PREBUILT_UI=OFF",
             "-DLLAMA_CURL=OFF", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", f"-DGGML_NATIVE={'ON' if build=='native' else 'OFF'}"]
    if build == "avx2":
        flags += [f"-DGGML_{f}=ON" for f in ("AVX", "AVX2", "BMI2", "FMA", "F16C")]
        flags += [f"-DGGML_{f}=OFF" for f in ("AVX_VNNI", "AVX512", "AVX512_VBMI", "AVX512_VNNI", "AVX512_BF16")]
    print(f"building {build} with four jobs", flush=True)
    with (results / f"cmake-{build}.log").open("w") as stream:
        command(["cmake", "-S", str(checkout), "-B", str(directory), "-G", "Ninja", *flags], stdout=stream, stderr=subprocess.STDOUT)
    with (results / f"build-{build}.log").open("w") as stream:
        command(["cmake", "--build", str(directory), "--target", "llama-server", "llama-embedding", "llama-bench", "-j4"], stdout=stream, stderr=subprocess.STDOUT)
    commands = json.loads((directory / "compile_commands.json").read_text())
    cpu_commands = [row["command"] for row in commands if "/ggml-cpu/" in row["file"]]
    if build == "avx2":
        assert all("-march=native" not in c and "-mavx512" not in c for c in cpu_commands)
        assert any("-mavx2" in c for c in cpu_commands)
    metadata["builds"][build] = {"cmake_flags": flags, "cpu_compile_commands": cpu_commands,
                                "server_sha256": file_hash(directory / "bin/llama-server")}
metadata["compiler"] = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
write_json(results / "build-manifest.json", metadata)
