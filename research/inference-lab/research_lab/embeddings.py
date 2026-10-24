from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import BITNET_REVISION, MODEL_REVISION, RESULTS, ROOT, file_hash, write_json


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("non-finite or zero embedding")
    return vectors / norms


def quantize(vectors):
    scale = np.maximum(np.max(np.abs(vectors), axis=1), np.finfo(np.float32).tiny) / 127
    codes = np.clip(np.rint(vectors / scale[:, None]), -127, 127).astype(np.int8)
    return codes, scale.astype(np.float32)


def metrics(ranking, relevant):
    relevant = set(relevant)
    hits = [int(int(node) in relevant) for node in ranking]
    recall = sum(hits) / len(relevant) if relevant else 1.0
    dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), len(ranking))))
    return recall, float(dcg / ideal) if ideal else 1.0


def top_k(scores, k=10):
    # Stable ID tie-break, including quantization-induced ties.
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores)))[:k].tolist()


def reciprocal_rank_fusion(a, b, k=10):
    scores = {}
    for ranking in (a, b):
        for rank, node in enumerate(ranking):
            scores[node] = scores.get(node, 0.0) + 1 / (60 + rank + 1)
    return sorted(scores, key=lambda node: (-scores[node], node))[:k]


class EmbeddingServer:
    """Native CPU model, loopback HTTP only. No remote inference service."""

    def __init__(self, build="native", threads=4):
        self.build = build
        self.threads = threads
        self.directory = ROOT.parent / "scratch/BitNet" / f"build-{build}"
        self.model = ROOT / "models/bitnet-embedding-270m-i2_s.gguf"
        self.process = None
        self.log = None
        self.peak_rss = 0

    def request(self, route, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=180)
        try:
            connection.request("GET" if body is None else "POST", route,
                               None if body is None else json.dumps(body),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read()
            if response.status != 200:
                raise RuntimeError(f"embedding HTTP {response.status}: {raw[:700]!r}")
            return json.loads(raw)
        finally:
            connection.close()

    def sample(self):
        if self.process is None:
            return {}
        try:
            values = {}
            for line in Path(f"/proc/{self.process.pid}/status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, number, _ = line.split()
                    values[key.rstrip(":")] = int(number) * 1024
            self.peak_rss = max(self.peak_rss, values.get("VmHWM", 0))
            fields = Path(f"/proc/{self.process.pid}/stat").read_text().split(") ", 1)[1].split()
            values["cpu_seconds"] = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
            return values
        except OSError:
            return {}

    def __enter__(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        RESULTS.mkdir(exist_ok=True)
        self.log = (RESULTS / f"embedding-server-{self.build}.log").open("a")
        command = [str(self.directory / "bin/llama-server"), "-m", str(self.model),
                   "--host", "127.0.0.1", "--port", str(self.port), "--embedding",
                   "--pooling", "last", "-c", "2048", "-b", "512", "-ub", "512",
                   "-np", "1", "-t", str(self.threads), "-tb", str(self.threads), "--no-warmup"]
        self.log.write("\nresearch_lab native server session: " + json.dumps(command) + "\n")
        self.log.flush()
        start = time.perf_counter()
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 60
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"native embedding server exited; see embedding-server-{self.build}.log")
                try:
                    self.request("/health")
                    self.startup_seconds = time.perf_counter() - start
                    self.sample()
                    return self
                except (OSError, RuntimeError):
                    time.sleep(0.1)
            raise TimeoutError("embedding server startup exceeded 60 seconds")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def encode(self, texts, *, kind="passage", batch_size=16, progress=None):
        output = []
        batch_metrics = []
        for start in range(0, len(texts), batch_size):
            batch = [f"{kind}: {text}" for text in texts[start:start + batch_size]]
            before = self.sample()
            tick = time.perf_counter_ns()
            response = self.request("/v1/embeddings", {"input": batch, "encoding_format": "float"})
            elapsed = time.perf_counter_ns() - tick
            items = sorted(response["data"], key=lambda item: item["index"])
            if len(items) != len(batch):
                raise ValueError("embedding result cardinality mismatch")
            output.extend(item["embedding"] for item in items)
            after = self.sample()
            batch_metrics.append({"items": len(batch), "elapsed_ns": elapsed,
                                  "cpu_seconds": after.get("cpu_seconds", 0) - before.get("cpu_seconds", 0)})
            if progress:
                progress(min(start + batch_size, len(texts)), len(texts))
        vectors = normalize(output)
        if vectors.shape != (len(texts), 640):
            raise ValueError(f"expected 640-dimensional vectors, got {vectors.shape}")
        return vectors, batch_metrics

    def __exit__(self, *_):
        if self.process is not None and self.process.poll() is None:
            self.sample()
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log is not None:
            self.log.close()


def model_manifest():
    path = ROOT / "models/bitnet-embedding-270m-i2_s.gguf"
    sha256 = file_hash(path)
    if sha256 != "8ee5ae971b103cd55758934be54e5c9f7cc2b58b15890615acce8e649988c751":
        raise ValueError("model artifact does not match the inspected pinned revision")
    return {"model": "microsoft/bitnet-embedding-270m", "revision": MODEL_REVISION,
            "bitnet_revision": BITNET_REVISION, "bytes": path.stat().st_size,
            "sha256": sha256, "dimension": 640, "format": "I2_S",
            "prefixes": {"document": "passage: ", "query": "query: "},
            "pooling": "last", "normalization": "L2", "threads": 4}
