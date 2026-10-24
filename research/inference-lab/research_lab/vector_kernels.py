"""Matched single-thread Rust FP32/INT8 search with no full INT8 decoding."""
from __future__ import annotations

import json
import sqlite3
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import DATA, RESULTS, ROOT, digest, emit, file_hash, json_lines, percentiles, profile_environment, write_json
from .data import lexical_search
from .embeddings import metrics, model_manifest, quantize, reciprocal_rank_fusion


def write_matrix(path, array):
    array = np.asarray(array)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError("matrix must have two nonempty dimensions")
    if array.dtype == np.float32:
        if not np.isfinite(array).all():
            raise ValueError("non-finite vector")
        kind = 0
        array = np.ascontiguousarray(array, dtype="<f4")
    elif array.dtype == np.int8:
        kind = 1
        array = np.ascontiguousarray(array)
    else:
        raise TypeError("matrix dtype must be float32 or int8")
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<4sIII", b"VEC1", kind, *array.shape))
        stream.write(array.tobytes())
    temporary.replace(path)


def load_verified_cache(directory):
    cache = directory / "embeddings-native.npz"
    metadata = directory / "embeddings-native.json"
    expected = digest({"documents": file_hash(directory / "documents.jsonl"),
                       "queries": file_hash(directory / "queries.jsonl"), "model": model_manifest(), "build": "native",
                       "runtime_hashes": {name: file_hash(ROOT.parent / "scratch/BitNet/build-native/bin" / name)
                                          for name in ("llama-server", "libllama.so", "libggml-cpu.so")}})
    if not cache.exists() or not metadata.exists() or json.loads(metadata.read_text()).get("fingerprint") != expected:
        raise RuntimeError("matching native embedding cache required; run the retrieval suite first")
    with np.load(cache, allow_pickle=False) as loaded:
        docs, queries = loaded["documents"], loaded["queries"]
    expected_docs = sum(1 for _ in json_lines(directory / "documents.jsonl"))
    expected_queries = sum(q["family"] == "retrieval" for q in json_lines(directory / "queries.jsonl"))
    for array, count in ((docs, expected_docs), (queries, expected_queries)):
        if array.shape != (count, 640) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("embedding cache shape/dtype/values invalid")
        if not np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-5):
            raise ValueError("embedding cache is not L2-normalized")
    return docs, queries, {"cache_sha256": file_hash(cache), "cache_fingerprint": expected}


def reference_scores(corpus, queries, selected):
    # The FP64 oracle lives in the controller, never the measured Rust worker.
    values = corpus.astype(np.float64)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    questions = queries[selected].astype(np.float64)
    questions /= np.linalg.norm(questions, axis=1, keepdims=True)
    scores = questions @ values.T
    return {int(index): row for index, row in zip(selected, scores)}


def check_matches(rows, references, tolerance=2e-6):
    maximum_error = 0.0
    tied_boundary_differences = 0
    for row in rows:
        if row["query"] not in references:
            continue
        expected = references[row["query"]]
        ids = row["ids"]
        scores = np.asarray(row["scores"])
        if len(set(ids)) != len(ids) or not np.isfinite(scores).all():
            raise AssertionError("invalid native top-k result")
        error = float(np.max(np.abs(scores - expected[ids])))
        maximum_error = max(error, maximum_error)
        if error > tolerance:
            raise AssertionError(f"native cosine differs from FP64 reference: {error}")
        top = np.lexsort((np.arange(len(expected)), -expected))[:len(ids)]
        missing = set(map(int, top)) - set(ids)
        for node in missing:
            if expected[node] > min(scores) + tolerance:
                raise AssertionError("native top-k omitted a reference match outside rounding tolerance")
        tied_boundary_differences += len(missing)
    return maximum_error, tied_boundary_differences


def benchmark(repeats=5, directory=DATA):
    docs, query_vectors, inputs = load_verified_cache(directory)
    questions = [q for q in json_lines(directory / "queries.jsonl") if q["family"] == "retrieval"]
    codes, _ = quantize(docs)
    files = {"fp32": directory / "kernel-corpus-fp32.vec", "int8": directory / "kernel-corpus-int8.vec"}
    query_file = directory / "kernel-queries.vec"
    write_matrix(files["fp32"], docs)
    write_matrix(files["int8"], codes)
    write_matrix(query_file, query_vectors)
    selected = np.linspace(0, len(questions) - 1, min(32, len(questions)), dtype=int)
    reference = {"fp32": reference_scores(docs, query_vectors, selected),
                 "int8": reference_scores(codes, query_vectors, selected)}
    db = sqlite3.connect(directory / "events.sqlite")
    lexical = [lexical_search(db, q["text"], 50) for q in questions]
    db.close()
    validation = {"reference": "FP64 cosine and tie-aware top-50 membership", "score_tolerance": 2e-6,
                  "sampled_queries": list(map(int, selected)), "inputs": inputs,
                  "file_sha256": {name: file_hash(path) for name, path in dict(files, queries=query_file).items()},
                  "binary_sha256": file_hash(ROOT / "target/release/vector-search"), "variants": {},
                  "raw_results": {}, "environment": profile_environment()}
    with (RESULTS / "vector-kernels.jsonl").open("w") as stream:
        for repetition in range(repeats):
            for variant in (("fp32", "int8") if repetition % 2 == 0 else ("int8", "fp32")):
                raw = RESULTS / f"vector-kernel-{variant}-{repetition}.jsonl"
                start = time.perf_counter_ns()
                with raw.open("w") as destination:
                    subprocess.run([str(ROOT / "target/release/vector-search"), str(files[variant]), str(query_file), "50"],
                                   check=True, text=True, stdout=destination)
                process_ns = time.perf_counter_ns() - start
                records = list(json_lines(raw))
                summary = records.pop()
                if len(records) != len(questions) or not summary.get("summary"):
                    raise AssertionError("native worker output count mismatch")
                if [r["query"] for r in records] != list(range(len(questions))):
                    raise AssertionError("native worker query order mismatch")
                error, ties = check_matches(records, reference[variant])
                quality = []
                hybrid = []
                for row, question, lexical_ids in zip(records, questions, lexical):
                    if len(row["ids"]) != min(50, len(docs)):
                        raise AssertionError("native worker top-k cardinality mismatch")
                    quality.append(metrics(row["ids"][:10], question["expected"]))
                    hybrid.append(metrics(reciprocal_rank_fusion(lexical_ids, row["ids"]), question["expected"]))
                stats = validation["variants"].setdefault(variant, {"max_abs_score_error": 0.0, "total_tied_boundary_membership_differences": 0})
                stats["max_abs_score_error"] = max(stats["max_abs_score_error"], error)
                stats["total_tied_boundary_membership_differences"] += ties
                validation["raw_results"][raw.name] = file_hash(raw)
                emit(stream, {"suite": "vector_kernels", "variant": variant, "repetition": repetition,
                              "kernel": summary["kernel"], "threads": summary["threads"], "queries": len(questions),
                              "recall_at_10": float(np.mean([v[0] for v in quality])),
                              "ndcg_at_10": float(np.mean([v[1] for v in quality])),
                              "hybrid_recall_at_10": float(np.mean([v[0] for v in hybrid])),
                              "hybrid_ndcg_at_10": float(np.mean([v[1] for v in hybrid])),
                              "search_latency_ns": percentiles([r["elapsed_ns"] for r in records]),
                              "total_search_ns": summary["total_search_ns"], "process_ns": process_ns,
                              "query_loop_cpu_ns": summary["query_loop_cpu_ns"],
                              "query_loop_wall_ns": summary["query_loop_wall_ns"],
                              "load_ns": summary["load_ns"], "corpus_bytes": summary["corpus_bytes"],
                              "query_bytes": summary["query_bytes"], "loaded_rss_bytes": summary["loaded_rss_bytes"],
                              "peak_rss_bytes": summary["peak_rss_bytes"], "max_abs_score_error": error,
                              "scope": "isolated single-thread Rust worker; search includes normalization, dot products and top-50 heap; excludes load/output/query encoding",
                              "cpu_scope": "scheduled runtime of the single worker thread over search plus result formatting/output; excludes file loading",
                              "no_full_int8_decode": variant == "int8"})
                print(f"native search {variant} repetition {repetition+1}/{repeats}: "
                      f"{summary['total_search_ns']/1e9:.3f}s, peak RSS {summary['peak_rss_bytes']/1024**2:.2f} MiB", flush=True)
    write_json(RESULTS / "vector-kernel-validation.json", validation)
