from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import resource
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import DATA, RESULTS, ROOT, canonical, digest, emit, file_hash, json_lines, peak_rss_bytes, percentiles, write_json
from .data import adjacency, impacted_reference, lexical_search, sql_decision, sql_reachable
from .embeddings import EmbeddingServer, metrics, model_manifest, normalize, quantize, reciprocal_rank_fusion, top_k
from .transport import Replica, ResyncRequired, decode, delta, encode, snapshot, state_hash


def verify_projection(directory=DATA):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert file_hash(directory / "events.jsonl") == manifest["source_sha256"]
    assert file_hash(directory / "queries.jsonl") == manifest["queries_sha256"]
    states = np.full(manifest["nodes"], 2, dtype=np.int8)
    snapshots = np.load(directory / "snapshots.npy")
    snapshot_sources = np.load(directory / "snapshot-sources.npy")
    latest_event = np.full(manifest["nodes"], -1, dtype=np.int64)
    db = sqlite3.connect(directory / "events.sqlite")
    verified = 0
    for event_id, epoch, node, health, issue, raw, sha in db.execute("SELECT * FROM events ORDER BY id"):
        event = json.loads(raw)
        assert hashlib.sha256(raw).hexdigest() == sha
        assert event["id"] == event_id and event["node"] == node and event["epoch"] == epoch
        assert event["health"] == ("healthy", "unhealthy", "unknown")[health]
        states[node] = health
        latest_event[node] = event_id
        verified += 1
        if (event_id + 1) % manifest["events_per_epoch"] == 0 or event_id + 1 == manifest["events"]:
            np.testing.assert_array_equal(states, snapshots[epoch])
            np.testing.assert_array_equal(latest_event, snapshot_sources[epoch])
            sql_states = np.array([h for h, in db.execute("SELECT health FROM snapshots WHERE epoch=? ORDER BY node", (epoch,))], dtype=np.int8)
            np.testing.assert_array_equal(sql_states, states)
    db.close()
    return verified


def exact_benchmark(repeats=5, directory=DATA):
    verified_events = verify_projection(directory)
    queries = [q for q in json_lines(directory / "queries.jsonl") if q["family"] != "retrieval"]
    output = RESULTS / "exact.jsonl"
    with output.open("w") as stream:
        for repetition in range(repeats):
            db = sqlite3.connect(directory / "events.sqlite")
            cpu_start = time.process_time()
            times = {family: [] for family in ("reconstruct", "relational", "decision")}
            counts = dict.fromkeys(times, 0)
            for q in queries:
                start = time.perf_counter_ns()
                if q["family"] == "reconstruct":
                    raw, stored_sha = db.execute("SELECT payload,sha256 FROM events WHERE id=?", (q["event_id"],)).fetchone()
                    correct = hashlib.sha256(raw).hexdigest() == stored_sha == q["expected_sha256"]
                elif q["family"] == "relational":
                    correct = sql_reachable(db, q["node"], q["depth"]) == q["expected"]
                else:
                    correct = sql_decision(db, q["epoch"], q["node"]) == q["expected"]
                times[q["family"]].append(time.perf_counter_ns() - start)
                counts[q["family"]] += int(correct)
                if not correct:
                    raise AssertionError(f"exact query disagrees with independent reference: {q['id']}")
            cpu_seconds = time.process_time() - cpu_start
            db.close()
            for family, elapsed in times.items():
                emit(stream, {"suite": "exact", "variant": "sqlite", "family": family,
                              "repetition": repetition, "queries": len(elapsed), "correct": counts[family],
                              "latency_ns": percentiles(elapsed), "total_ns": sum(elapsed),
                              "process_peak_rss_bytes": peak_rss_bytes(), "all_families_cpu_seconds": cpu_seconds,
                              "cache_note": "new connection per repetition; OS cache not flushed"})
            print(f"exact repetition {repetition + 1}/{repeats}: {sum(counts.values())}/{len(queries)} correct", flush=True)
    write_json(RESULTS / "verification.json", {"events_replayed_and_verified": verified_events,
               "exact_queries_per_repetition": len(queries), "repetitions": repeats, "all_correct": True})


def verify_proofs(result, seeds, rules):
    facts = set(result["fact_ids"])
    proofs = dict(result["proofs"])
    assert facts == set(proofs)
    complete = set()
    active = set()
    def visit(fact):
        if fact in complete:
            return
        if fact in active:
            raise AssertionError("cyclic proof")
        active.add(fact)
        rule_id = proofs[fact]
        if rule_id is None:
            assert fact in seeds
        else:
            head, premises = rules[rule_id]
            assert head == fact
            for premise in premises:
                assert premise in facts
                visit(premise)
        active.remove(fact)
        complete.add(fact)
    for fact in facts:
        visit(fact)


def snapshot_closure(repeats=5, directory=DATA):
    snapshots = np.load(directory / "snapshots.npy")
    snapshot_sources = np.load(directory / "snapshot-sources.npy")
    nodes = snapshots.shape[1]
    graph = adjacency(nodes)
    rules = [(node * 4 + 3, [node * 4 + 1]) for node in range(nodes)]
    rules.extend((child * 4 + 3, [parent * 4 + 3]) for parent in graph for child in graph[parent])
    rule_text = "".join("r " + " ".join(map(str, [head, *premises])) + "\n" for head, premises in rules)
    program_path = directory / "snapshot-program.txt"
    with (RESULTS / "snapshot-closure.jsonl").open("w") as stream:
        for repetition in range(repeats):
            for epoch, states in enumerate(snapshots):
                seeds = {node * 4 + int(health) for node, health in enumerate(states)}
                expected_impacted = impacted_reference(graph, np.flatnonzero(states == 1))
                expected = seeds | {node * 4 + 3 for node in expected_impacted}
                tick = time.perf_counter_ns()
                program_path.write_text(rule_text + "".join(f"s {fact}\n" for fact in sorted(seeds)))
                output = subprocess.check_output([str(ROOT / "target/release/bench"), "infer", str(program_path)], text=True)
                elapsed = time.perf_counter_ns() - tick
                result = json.loads(output)
                assert set(result["fact_ids"]) == expected, (repetition, epoch)
                verify_proofs(result, seeds, rules)
                assert np.all((snapshot_sources[epoch] >= 0) | (states == 2))
                if repetition == 0 and epoch + 1 == len(snapshots):
                    target = max(expected_impacted) * 4 + 3 if expected_impacted else min(seeds)
                    proofs = dict(result["proofs"])
                    explanation = []
                    visited = set()
                    db = sqlite3.connect(directory / "events.sqlite")
                    def explain(fact):
                        if fact in visited:
                            return
                        visited.add(fact)
                        rule_id = proofs[fact]
                        if rule_id is None:
                            event_id = int(snapshot_sources[epoch, fact // 4])
                            source = db.execute("SELECT payload,sha256 FROM events WHERE id=?", (event_id,)).fetchone()
                            explanation.append({"fact_id": fact, "node": fact // 4, "kind": "snapshot_seed",
                                                "source_event": json.loads(source[0]) if source else None,
                                                "source_sha256": source[1] if source else None})
                        else:
                            _, premises = rules[rule_id]
                            for p in premises:
                                explain(p)
                            explanation.append({"fact_id": fact, "node": fact // 4, "kind": "derived_impact",
                                                "rule_id": rule_id, "premises": premises})
                    explain(target)
                    write_json(RESULTS / "proof-example.json", {"epoch": epoch, "target_fact": target, "proof_dag": explanation})
                    db.close()
                emit(stream, {"suite": "snapshot_closure", "variant": "worklist_proofs", "repetition": repetition,
                              "epoch": epoch, "facts": result["facts"], "rules": len(rules),
                              "engine_ns": result["elapsed_ns"], "end_to_end_ns": elapsed,
                              "dependency_events": result["dependency_events"], "proofs_verified": len(result["proofs"]),
                              "seed_provenance_references": len(seeds),
                              "correct": True})
            print(f"snapshot closure repetition {repetition + 1}/{repeats}: {len(snapshots)} snapshots and proofs verified", flush=True)


def communication_benchmark(repeats=5, directory=DATA):
    snapshots = np.load(directory / "snapshots.npy")
    states = [{i: int(h) for i, h in enumerate(row)} for row in snapshots]
    context = json.loads((directory / "context.json").read_text())
    context_hash = digest(context)
    allowed = set(range(len(states[0])))
    with (RESULTS / "communication.jsonl").open("w") as stream:
        for codec, incremental in (("json", False), ("cbor", False), ("cbor", True)):
            variant = f"{codec}_{'delta' if incremental else 'full'}"
            for compressed in (False, True):
                for fault_rate in (0.0, 0.01, 0.05, 0.20):
                    for repetition in range(repeats):
                        replica = Replica(context_hash, allowed_ids=allowed)
                        bootstrap = encode(context, codec, compressed)
                        assert digest(decode(bootstrap, codec, compressed)) == context_hash
                        total_bytes = len(bootstrap)
                        fallback_bytes = 0
                        fallbacks = 0
                        duplicate_checks = 0
                        latency = []
                        sequence_faults = random.Random(42)
                        selected = set(sequence_faults.sample(range(1, len(states)), round((len(states) - 1) * fault_rate)))
                        faults = ("stale_context", "missing_base", "unknown_id", "out_of_order", "duplicate", "drop")
                        for epoch, after in enumerate(states):
                            packet = delta(states[epoch - 1], after, context_hash, epoch) if incremental and epoch else snapshot(after, context_hash, epoch)
                            fault = faults[epoch % len(faults)] if epoch in selected else None
                            if fault == "stale_context": replica.context = "stale"
                            elif fault == "missing_base": replica.state.pop(epoch % len(after), None)
                            elif fault == "unknown_id":
                                if packet["kind"] == "delta": packet["updates"].append([len(after) + 1, 1])
                                else:
                                    packet["state"] = dict(packet["state"])
                                    packet["state"][len(after) + 1] = 1
                            elif fault == "out_of_order": packet["sequence"] += 1
                            tick = time.perf_counter_ns()
                            payload = encode(packet, codec, compressed)
                            total_bytes += len(payload)
                            try:
                                if fault != "drop":
                                    replica.apply(decode(payload, codec, compressed))
                                if fault == "duplicate":
                                    assert replica.apply(decode(payload, codec, compressed)) == "duplicate"
                                    total_bytes += len(payload)
                                    duplicate_checks += 1
                                # Task request carries expected epoch/hash, so dropped updates cannot silently authorize a stale answer.
                                if replica.sequence != epoch or replica.context != context_hash or state_hash(replica.state) != state_hash(after):
                                    raise ResyncRequired("task state freshness check failed")
                            except ResyncRequired:
                                fallbacks += 1
                                repair = encode(snapshot(after, context_hash, epoch), codec, compressed)
                                fallback_bytes += len(bootstrap) + len(repair)
                                total_bytes += len(bootstrap) + len(repair)
                                verified_context = decode(bootstrap, codec, compressed)
                                assert digest(verified_context) == context_hash
                                replica = Replica(context_hash, allowed_ids=allowed)
                                replica.apply(decode(repair, codec, compressed))
                            assert replica.state == after
                            assert replica.sequence == epoch
                            latency.append(time.perf_counter_ns() - tick)
                        emit(stream, {"suite": "communication", "variant": variant,
                                      "compressed": compressed, "compression": "zstd-3" if compressed else "none",
                                      "fault_rate": fault_rate, "injected_faults": len(selected), "repetition": repetition,
                                      "snapshots": len(states), "bootstrap_bytes": len(bootstrap), "total_bytes": total_bytes,
                                      "fallback_bytes": fallback_bytes, "fallbacks": fallbacks, "duplicate_checks": duplicate_checks,
                                      "latency_ns": percentiles(latency), "stale_decisions": 0, "all_states_exact": True,
                                      "byte_scope": "encoded application payloads including bootstrap, retries, resync; excludes transport framing",
                                      "state_scope": "current health projection; full event log retained separately"})
            print(f"communication: {variant} complete", flush=True)


def retrieval_benchmark(repeats=5, directory=DATA, build="native"):
    documents = list(json_lines(directory / "documents.jsonl"))
    queries = [q for q in json_lines(directory / "queries.jsonl") if q["family"] == "retrieval"]
    model_info = model_manifest()
    write_json(RESULTS / "model.json", model_info)
    cache = directory / f"embeddings-{build}.npz"
    cache_meta = directory / f"embeddings-{build}.json"
    fingerprint = digest({"documents": file_hash(directory / "documents.jsonl"),
                          "queries": file_hash(directory / "queries.jsonl"), "model": model_info, "build": build,
                          "runtime_hashes": {name: file_hash(ROOT.parent / "scratch/BitNet" / f"build-{build}/bin" / name)
                                             for name in ("llama-server", "libllama.so", "libggml-cpu.so")}})
    inference_metrics = []
    with EmbeddingServer(build=build) as server:
        if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text()).get("fingerprint") == fingerprint:
            loaded = np.load(cache)
            doc_vectors, query_vectors = loaded["documents"], loaded["queries"]
            inference_metrics.extend(dict(t, reused_measurement_from_cache=True) for t in json.loads(cache_meta.read_text()).get("initial_encoding", []))
            print(f"retrieval: verified {build} embedding cache reused", flush=True)
        else:
            doc_vectors, timing = server.encode([d["text"] for d in documents], progress=lambda n, total: print(f"embedding documents {n}/{total}", flush=True) if n % 256 == 0 or n == total else None)
            inference_metrics.extend(dict(kind="corpus", **t) for t in timing)
            query_vectors, timing = server.encode([q["text"] for q in queries], kind="query", progress=lambda n, total: print(f"embedding queries {n}/{total}", flush=True) if n % 256 == 0 or n == total else None)
            inference_metrics.extend(dict(kind="queries", **t) for t in timing)
            np.savez(cache, documents=doc_vectors, queries=query_vectors)
            write_json(cache_meta, {"fingerprint": fingerprint, "build": build, "dimension": 640,
                                   "document_count": len(documents), "query_count": len(queries),
                                   "initial_encoding": inference_metrics})
        # Same deterministic micro-workload, five warmed timing repetitions per build.
        benchmark_texts = [q["text"] for q in queries[:128]]
        for repetition in range(repeats):
            _, timings = server.encode(benchmark_texts, kind="query")
            for t in timings:
                inference_metrics.append(dict(kind="warm_encode_128_queries", repetition=repetition, **t))
        server_peak = server.peak_rss
        startup = server.startup_seconds
    codes, scales = quantize(doc_vectors)
    np.save(directory / "vectors-int8.npy", codes)
    np.save(directory / "vector-scales.npy", scales)
    np.save(directory / "vectors-fp32.npy", doc_vectors)
    reconstructed = normalize(codes.astype(np.float32) * scales[:, None])
    storage = {"fp32_vector_bytes": doc_vectors.nbytes,
               "int8_vector_bytes": codes.nbytes + scales.nbytes,
               "int8_search_decoded_bytes": reconstructed.nbytes,
               "note": "INT8 is a storage experiment; NumPy search decodes to FP32 and is not a native INT8 kernel",
               "model_bytes": model_info["bytes"], "native_server_peak_rss_bytes": server_peak,
               "native_server_startup_seconds": startup,
               "minimum_cosine_fp32_vs_int8": float(np.min(np.sum(doc_vectors * reconstructed, axis=1)))}
    write_json(RESULTS / "vector-storage.json", storage)
    with (RESULTS / f"embedding-timing-{build}.jsonl").open("w") as stream:
        for t in inference_metrics:
            emit(stream, {"suite": "embedding_encode", "build": build, "threads": 4, **t})
    db = sqlite3.connect(directory / "events.sqlite")
    with (RESULTS / "retrieval.jsonl").open("w") as stream:
        for repetition in range(repeats):
            accum = {v: {"recall": [], "ndcg": [], "ns": []} for v in ("fts5", "fp32", "int8_storage", "fts5_plus_vectors")}
            cpu_start = time.process_time()
            for i, q in enumerate(queries):
                tick = time.perf_counter_ns()
                lexical = lexical_search(db, q["text"], 50)
                lexical_ns = time.perf_counter_ns() - tick
                tick = time.perf_counter_ns()
                full_ranking = top_k(doc_vectors @ query_vectors[i], 50)
                full_ns = time.perf_counter_ns() - tick
                tick = time.perf_counter_ns()
                quant_ranking = top_k(reconstructed @ query_vectors[i], 10)
                quant_ns = time.perf_counter_ns() - tick
                tick = time.perf_counter_ns()
                combined = reciprocal_rank_fusion(lexical, full_ranking)
                fusion_ns = time.perf_counter_ns() - tick
                for variant, ranking, elapsed in (("fts5", lexical[:10], lexical_ns), ("fp32", full_ranking[:10], full_ns),
                                                   ("int8_storage", quant_ranking, quant_ns),
                                                   ("fts5_plus_vectors", combined, lexical_ns + full_ns + fusion_ns)):
                    recall, ndcg = metrics(ranking, q["expected"])
                    accum[variant]["recall"].append(recall)
                    accum[variant]["ndcg"].append(ndcg)
                    accum[variant]["ns"].append(elapsed)
            cpu_seconds = time.process_time() - cpu_start
            for variant, m in accum.items():
                emit(stream, {"suite": "retrieval", "variant": variant, "repetition": repetition,
                              "queries": len(queries), "recall_at_10": float(np.mean(m["recall"])),
                              "ndcg_at_10": float(np.mean(m["ndcg"])), "search_latency_ns": percentiles(m["ns"]),
                              "total_search_ns": sum(m["ns"]), "all_variants_cpu_seconds": cpu_seconds,
                              "process_peak_rss_bytes": peak_rss_bytes(),
                              "timing_scope": "cached query vectors; encoding reported separately; FP32 decoded search for INT8 storage"})
            print(f"retrieval repetition {repetition + 1}/{repeats} complete", flush=True)
    db.close()


def compare_builds(repeats=5, directory=DATA):
    queries = [q["text"] for q in json_lines(directory / "queries.jsonl") if q["family"] == "retrieval"][:128]
    vectors = {}
    rows = []
    for build in ("native", "avx2"):
        with EmbeddingServer(build=build) as server:
            for repetition in range(repeats):
                output, timing = server.encode(queries, kind="query")
                vectors[build] = output
                rows.append({"suite": "instruction_set", "build": build, "repetition": repetition,
                             "items": len(queries), "elapsed_ns": sum(t["elapsed_ns"] for t in timing),
                             "cpu_seconds": sum(t["cpu_seconds"] for t in timing),
                             "startup_seconds": server.startup_seconds, "peak_rss_bytes": server.peak_rss})
    cosines = np.clip(np.sum(vectors["native"] * vectors["avx2"], axis=1), -1, 1)
    assert np.isfinite(cosines).all()
    assert np.min(cosines) > 0.999, "instruction-set outputs disagree materially"
    with (RESULTS / "instruction-set.jsonl").open("w") as stream:
        for row in rows:
            emit(stream, row)
    write_json(RESULTS / "instruction-set-equivalence.json", {
        "minimum_cosine": float(np.min(cosines)), "mean_cosine": float(np.mean(cosines)),
        "max_abs_coordinate_difference": float(np.max(np.abs(vectors["native"] - vectors["avx2"]))),
        "note": "Same cloud CPU and four threads; AVX2-constrained build is not a laptop performance prediction"})
