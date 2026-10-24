from __future__ import annotations

import csv
import html
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone

from .common import DATA, RESULTS, ROOT, file_hash, json_lines, write_json
from .artifacts import verify_suites


def esc(value):
    return html.escape(str(value))


def table(headers, rows):
    return "<div class='table'><table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def median(rows, key):
    return statistics.median(row[key] for row in rows)


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def mib(value):
    return f"{value / 1024**2:,.2f} MiB"


def render():
    required = ["environment.json", "verification.json", "exact.jsonl", "closure.jsonl", "snapshot-closure.jsonl",
                "communication.jsonl", "retrieval.jsonl", "vector-storage.json", "instruction-set.jsonl",
                "instruction-set-equivalence.json", "index-build.jsonl", "sqlite-object-storage.json",
                "vector-kernels.jsonl", "vector-kernel-validation.json"]
    missing = [name for name in required if not (RESULTS / name).exists()]
    if missing:
        raise RuntimeError("report incomplete; missing measured results: " + ", ".join(missing))
    verify_suites()
    env = json.loads((RESULTS / "environment.json").read_text())
    fixture = json.loads((DATA / "manifest.json").read_text())
    verified = json.loads((RESULTS / "verification.json").read_text())
    exact = list(json_lines(RESULTS / "exact.jsonl"))
    closures = list(json_lines(RESULTS / "closure.jsonl"))
    snapshots = list(json_lines(RESULTS / "snapshot-closure.jsonl"))
    communication = list(json_lines(RESULTS / "communication.jsonl"))
    retrieval = list(json_lines(RESULTS / "retrieval.jsonl"))
    vectors = json.loads((RESULTS / "vector-storage.json").read_text())
    isa = list(json_lines(RESULTS / "instruction-set.jsonl"))
    equivalence = json.loads((RESULTS / "instruction-set-equivalence.json").read_text())
    assert all(row["correct"] == row["queries"] for row in exact)
    assert all(row["correct"] for row in snapshots)
    assert all(row["all_states_exact"] and row["stale_decisions"] == 0 for row in communication)
    assert all(row["queries"] == fixture["queries"] // 4 for row in exact + retrieval)
    assert all(row["snapshots"] == fixture["epochs"] for row in communication)
    assert {row["epoch"] for row in snapshots} == set(range(fixture["epochs"]))
    summary = {"fixture": fixture, "verification": verified, "environment": env,
               "all_exact_checks_pass": True, "proofs_verified": sum(row["proofs_verified"] for row in snapshots),
               "communication_states_verified": sum(row["snapshots"] for row in communication),
               "energy_joules": None, "energy_reason": env["energy_reason"],
               "retrieval": {}, "closure": {}, "communication": {},
               "index_build": {}, "instruction_set": {}, "instruction_set_equivalence": equivalence,
               "storage": vectors, "initial_encoding": {}, "vector_kernels": {}}
    sections = []
    sections.append("<h1>Compact state, closure, and ternary retrieval</h1>"
                    f"<p class='subtitle'>Cloud pilot · {datetime.now(timezone.utc).strftime('%Y-%m-%d')} · measured results</p>")
    sections.append(f"<p>The pilot uses <strong>{fixture['events']:,} events, {fixture['nodes']:,} entities and {fixture['queries']:,} queries</strong>. "
                    f"All {verified['events_replayed_and_verified']:,} original event records were replayed and checked. "
                    f"Exact queries agreed with their independent references in every repetition; {summary['proofs_verified']:,} proof entries were checked. "
                    "Approximate retrieval is evaluated separately and never authorizes state transitions.</p>")
    sections.append("<h2>Environment and measurement boundaries</h2>")
    sections.append(table(["Item", "Observed"], [
        ["CPU", env["cpu_model"]], ["Quota", f"{env['cpu_max']} (four cores); {env['visible_cpus']} visible CPUs"],
        ["RAM limit", mib(int(env["memory_max"]))], ["GPU", "None exposed"],
        ["Toolchain", f"Python {env['python'].split()[0]}, {env['rust']}, SQLite {env['sqlite']} / FTS5, GCC 14.2"],
        ["Model", "Microsoft BitNet embedding 270M / native I2_S / last-token pooling / 640 dimensions"],
        ["Energy", "Unavailable: no exposed sensor; no joule or energy-efficiency claim"],
        ["Network", "Local codec and replica experiment; no LAN/RPC latency measurement"],
    ]))
    sections.append(f"<p>All final timing runs occur after native builds finish. {verified['repetitions']} repetitions are recorded. SQLite repetitions open new connections but do not flush the OS page cache. "
                    "Native model startup is measured separately from warm encoding. Host contention outside this workspace is uncontrolled. "
                    "AVX2 results use the same cloud CPU and are not predictions for either laptop.</p>")
    sections.append("<h2>Exact facts and decisions</h2>")
    rows = []
    for (family,), values in grouped(exact, ["family"]).items():
        rows.append([family, values[0]["queries"], f"{sum(v['correct'] for v in values):,}/{sum(v['queries'] for v in values):,}",
                     f"{statistics.median(v['latency_ns']['p50'] for v in values)/1000:.1f} µs",
                     f"{statistics.median(v['latency_ns']['p95'] for v in values)/1000:.1f} µs"])
    sections.append(table(["Task", "Queries/repetition", "Correct/all repetitions", "Median p50", "Median p95"], rows))
    sections.append("<p>Reconstruction checks the canonical source record's SHA-256. Relational queries compare SQLite recursive CTE results with bounded breadth-first graph traversal. "
                    "Restart decisions use an independent group-state oracle: unknown membership/health blocks authorization; otherwise at least three healthy replicas must remain. "
                    "The complete event log and provenance remain authoritative.</p>")
    index_rows = []
    for (variant,), values in grouped(list(json_lines(RESULTS / "index-build.jsonl")), ["variant"]).items():
        summary["index_build"][variant] = {"elapsed_ns": median(values, "elapsed_ns"), "cpu_seconds": median(values, "cpu_seconds")}
        index_rows.append([variant, f"{median(values, 'elapsed_ns')/1e6:.3f}", f"{median(values, 'cpu_seconds'):.4f}"])
    sections.append(table(["Index construction on copied input", "Median ms", "CPU seconds"], index_rows))
    sections.append("<p>Index construction includes create/populate/commit, measured on disposable copies. Copying and removing old indexes are excluded; the authoritative database checksum is verified unchanged. "
                    "See sqlite-object-storage.json for allocated pages by table/index.</p>")
    sections.append("<h2>Closure computation</h2>")
    ground = grouped([r for r in closures if r["suite"] == "ground_closure"], ["fixture", "variant"])
    rows = []
    for name in ("chain", "fanout", "conjunction_cycle", "mostly_blocked"):
        naive = median(ground[name, "naive"], "elapsed_ns")
        worklist = median(ground[name, "worklist"], "elapsed_ns")
        proof = median(ground[name, "worklist_proofs"], "elapsed_ns")
        d = {"naive_ns": naive, "delta_ns": median(ground[name, "delta_rounds"], "elapsed_ns"),
             "worklist_ns": worklist, "reused_index_ns": median(ground[name, "worklist_reused_index"], "elapsed_ns"),
             "proof_ns": proof, "speedup_vs_naive": naive / worklist,
             "worklist_allocations": ground[name, "worklist"][0]["allocations"],
             "worklist_peak_requested_heap_bytes": ground[name, "worklist"][0]["peak_heap_bytes"],
             "worklist_yielded_facts_per_second": ground[name, "worklist"][0]["facts"] / (worklist/1e9)}
        summary["closure"][name] = d
        rows.append([name, f"{naive/1e6:.3f}", f"{d['delta_ns']/1e6:.3f}", f"{worklist/1e6:.3f}",
                     f"{d['reused_index_ns']/1e6:.3f}", f"{naive/worklist:.1f}×", f"{proof/worklist:.2f}×"])
    sections.append(table(["Ground fixture", "Naive ms", "Delta ms", "Worklist ms", "Reused index ms", "Naive/worklist", "Proof time multiplier"], rows))
    sections.append("<p>Ground evaluators use identical rules and seeds. The chain and conjunction fixtures emphasize long deduction depth against synchronous naive rounds; speedups are workload-specific. "
                    "Worklist time includes index construction unless the column explicitly says reused index. "
                    "The first proof is retained, not all possible proofs. Allocations and peak requested heap bytes are profiled in a separate pass; timed allocations retain an enable-flag check. "
                    "These heap counts exclude input fixtures, allocator metadata and RSS. Proof multipliers include proof construction and the same index setup.</p>")
    rows = []
    for (name, variant), values in grouped([r for r in closures if r["suite"] == "relational_closure"], ["fixture", "variant"]).items():
        rows.append([name, variant, f"{median(values,'elapsed_ns')/1e6:.3f}", values[0]["join_probes"], values[0]["join_matches"]])
    sections.append(table(["Relational fixture", "Evaluator", "Median ms", "Join probes", "Join matches"], rows))
    sections.append(f"<p>The relational comparison uses the same indexed reachability join in both evaluators. It is separate from the ground-rule comparison. "
                    f"The full fixture rebuilt closure from each of {fixture['epochs']} snapshots, across {len(set(r['repetition'] for r in snapshots))} repetitions, and checked results against graph propagation. "
                    f"Median engine time per snapshot was {statistics.median(r['engine_ns'] for r in snapshots)/1e6:.3f} ms; "
                    f"the median program-write/process/engine time was {statistics.median(r['end_to_end_ns'] for r in snapshots)/1e6:.3f} ms.</p>")
    sections.append("<p>Closure theory applies to the finite positive Horn layer: extensivity, monotonicity and idempotence are checked against an independent naive evaluator. "
                    "The mutable world is projected into a new snapshot before reasoning; deletion is handled by recomputation. Counts, quorum checks and missing evidence stay outside monotone inference. "
                    "Closure computes consequences of known facts and rules, rather than supplying new observations. "
                    "These checks establish behavior under the fixture rules, not the real-world soundness of arbitrary rules. "
                    "A <a href='results/proof-example.json'>proof example</a> follows a derived impact back to its source event and checksum.</p>")
    sections.append("<h2>Shared context and communication</h2>")
    rows = []
    for (variant, compressed, rate), values in grouped(communication, ["variant", "compressed", "fault_rate"]).items():
        if rate != 0:
            continue
        total = median(values, "total_bytes")
        label = f"{variant} / {'zstd-3' if compressed else 'raw'}"
        p50_ms = statistics.median(v["latency_ns"]["p50"] for v in values)/1e6
        summary["communication"][label] = {"total_bytes": total, "bootstrap_bytes": values[0]["bootstrap_bytes"], "codec_apply_check_p50_ms": p50_ms}
        rows.append([variant, "zstd-3" if compressed else "none", f"{total:,.0f}", f"{values[0]['bootstrap_bytes']:,}",
                     f"{total/values[0]['snapshots']:,.1f}", f"{p50_ms:.3f}"])
    sections.append(table(["Representation", "Compression", "Total bytes", "Included bootstrap", "Bytes/exact state update", "Codec/apply/check p50 ms"], rows))
    full = summary["communication"]["cbor_full / zstd-3"]["total_bytes"]
    inc = summary["communication"]["cbor_delta / zstd-3"]["total_bytes"]
    json_full = summary["communication"]["json_full / zstd-3"]["total_bytes"]
    sections.append(f"<p>For the same compressed CBOR codec, full-snapshot bytes divided by delta bytes were <strong>{full/inc:.2f}×</strong>, including initial shared context. "
                    "This compares health-state projections, not lossless transfer of every historical event. The event log is retained separately. "
                    "Byte counts include encoded application payloads, bootstrap, duplicate sends and recovery; they exclude TCP/TLS framing. No physical wire bandwidth is inferred.</p>")
    sections.append("<p>The local timing interval includes encoding, decoding, applying updates and freshness checks, including recovery when needed. "
                    "It excludes packet construction/delta discovery and initial dictionary bootstrap. It is not full communication latency.</p>")
    if json_full < full:
        sections.append(f"<p>Compressed JSON snapshots used {100*(1-json_full/full):.1f}% fewer bytes than compressed CBOR snapshots. "
                        "A smaller raw binary representation did not imply a smaller compressed payload on this fixture. Delta reuse is evaluated against the same CBOR codec to isolate that benefit.</p>")
    rows = []
    for (variant, compressed, rate), values in grouped(communication, ["variant", "compressed", "fault_rate"]).items():
        if variant != "cbor_delta" or not compressed:
            continue
        rows.append([f"{rate:.0%}", values[0]["injected_faults"], values[0]["fallbacks"], f"{median(values,'total_bytes'):,.0f}",
                     f"{median(values,'fallback_bytes'):,.0f}", "0"])
    sections.append(table(["Fault rate", "Faults/run", "Resyncs/run", "Total bytes", "Recovery bytes", "Stale decisions"], rows))
    sections.append("<p>Fault injection covers context/base mismatch, unknown identifiers, out-of-order or dropped updates, and duplicate delivery. "
                    "The task freshness gate checks the expected epoch and state hash. Failure requires verified context and snapshot resynchronization before a dependent decision. "
                    "Replay and malformed-state handling are also covered by unit tests.</p>")
    sections.append("<h2>Associative retrieval and vector storage</h2>")
    rows = []
    for (variant,), values in grouped(retrieval, ["variant"]).items():
        quality = {"recall_at_10": median(values, "recall_at_10"), "ndcg_at_10": median(values, "ndcg_at_10")}
        summary["retrieval"][variant] = quality
        rows.append([variant, f"{quality['recall_at_10']:.4f}", f"{quality['ndcg_at_10']:.4f}",
                     f"{statistics.median(v['search_latency_ns']['p50'] for v in values)/1e6:.3f}",
                     f"{statistics.median(v['search_latency_ns']['p95'] for v in values)/1e6:.3f}"])
    sections.append(table(["Retriever", "Recall@10", "nDCG@10", "Search p50 ms", "Search p95 ms"], rows))
    best = max(summary["retrieval"], key=lambda name: summary["retrieval"][name]["ndcg_at_10"])
    sections.append(f"<p><strong>{esc(best)}</strong> has the highest nDCG@10 on this fixture. The four query templates use paraphrases absent from the canonical document wording. "
                    "Relevance labels come from exact attributes, not model judgments. These synthetic results do not establish performance on natural documents. "
                    "Hybrid retrieval uses reciprocal-rank fusion of the top 50 lexical and vector candidates. Search timing uses cached query embeddings; encoding cost is shown separately.</p>")
    sections.append(table(["Storage or memory component", "Bytes"], [
        ["SQLite exact state and indexes", f"{(DATA/'events.sqlite').stat().st_size:,}"],
        ["Canonical document records used for retrieval", f"{(DATA/'documents.jsonl').stat().st_size:,}"],
        ["Native ternary model artifact", f"{vectors['model_bytes']:,}"],
        ["FP32 document vectors (logical payload)", f"{vectors['fp32_vector_bytes']:,}"],
        ["INT8 codes + FP32 row scales (logical payload)", f"{vectors['int8_vector_bytes']:,}"],
        ["Decoded INT8 search matrix (additional resident FP32)", f"{vectors['int8_search_decoded_bytes']:,}"],
        ["Native embedding server peak RSS", f"{vectors['native_server_peak_rss_bytes']:,}"],
    ]))
    sections.append(f"<p>Output-vector quantization reduces logical vector storage by {vectors['fp32_vector_bytes']/vectors['int8_vector_bytes']:.2f}×; "
                    f"the minimum reconstructed cosine is {vectors['minimum_cosine_fp32_vs_int8']:.6f}. "
                    "This is separate from ternary model-weight quantization. The initial NumPy path decodes INT8 vectors to FP32; the native follow-up below keeps the packed corpus resident. "
                    "Both variants retain the original exact records. The model and index costs can exceed savings in small corpora.</p>")
    common_bytes = vectors["model_bytes"] + (DATA / "events.sqlite").stat().st_size
    sections.append(f"<p>Counting SQLite, the model artifact and logical document vectors, the subtotal falls by "
                    f"{100*(vectors['fp32_vector_bytes']-vectors['int8_vector_bytes'])/(common_bytes+vectors['fp32_vector_bytes']):.2f}% with vector quantization. "
                    "That subtotal excludes runtime/dependency files. A fourfold reduction in vector payload is not a fourfold reduction in the whole system.</p>")
    sections.append("<h2>Native packed-vector search follow-up</h2>")
    native_rows = list(json_lines(RESULTS / "vector-kernels.jsonl"))
    validation = json.loads((RESULTS / "vector-kernel-validation.json").read_text())
    rows = []
    for (variant,), values in grouped(native_rows, ["variant"]).items():
        if not all(v["queries"] == fixture["queries"]//4 and v["threads"] == 1 for v in values):
            raise RuntimeError("native vector comparison has inconsistent workload or thread count")
        cpu_values = [v["query_loop_cpu_ns"] for v in values if v["query_loop_cpu_ns"] is not None]
        result = {"median_total_search_ns": median(values, "total_search_ns"),
                  "min_total_search_ns": min(v["total_search_ns"] for v in values),
                  "max_total_search_ns": max(v["total_search_ns"] for v in values),
                  "query_loop_cpu_ns": statistics.median(cpu_values) if cpu_values else None,
                  "repetitions": len(values),
                  "search_p50_ns": statistics.median(v["search_latency_ns"]["p50"] for v in values),
                  "search_p95_ns": statistics.median(v["search_latency_ns"]["p95"] for v in values),
                  "corpus_bytes": median(values, "corpus_bytes"),
                  "loaded_rss_bytes": median(values, "loaded_rss_bytes"),
                  "peak_rss_bytes": median(values, "peak_rss_bytes"),
                  "recall_at_10": median(values, "recall_at_10"),
                  "ndcg_at_10": median(values, "ndcg_at_10"),
                  "hybrid_recall_at_10": median(values, "hybrid_recall_at_10"),
                  "hybrid_ndcg_at_10": median(values, "hybrid_ndcg_at_10"),
                  "max_abs_score_error": max(v["max_abs_score_error"] for v in values)}
        summary["vector_kernels"][variant] = result
        rows.append([variant, f"{result['search_p50_ns']/1e6:.3f}", f"{result['search_p95_ns']/1e6:.3f}",
                     mib(result["corpus_bytes"]), mib(result["peak_rss_bytes"]),
                     f"{result['query_loop_cpu_ns']/1e9:.3f}" if cpu_values else "unavailable",
                     f"{result['recall_at_10']:.4f}", f"{result['hybrid_recall_at_10']:.4f}"])
    sections.append(table(["Rust / one thread", "Search p50 ms", "Search p95 ms", "Corpus + norms", "Worker peak RSS", "Query-loop CPU s", "Recall@10", "Hybrid Recall@10"], rows))
    fp = summary["vector_kernels"]["fp32"]
    quant = summary["vector_kernels"]["int8"]
    speedup = fp["median_total_search_ns"]/quant["median_total_search_ns"]
    sections.append(f"<p>The same Rust top-50 heap and {native_rows[0]['kernel']} instruction path are used for FP32 and INT8, with one CPU thread in each isolated worker. "
                    f"Over {fp['repetitions']} repetitions, FP32 total search time divided by INT8 time was <strong>{speedup:.2f}×</strong>. "
                    f"The resident corpus allocation was {fp['corpus_bytes']/quant['corpus_bytes']:.2f}× smaller; "
                    f"worker peak RSS changed from {mib(fp['peak_rss_bytes'])} to {mib(quant['peak_rss_bytes'])}.</p>")
    sections.append(f"<p>Across repetitions, FP32 search time ranged from {fp['min_total_search_ns']/1e9:.3f} to {fp['max_total_search_ns']/1e9:.3f} seconds; "
                    f"INT8 ranged from {quant['min_total_search_ns']/1e9:.3f} to {quant['max_total_search_ns']/1e9:.3f} seconds. "
                    "These ranges expose the variability of this cloud run.</p>")
    sections.append("<p>Queries remain FP32. INT8 codes are expanded eight values at a time in AVX2 registers (or individually in the scalar fallback); "
                    "there is no full decoded corpus matrix. Positive row scales cancel in cosine similarity, so only an inverse norm per row is retained. "
                    "This is mixed FP32-query/INT8-storage arithmetic, not an integer-only dot product. "
                    "Search timing includes normalization, dot products and top-k selection; file loading, output serialization and query encoding are separate. "
                    "The CPU counter measures scheduled thread runtime over the query loop, including result formatting/output, and excludes loading. "
                    "The isolated RSS includes shared-shape query vectors and worker overhead, and excludes the Python controller and embedding model.</p>")
    sections.append(f"<p>Native scores and top-50 membership were checked against an independent FP64 cosine oracle on {len(validation['sampled_queries'])} evenly spaced queries. "
                    f"Maximum observed score errors were {fp['max_abs_score_error']:.3g} (FP32) and {quant['max_abs_score_error']:.3g} (INT8), "
                    "with a 2e-6 rounding tolerance. Signed-code, scalar-tail, invalid-input and stable-tie cases are covered by tests. "
                    "Worker order alternates between repetitions. The NumPy table above uses four BLAS threads and a different selection routine, so it is not a matched speed baseline.</p>")
    sections.append("<h2>Encoding and instruction-set comparison</h2>")
    encoding_rows = list(json_lines(RESULTS / "embedding-timing-native.jsonl"))
    build_rows = []
    for kind in ("corpus", "queries"):
        values = [v for v in encoding_rows if v["kind"] == kind]
        if values:
            elapsed = sum(v["elapsed_ns"] for v in values)/1e9
            count = sum(v["items"] for v in values)
            summary["initial_encoding"][kind] = {"texts": count, "request_seconds": elapsed,
                                                "server_cpu_seconds": sum(v["cpu_seconds"] for v in values)}
            build_rows.append([kind, count, f"{elapsed:.3f}", f"{count/elapsed:.2f}", f"{sum(v['cpu_seconds'] for v in values):.2f}"])
    sections.append(table(["Initial encoding / one measured pass", "Texts", "Request seconds", "Texts/sec", "Server CPU seconds"], build_rows))
    sections.append("<p>Initial corpus/query encoding is measured once and cached with model, runtime and fixture fingerprints. "
                    "The five-repetition timing comparison below uses a fixed 128-query sample. Encoding and cached search are separate costs.</p>")
    rows = []
    for (build,), values in grouped(isa, ["build"]).items():
        elapsed = median(values, "elapsed_ns")
        summary["instruction_set"][build] = {"texts": values[0]["items"], "median_elapsed_ns": elapsed,
                                            "texts_per_second": values[0]["items"]/(elapsed/1e9),
                                            "peak_rss_bytes": max(v["peak_rss_bytes"] for v in values)}
        rows.append([build, values[0]["items"], f"{elapsed/1e9:.3f}", f"{values[0]['items']/(elapsed/1e9):.2f}",
                     f"{median(values,'startup_seconds'):.3f}", mib(max(v["peak_rss_bytes"] for v in values))])
    sections.append(table(["Build / 4 threads", "Queries", "Warm batch seconds", "Queries/sec", "Startup seconds", "Peak RSS"], rows))
    sections.append(f"<p>The identical 128-query workload is encoded five times per build. This is batched request throughput, not single-query latency or generated tokens/sec. "
                    f"Minimum native/AVX2 embedding cosine: {equivalence['minimum_cosine']:.8f}; maximum coordinate difference: {equivalence['max_abs_coordinate_difference']:.8g}. "
                    "A separate FP16 model baseline was not run, so these measurements do not establish a ternary-versus-FP16 speedup or teacher-quality parity.</p>")
    sections.append("<p>The native and AVX2 outputs are numerically close but are not bit-identical. The cosine sanity check does not establish lossless cross-kernel equivalence.</p>")
    sections.append("<p>The native server uses one active slot (-np 1), four CPU threads, 2,048 context tokens and 512-token batch/microbatch limits. "
                    "HTTP input arrays contain 16 texts. This is the configured baseline, not a tuned maximum-throughput result.</p>")
    sections.append("<h2>Evidence audit and next experiments</h2>")
    sections.append("<p>The <a href='evidence/claim-audit.csv'>claim register</a> distinguishes source support, conditional claims, contradicted interpretations and unverified hardware. "
                    "Key corrections: native ternary execution is not lossless conversion of arbitrary FP16 models; Maple uses eight active experts; Bonsai 27B is dense and has approximately 7.2 GB of deployed weights; "
                    "PyO3 uses abi3t-py315; CPU memory bandwidth is not PCIe bandwidth; summed GPU capacities are not unified VRAM.</p>")
    with (ROOT / "evidence/claim-audit.csv").open() as stream:
        audit = list(csv.DictReader(stream))
    sections.append("<details><summary>Show all claim assessments and sources</summary>" +
                    table(["ID", "Claim", "Status", "Finding"], [[r["id"], r["claim"], r["status"], r["finding"]] for r in audit]) +
                    "<ol>" + "".join(f"<li>{esc(r['id'])}: <a href='{esc(r['source'])}'>{esc(r['source'])}</a></li>" for r in audit) + "</ol></details>")
    sections.append("<p>Next: validate laptop hardware and memory bandwidth; build matched supported CPU/GPU runtimes; budget master weights, gradients, optimizer state and activations before LoRA/MoTE training; "
                    "then benchmark single-node baselines before RPC. Keep VSA/HDC and learned semantic codecs as separate ablations. "
                    "Adopt approximate retrieval only where measured task quality justifies its complete cost; retain exact state and verified context for decisions.</p>")
    if (RESULTS / "learning/default/manifest.json").exists():
        from .learning.experiment import verify_manifest as verify_learning_manifest
        verify_learning_manifest(RESULTS / "learning/default")
        sections.append("<h2>Online learning follow-up</h2><p>A separate <a href='results/learning/default/REPORT.html'>controlled learning experiment</a> "
                        "compares rule learning, information-seeking questions, noisy feedback, and adaptation to changing worlds. "
                        "Its results and fixtures are independent of the storage and inference benchmarks above.</p>")
    discovery_manifest = RESULTS / "discovery/default/manifest.json"
    if discovery_manifest.exists() and json.loads(discovery_manifest.read_text()).get("status") == "complete":
        from .discovery.experiment import verify as verify_discovery_manifest
        verify_discovery_manifest(RESULTS / "discovery/default")
        sections.append("<h2>Discovery worlds follow-up</h2><p>The <a href='results/discovery/default/REPORT.html'>discovery experiment</a> "
                        "tests generated explanations, relational concept reuse, and recursive relay models under noisy feedback. "
                        "See its separate fixtures, source-linked research, and evidence for the results and limitations.</p>")
    active_manifest = RESULTS / "active/default/manifest.json"
    if active_manifest.exists() and json.loads(active_manifest.read_text()).get("status") == "complete":
        from .active.study import verify as verify_active
        verify_active(RESULTS / "active/default")
        sections.append("<h2>Active probabilistic learning</h2><p>The <a href='results/active/default/REPORT.html'>active learning and concept-transfer study</a> "
                        "compares noise-aware probability updates, prediction-oriented questions, changing and returning rules, "
                        "and reusable typed concepts. Its report includes confidence checks, acquisition costs, and negative findings.</p>")
    sections.append("<h2>Reproduction and artifacts</h2><p>See <a href='README.md'>README</a>, <a href='results/summary.json'>machine-readable summary</a>, "
                    "<a href='results/summary.csv'>summary CSV</a>, <a href='data/manifest.json'>fixture manifest</a>, and <a href='results/manifest.json'>artifact manifest</a>. "
                    "Raw per-repetition JSONL files and native build/runtime logs are in results/.</p>")
    document = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>" \
        "<title>Compact state research pilot</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#17232c;background:#fafbfc}h1{font-size:2.25rem;line-height:1.2}h2{margin-top:2.5rem}p{max-width:100ch}.subtitle{color:#586874}a{color:#0b6477}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{text-align:left;padding:10px 12px;border-bottom:1px solid #dce3e8;vertical-align:top}th{background:#eaf0f4}details{margin-top:20px}summary{cursor:pointer}</style><body>" + "\n".join(sections) + "</body></html>"
    path = ROOT / "REPORT.html"
    path.write_text(document)
    write_json(RESULTS / "summary.json", summary)
    with (RESULTS / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("suite", "variant", "metric", "value"))
        for suite in ("closure", "retrieval", "communication", "index_build", "instruction_set", "initial_encoding", "vector_kernels"):
            for variant, values in summary[suite].items():
                for key, value in values.items():
                    writer.writerow((suite, variant, key, value))
    artifacts = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": file_hash(p)}
                 for p in sorted(RESULTS.iterdir()) if p.is_file() and p.name != "manifest.json" and p.suffix in (".json", ".jsonl", ".csv")}
    artifacts["REPORT.html"] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    write_json(RESULTS / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "files": artifacts,
               "source_files": {str(p.relative_to(ROOT)): file_hash(p) for folder in ("research_lab", "src", "tests", "scripts", "examples")
                                for p in sorted((ROOT / folder).rglob("*")) if p.suffix in (".py", ".rs", ".sh")},
               "configuration_files": {**{name: file_hash(ROOT / name) for name in ("Cargo.toml", "Cargo.lock", "requirements.lock", "README.md", ".gitignore")},
                                       **{str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "configs").rglob("*.json"))}}})
    return path
