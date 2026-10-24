import copy
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from research_lab.common import digest, json_lines
from research_lab.data import generate, sql_decision, sql_reachable
from research_lab.embeddings import metrics, normalize, quantize, reciprocal_rank_fusion, top_k
from research_lab.runner import verify_projection
from research_lab.transport import Replica, ResyncRequired, decode, delta, encode, snapshot, state_hash
from research_lab.artifacts import record_suites, verify_suites
from research_lab.indexing import index_build_benchmark
from research_lab.common import file_hash


class FixtureTests(unittest.TestCase):
    def test_deterministic_generation_and_independent_oracles(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            left, right = Path(a), Path(b)
            args = dict(events=1_000, nodes=50, queries=80, seed=7)
            first = generate(left, **args)
            second = generate(right, **args)
            self.assertEqual(first, second)
            self.assertEqual(verify_projection(left), 1_000)
            db = sqlite3.connect(left / "events.sqlite")
            for q in json_lines(left / "queries.jsonl"):
                if q["family"] == "relational":
                    self.assertEqual(sql_reachable(db, q["node"], q["depth"]), q["expected"])
                elif q["family"] == "decision":
                    self.assertEqual(sql_decision(db, q["epoch"], q["node"]), q["expected"])
            db.close()

    def test_invalid_fixture_configuration(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(ValueError):
                generate(Path(path), events=100, nodes=24, queries=40)

    def test_index_build_preserves_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "data"
            results = root / "results"
            results.mkdir()
            generate(directory, events=1000, nodes=50, queries=80)
            before = file_hash(directory / "events.sqlite")
            with patch("research_lab.indexing.RESULTS", results):
                index_build_benchmark(1, directory)
            self.assertEqual(file_hash(directory / "events.sqlite"), before)
            self.assertEqual(len(list(json_lines(results / "index-build.jsonl"))), 2)


class ReplicaTests(unittest.TestCase):
    def setUp(self):
        self.context = digest({"schema": 1, "nodes": [0, 1, 2]})
        self.before = {0: 0, 1: 1, 2: 2}
        self.after = {0: 1, 1: 0}

    def replica(self):
        replica = Replica(self.context, allowed_ids={0, 1, 2})
        replica.apply(snapshot(self.before, self.context))
        return replica

    def test_round_trip_codecs_and_idempotent_replay(self):
        for codec in ("json", "cbor"):
            for compressed in (False, True):
                replica = self.replica()
                packet = delta(self.before, self.after, self.context, 1)
                decoded = decode(encode(packet, codec, compressed), codec, compressed)
                self.assertEqual(replica.apply(decoded), "applied")
                self.assertEqual(replica.state, self.after)
                self.assertEqual(replica.apply(decoded), "duplicate")

    def test_failures_are_atomic_and_require_resync(self):
        mutations = [
            lambda p: p.update(context="stale"),
            lambda p: p.update(base_hash="missing"),
            lambda p: p.update(sequence=5),
            lambda p: p.update(result_hash="damaged"),
            lambda p: p["updates"].append([99, 0]),
            lambda p: p.update(kind="unknown"),
        ]
        for mutate in mutations:
            replica = self.replica()
            packet = delta(self.before, self.after, self.context, 1)
            mutate(packet)
            with self.assertRaises(ResyncRequired):
                replica.apply(packet)
            self.assertEqual(replica.state, self.before)
            self.assertEqual(replica.sequence, 0)
            replica.apply(snapshot(self.after, self.context, 1))
            self.assertEqual(replica.state, self.after)

    def test_unknown_identifiers_in_snapshot_and_invalid_health(self):
        replica = self.replica()
        for bad in ({99: 0}, {0: 3}, {0: True}):
            with self.assertRaises(ResyncRequired):
                replica.apply(snapshot(bad, self.context, 1))
        self.assertEqual(replica.state, self.before)

    def test_stale_snapshot_cannot_roll_back_state(self):
        replica = self.replica()
        old = snapshot(self.before, self.context)
        replica.apply(delta(self.before, self.after, self.context, 1))
        with self.assertRaises(ResyncRequired):
            replica.apply(old)
        self.assertEqual(replica.state, self.after)


class EmbeddingRepresentationTests(unittest.TestCase):
    def test_quantization_and_cosine_ranking(self):
        rng = np.random.default_rng(42)
        vectors = normalize(rng.normal(size=(32, 640)))
        codes, scales = quantize(vectors)
        restored = normalize(codes.astype(np.float32) * scales[:, None])
        self.assertEqual(codes.dtype, np.int8)
        self.assertTrue(np.all(np.sum(restored * vectors, axis=1) > 0.999))
        self.assertLess(codes.nbytes + scales.nbytes, vectors.nbytes)
        self.assertEqual(top_k(np.array([0.5, 1.0, 1.0]), 2), [1, 2])

    def test_known_retrieval_metrics_and_fusion(self):
        self.assertEqual(metrics([1, 2], {1, 2}), (1.0, 1.0))
        recall, ndcg = metrics([3, 1], {1, 2})
        self.assertEqual(recall, 0.5)
        self.assertGreater(ndcg, 0)
        self.assertLess(ndcg, 1)
        self.assertEqual(reciprocal_rank_fusion([1, 2], [2, 3])[0], 2)

    def test_invalid_embeddings_fail(self):
        for values in ([[0.0, 0.0]], [[float("nan"), 1.0]]):
            with self.assertRaises(ValueError):
                normalize(values)


class ArtifactBindingTests(unittest.TestCase):
    def test_registered_additional_output_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            results = root / "results"
            data.mkdir()
            results.mkdir()
            (data / "manifest.json").write_text('{"seed":42}')
            (results / "summary.json").write_text('{}')
            extra = results / "per-query.jsonl"
            extra.write_text('{"score":0.5}\n')
            with patch("research_lab.artifacts.RESULTS", results), patch("research_lab.artifacts.SUITES", {"test": ["summary.json"]}):
                record_suites(["test"], data)
                bindings = json.loads((results / "suite-inputs.json").read_text())
                bindings["test"]["outputs"][extra.name] = file_hash(extra)
                (results / "suite-inputs.json").write_text(json.dumps(bindings))
                verify_suites(data)
                extra.write_text('{"score":0.9}\n')
                with self.assertRaises(RuntimeError):
                    verify_suites(data)

    def test_stale_or_changed_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "data"
            results = root / "results"
            directory.mkdir()
            results.mkdir()
            (directory / "manifest.json").write_text('{"seed":42}')
            raw = results / "test.jsonl"
            raw.write_text('{"correct":true}\n')
            with patch("research_lab.artifacts.RESULTS", results), patch("research_lab.artifacts.SUITES", {"test": ["test.jsonl"]}):
                with self.assertRaises(RuntimeError):
                    verify_suites(directory)
                record_suites(["test"], directory)
                verify_suites(directory)
                raw.write_text('{"correct":false}\n')
                with self.assertRaises(RuntimeError):
                    verify_suites(directory)
                record_suites(["test"], directory)
                (directory / "manifest.json").write_text('{"seed":43}')
                with self.assertRaises(RuntimeError):
                    verify_suites(directory)


if __name__ == "__main__":
    unittest.main()
