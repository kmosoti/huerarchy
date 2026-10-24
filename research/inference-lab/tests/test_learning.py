from dataclasses import asdict
from fractions import Fraction
import gzip
import itertools
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from research_lab.learning.components import (
    BayesianLearner, DirectReasoner, MemoryLearner, PageHinkley, RustReasoner,
    SQLiteMemory, hypothesis_space,
)
from research_lab.learning.experiment import make_learner, recovery_delays, run, verify_manifest
from research_lab.learning.types import LearningEvent, Observation, catalog, create
from research_lab.learning.worlds import configuration, make_worlds, truth_at


def observation(sequence, features, label):
    return Observation(sequence, features, bool(label), "test-teacher")


class PosteriorTests(unittest.TestCase):
    def test_bayes_matches_exact_rational_reference(self):
        learner = BayesianLearner(dimensions=3, noise=0.1)
        examples = [(7, True), (3, False), (6, True), (6, False)]
        expected = []
        # Independent enumeration and rational likelihood, including constant false.
        for signs in [None, *itertools.product((-1, 0, 1), repeat=3)]:
            size = sum(sign != 0 for sign in signs) if signs else 0
            weight = Fraction(1, 2**size)
            for mask, label in examples:
                predicted = signs is not None and all(sign == 0 or bool(mask & (1 << i)) == (sign == 1)
                                                      for i, sign in enumerate(signs))
                weight *= Fraction(9 if predicted == label else 1, 10)
            expected.append(weight)
        for seq, (mask, label) in enumerate(examples, 1):
            learner.observe(observation(seq, mask, label))
        total = sum(expected)
        np.testing.assert_allclose(learner.weights(), [float(w / total) for w in expected], atol=1e-14)
        self.assertEqual(len(hypothesis_space(6)[0]), 730)

    def test_window_uses_only_recent_evidence(self):
        recent = BayesianLearner(dimensions=3, mode="window", window=3)
        examples = [(7, True), (3, False), (6, True), (6, False), (0, False), (1, True)]
        for seq, (mask, label) in enumerate(examples, 1):
            recent.observe(observation(seq, mask, label))
        reference = BayesianLearner(dimensions=3)
        for seq, (mask, label) in enumerate(examples[-3:], 1):
            reference.observe(observation(seq, mask, label))
        np.testing.assert_allclose(recent.weights(), reference.weights(), atol=1e-14)

    def test_noisy_contradictions_remain_probabilistic(self):
        learner = BayesianLearner(dimensions=3)
        for seq in range(1, 101):
            learner.observe(observation(seq, 7, seq % 3 == 0))
        self.assertTrue(np.all(np.isfinite(learner.log_weights)))
        self.assertAlmostEqual(float(learner.weights().sum()), 1)
        self.assertGreater(learner.predict(7).probability, 0)
        self.assertLess(learner.predict(7).probability, 1)
        with self.assertRaises(ValueError):
            learner.observe(observation(102, 7, True))

    def test_rule_revision_changes_predictions(self):
        learner = BayesianLearner(dimensions=3, mode="window", window=16)
        for seq in range(1, 65):
            mask = (seq - 1) % 8
            learner.observe(observation(seq, mask, mask & 1))
        self.assertGreater(learner.predict(1).probability, 0.99)
        for seq in range(65, 129):
            mask = (seq - 1) % 8
            learner.observe(observation(seq, mask, not mask & 1))
        self.assertLess(learner.predict(1).probability, 0.01)
        self.assertGreater(learner.predict(0).probability, 0.99)

    def test_frozen_and_memory_boundaries(self):
        frozen = create("learner", "frozen", dimensions=3)
        before = frozen.probabilities().copy()
        memory = MemoryLearner(dimensions=3)
        for seq in range(1, 11):
            item = observation(seq, 1, True)
            frozen.observe(item)
            memory.observe(item)
        np.testing.assert_array_equal(before, frozen.probabilities())
        self.assertEqual(memory.predict(2).probability, 0.5)
        self.assertGreater(memory.predict(1).probability, 0.9)


class ComponentTests(unittest.TestCase):
    def test_checkpoint_restores_future_learning_and_questions(self):
        config = configuration({"dimensions": 3})
        for name in ("full-history", "window", "drift-reset", "frozen", "memory"):
            for policy_name in ("random", "information-gain"):
                learner = make_learner(config, name)
                policy = create("policy", policy_name, seed=83)
                for seq in range(1, 31):
                    question = policy.choose(learner.probabilities(), tuple(range(8)), 0.1)
                    learner.observe(observation(seq, question.features, bool(question.features & 1) ^ (seq % 7 == 0)))
                state = json.loads(json.dumps({"learner": learner.snapshot(), "policy": policy.snapshot()}))
                restored = make_learner(config, name)
                restored.restore(state["learner"])
                restored_policy = create("policy", policy_name, seed=99)
                restored_policy.restore(state["policy"])
                for seq in range(31, 61):
                    np.testing.assert_array_equal(learner.probabilities(), restored.probabilities())
                    a = policy.choose(learner.probabilities(), tuple(range(8)), 0.1)
                    b = restored_policy.choose(restored.probabilities(), tuple(range(8)), 0.1)
                    self.assertEqual(a, b)
                    item = observation(seq, a.features, bool(a.features & 2) ^ (seq % 11 == 0))
                    self.assertEqual(learner.observe(item), restored.observe(item))

    def test_information_selection_uses_posterior_not_truth(self):
        policy = create("policy", "information-gain", seed=2, exploration=0)
        probabilities = np.array([0.001, 0.5, 0.999, 0.2])
        chosen = policy.choose(probabilities, (0, 1, 2, 3), 0.1)
        self.assertEqual(chosen.features, 1)
        self.assertAlmostEqual(chosen.expected_information_bits, 1 + 0.1 * math.log2(0.1) + 0.9 * math.log2(0.9))
        self.assertNotIn("target", asdict(chosen))

    def test_detector_and_reset_state(self):
        detector = PageHinkley(minimum=20, threshold=5)
        self.assertFalse(any(detector.update(0.1) for _ in range(50)))
        saved = detector.snapshot()
        restored = PageHinkley(minimum=20, threshold=5)
        restored.restore(saved)
        self.assertEqual([detector.update(2.0) for _ in range(10)], [restored.update(2.0) for _ in range(10)])
        self.assertGreater(detector.cumulative - detector.lowest, 5)
        config = configuration({"dimensions": 3})
        learner = make_learner(config, "drift-reset")
        for seq in range(1, 121):
            mask = seq % 8
            learner.observe(observation(seq, mask, bool(mask & 1) ^ (seq > 60)))
        self.assertGreater(learner.resets, 0)
        self.assertLessEqual(len(learner.recent), 8)

    def test_sqlite_replay_and_checkpoint(self):
        connection = sqlite3.connect(":memory:")
        memory = SQLiteMemory(connection, "a")
        learner = BayesianLearner(dimensions=3)
        for seq in range(1, 10):
            item = observation(seq, seq % 8, seq % 2 == 0)
            memory.append(LearningEvent(seq, "teaching", {"observation": asdict(item)}))
            learner.observe(item)
        replay = BayesianLearner(dimensions=3)
        for event in memory.events():
            replay.observe(Observation(**event.payload["observation"]))
        np.testing.assert_array_equal(learner.probabilities(), replay.probabilities())
        memory.checkpoint({"learner": learner.snapshot()})
        self.assertEqual(memory.restore()["learner"], learner.snapshot())
        with self.assertRaises(sqlite3.IntegrityError):
            memory.append(memory.events()[0])
        self.assertEqual(SQLiteMemory(connection, "b").events(), [])
        connection.close()

    def test_reasoner_plugins_signed_literals_and_constants(self):
        direct, rust = DirectReasoner(), RustReasoner()
        hypotheses = hypothesis_space(3)[0]
        for hypothesis in hypotheses:
            for mask in (0, 7):
                a, b = direct.explain(hypothesis, mask, 3), rust.explain(hypothesis, mask, 3)
                self.assertEqual(a["outcome"], b["outcome"])
                self.assertTrue(b["conditional_on_hypothesis"])
                facts = set(b["proof"]["fact_ids"])
                self.assertNotEqual(6 in facts, 7 in facts)
        self.assertIn("noisy-world", catalog()["teacher"])
        with self.assertRaises(ValueError):
            create("learner", "not-a-plugin")


class ExperimentTests(unittest.TestCase):
    def test_default_file_matches_registry_configuration(self):
        path = Path(__file__).resolve().parents[1] / "configs/learning/default.json"
        self.assertEqual(configuration(json.loads(path.read_text())), configuration())

    def test_evaluator_confidence_and_proper_scores(self):
        evaluator = create("evaluator", "heldout-classification")
        scores = evaluator.evaluate(np.array([0.99, 0.01, 0.5, 0.5]), np.array([False, True, False, True]))
        self.assertEqual(scores["confident_count"], 2)
        self.assertEqual(scores["confident_errors"], 2)
        self.assertAlmostEqual(scores["balanced_accuracy"], 0.25)
        self.assertAlmostEqual(scores["brier"], (2 * 0.99**2 + 0.5) / 4)
        with self.assertRaises(ValueError):
            evaluator.evaluate(np.array([float("nan"), 1]), np.array([False, True]))

    def test_fixtures_noise_pairing_and_teacher_boundary(self):
        config = configuration({"stable_worlds": 1, "changing_worlds": 1})
        worlds = make_worlds(config)
        self.assertEqual(worlds, make_worlds(config))
        world = worlds[1]
        self.assertFalse(set(world["question_pool"]) & set(world["held_out"]))
        self.assertFalse(np.array_equal(truth_at(world, 80, 6), truth_at(world, 81, 6)))
        teachers = [create("teacher", "noisy-world", world=world, dimensions=6, noise=0.2, seed=17, steps=240) for _ in range(2)]
        for seq in range(1, 241):
            items = [teacher.answer(world["question_pool"][offset]) for teacher, offset in zip(teachers, (0, -1))]
            clean = truth_at(world, seq, 6)
            self.assertEqual(items[0].label != clean[items[0].features], items[1].label != clean[items[1].features])
            self.assertEqual(set(asdict(items[0])), {"sequence", "features", "label", "source"})
        self.assertEqual(teachers[0].schedule_hash(), teachers[1].schedule_hash())
        with self.assertRaises(ValueError):
            teachers[0].answer(world["held_out"][0])

    def test_small_feature_spaces_remain_splittable(self):
        for dimensions in (2, 3):
            config = configuration({"dimensions": dimensions, "stable_worlds": 1, "changing_worlds": 1})
            for world in make_worlds(config):
                for sequence in (1, 81, 161):
                    truth = truth_at(world, sequence, dimensions)
                    for part in (world["question_pool"], world["held_out"]):
                        self.assertEqual(set(truth[part]), {False, True})

    def test_recovery_censoring_and_confirmation_time(self):
        self.assertEqual(recovery_delays([1] * 40, [10, 30], 40), [
            {"after_question": 10, "confirmed_recovery_delay": 10, "censoring_horizon": 20},
            {"after_question": 30, "confirmed_recovery_delay": 10, "censoring_horizon": 10}])
        self.assertIsNone(recovery_delays([0.5] * 40, [10], 40)[0]["confirmed_recovery_delay"])

    def test_configuration_rejects_unsafe_or_ambiguous_values(self):
        for overrides in ({"surprise": True}, {"noise_rates": [0.5]}, {"changes": [160, 80]},
                          {"seeds": [0, 0]}, {"variants": [{"learner": "missing", "policy": "random"}]},
                          {"window": 0}, {"dimensions": 99}):
            with self.assertRaises(ValueError):
                configuration(overrides)

    def test_end_to_end_artifact_binding_and_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "experiment"
            run({"stable_worlds": 1, "changing_worlds": 1, "steps": 36,
                 "changes": [12, 24], "noise_rates": [0.1], "seeds": [0]}, output)
            manifest = verify_manifest(output)
            self.assertEqual(manifest["runs"], 24)
            self.assertEqual(manifest["teaching_turns"], 864)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["checkpoint_verified_runs"], 24)
            self.assertTrue((output / "REPORT.html").exists())
            connection = sqlite3.connect(output / "memory.sqlite")
            for payload, in connection.execute("SELECT payload FROM events"):
                event = json.loads(payload)
                self.assertEqual(set(event), {"observation", "question", "update"})
                self.assertNotIn("truth", event)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 24)
            connection.close()
            with gzip.open(output / "trace.jsonl.gz", "rt") as stream:
                self.assertEqual(sum(1 for _ in stream), 864)
            with self.assertRaises(ValueError):
                run({}, output)
            with (output / "runs.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify_manifest(output)


if __name__ == "__main__":
    unittest.main()
