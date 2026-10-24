from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_lab.active.beliefs import FinitePosterior
from research_lab.active.policies import information_scores, InformationPolicy
from research_lab.active.types import TaskView, configuration
from research_lab.active.worlds import core_tasks, transfer_tasks, Teacher, OracleTask, controls
from research_lab.active.learner import ActiveLearner, evaluate_state, expanded_program
from research_lab.active.concepts import TemplateLibrary, template
from research_lab.active.experiment import Writer, run_task, verify_block
from research_lab.active.legacy import LegacyAdapter
from research_lab.active.study import choose_count, _phase
from research_lab.discovery.programs import Atom as A, Clause as C, Program as P, BatchEvaluator, Feedback, rust_audit, validate, Scene
from research_lab.discovery.worlds import SCHEMA, BOOLEAN_SCHEMA
from research_lab.discovery.learner import DiscoveryLearner, QuestionPolicy
from research_lab.discovery.worlds import PublicTask


class PosteriorTests(unittest.TestCase):
    def test_window_is_batch_bayes_and_two_answers_are_81_over_82(self):
        p = FinitePosterior(np.log([.5, .5]), window=2)
        p.observe(np.log([.9, .1])); p.observe(np.log([.9, .1]))
        self.assertAlmostEqual(p.weights()[0], 81 / 82)
        p.observe(np.log([.1, .9]))
        np.testing.assert_allclose(p.weights(), [.5, .5])

    def test_change_joint_matches_exhaustive_segmentation(self):
        prior = np.array([.3, .7]); hazard = .2
        likelihoods = np.array([[.9, .1], [.8, .2], [.1, .9], [.2, .8]])
        obj = FinitePosterior(np.log(prior), "change", hazard=hazard, branches=16)
        for likelihood in likelihoods:
            obj.observe(np.log(likelihood))
        expected = {}
        for changes in itertools.product((False, True), repeat=3):
            starts = [0] + [i + 1 for i, change in enumerate(changes) if change]
            ends = starts[1:] + [4]
            mass = math.prod(hazard if x else 1 - hazard for x in changes)
            for start, end in zip(starts[:-1], ends[:-1]):
                mass *= np.sum(prior * likelihoods[start:end].prod(axis=0))
            r = 4 - starts[-1]
            expected[r] = expected.get(r, np.zeros(2)) + mass * prior * likelihoods[starts[-1]:].prod(axis=0)
        total = sum(x.sum() for x in expected.values())
        for row, length in zip(np.exp(obj.log_joint), obj.runs):
            np.testing.assert_allclose(row, expected[length] / total)

    def test_branch_pruning_and_future_restore(self):
        obj = FinitePosterior(np.log([.3, .7]), "change", hazard=.2, branches=3)
        for i in range(9):
            obj.observe(np.log([.8, .2] if i < 4 else [.2, .8]))
        self.assertEqual(len(obj.runs), 3)
        self.assertIn(1, obj.runs)
        self.assertGreater(obj.pruned_mass, 0)
        other = FinitePosterior.from_state(json.loads(json.dumps(obj.snapshot())))
        for likelihood in ([.9, .1], [.3, .7]):
            obj.observe(np.log(likelihood)); other.observe(np.log(likelihood))
            np.testing.assert_array_equal(obj.weights(), other.weights())

    def test_no_change_matches_static_bayes(self):
        obj = FinitePosterior(np.log([.5, .5]), "change", hazard=0)
        other = FinitePosterior(np.log([.5, .5]), window=100)
        for value in ([.9, .1], [.8, .2], [.1, .9]):
            obj.observe(np.log(value)); other.observe(np.log(value))
        np.testing.assert_allclose(obj.weights(), other.weights())

    def test_next_answer_prediction_includes_possible_change(self):
        obj = FinitePosterior(np.log([.5, .5]), "change", hazard=.2)
        obj.observe(np.log([.99, .01]))
        np.testing.assert_allclose(obj.next_weights(), [.892, .108])
        self.assertAlmostEqual(obj.next_weights() @ np.array([.1, .9]), .1864)


class PolicyTests(unittest.TestCase):
    def test_randomness_is_not_model_information(self):
        p = np.full((3, 4), .5); w = np.array([.2, .3, .5])
        for kind in ("eig", "epig"):
            np.testing.assert_allclose(information_scores(p, w, kind), 0, atol=1e-12)

    def test_epig_prefers_relevant_prediction(self):
        # Two independent binary latent variables: query 0 reveals the relevant one,
        # query 1 reveals only an irrelevant one. EIG ties, EPIG must separate them.
        p = .1 + .8 * np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        w = np.full(4, .25)
        np.testing.assert_allclose(information_scores(p, w), [information_scores(p, w)[0]] * 2)
        values = information_scores(p, w, "epig", p[:, :1])
        self.assertGreater(values[0], .1)
        self.assertAlmostEqual(values[1], 0)

    def test_policy_restore_preserves_exploration_and_ties(self):
        p = np.array([[.1, .9], [.9, .1]])
        a, b = InformationPolicy("epig", 4, .5), InformationPolicy("epig", 0, .5)
        a.choose(p, np.array([.5, .5])); b.restore(a.snapshot())
        for _ in range(10):
            self.assertEqual(a.choose(p, np.array([.5, .5])), b.choose(p, np.array([.5, .5])))


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration({"steps": 40, "scene_count": 12, "source_steps": 48,
                                     "validation_steps": 24, "validation_warmup": 12, "noise_rates": [.1], "seeds": [0]})

    def test_boundary_world_return_and_paired_feedback(self):
        a = core_tasks(0, "aba", self.config)[-1]
        self.assertEqual(a.targets[0], a.targets[2])
        self.assertEqual(set(asdict(a.view)), {"schema", "questions"})
        self.assertNotIn("evaluation", asdict(a.view))
        t1, t2 = Teacher(a, .2, 5, 40), Teacher(a, .2, 5, 40)
        for index in [0, 1, 4, 2, 7]:
            self.assertEqual(t1.answer(index), t2.answer(index))
        b = OracleTask.from_dict(json.loads(json.dumps(a.to_dict())))
        self.assertEqual(a.to_dict(), b.to_dict())
        t2.restore(t1.snapshot()); self.assertEqual(t1.answer(3), t2.answer(3))

    def test_independent_worlds_vary_mechanisms_not_only_scenes(self):
        families = [core_tasks(i, "abc", self.config) for i in range(8)]
        self.assertGreaterEqual(len({tasks[0].targets[0]["operation"] for tasks in families}), 3)
        self.assertEqual({tasks[-1].targets[0]["tag"] for tasks in families}, {"tag0", "tag1"})

    def test_checkpoint_preserves_updates_queries_and_search_budgets(self):
        task = core_tasks(0, "abc", self.config)[-1]
        a = ActiveLearner(task.view, self.config, "change", archive=True)
        q = InformationPolicy("epig", 5)
        teacher = Teacher(task, .1, 3, 40)
        for _ in range(19):
            index, _ = q.choose(a.conditional, a.posterior.weights()); a.observe(teacher.answer(index))
        b = ActiveLearner(task.view, self.config, "change", archive=True, initialize=False)
        b.restore(json.loads(json.dumps(a.snapshot())))
        r = InformationPolicy("epig", 0); r.restore(q.snapshot())
        for _ in range(15):
            qa = q.choose(a.conditional, a.posterior.weights()); qb = r.choose(b.conditional, b.posterior.weights())
            self.assertEqual(qa, qb)
            feedback = teacher.answer(qa[0]); a.observe(feedback); b.observe(feedback)
            np.testing.assert_array_equal(a.probabilities(), b.probabilities())
            self.assertEqual(a.engine.context_evaluations, b.engine.context_evaluations)

    def test_imported_concept_checkpoint_and_read_only_evaluation(self):
        task = core_tasks(0, "stable", self.config)[-1]
        definition = C(A("compat", (0, 1)), (A("edge", (0, 1)), A("tag1", (0, 2)), A("tag1", (1, 2))))
        lib = {"compat": definition}
        a = ActiveLearner(task.view, self.config, library=lib, transfer=True)
        for i in range(17):
            a.observe(Feedback(i + 1, i % len(task.view.questions), bool(task.outcome(task.view.questions[i % len(task.view.questions)], i + 1)), "teacher"))
        from research_lab.common import digest
        original = digest(a.snapshot())
        evaluate_state(a.prediction_state(), BatchEvaluator(task.evaluation, task.view.schema, lib))
        self.assertEqual(original, digest(a.snapshot()))
        b = ActiveLearner(task.view, self.config, library=lib, transfer=True, initialize=False)
        b.restore(json.loads(json.dumps(a.snapshot())))
        np.testing.assert_array_equal(a.probabilities(), b.probabilities())

    def test_random_data_increases_stochastic_mass(self):
        task = controls(0, self.config)[0]
        c = {**self.config, "proposal_interval": 10000}
        a = ActiveLearner(task.view, c)
        # Exactly balanced repeated answers to every input rule out deterministic labels.
        for i in range(64):
            a.observe(Feedback(i + 1, (i // 2) % len(task.view.questions), bool(i % 2), "teacher"))
        grid = np.array(c["noise_grid"])
        coin_likelihood = .5 ** 64
        deterministic_likelihood = np.mean((grid * (1 - grid)) ** 32)
        expected = .05 * coin_likelihood / (.05 * coin_likelihood + .95 * deterministic_likelihood)
        self.assertAlmostEqual(a.program_weights()[1], expected)
        self.assertLess(np.max(np.abs(a.probabilities() - .5)), .1)

    def test_broad_adapter_matches_historical_predictions_and_questions(self):
        task = core_tasks(0, "stable", self.config)[-1]
        adapter = LegacyAdapter(task.view, self.config, "broad-upfront", 8)
        original = DiscoveryLearner(PublicTask(task.view.schema, task.view.questions, task.view.questions), adapter.inner.config, "broad-upfront")
        query = QuestionPolicy("information", 8, .1)
        teacher = Teacher(task, .1, 2, 40)
        for _ in range(20):
            i, _ = adapter.choose()
            self.assertEqual(i, query.choose(original.probabilities(), task.view.questions, .1).index)
            feedback = teacher.answer(i)
            adapter.observe(feedback); result = original.observe(feedback)
            query.observe(task.view.questions[i].group, result["pre_update_log_loss"])
            np.testing.assert_array_equal(adapter.inner.probabilities(), original.probabilities())

    def test_artifacts_checkpoint_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "block"; w = Writer(out, self.config)
            task = core_tasks(0, "stable", self.config)[-1]
            record, feedback, _ = run_task(w, task, "epig-change", .1, 0, steps=20)
            self.assertEqual(record["acquired_answers"], 20)
            self.assertTrue(record["checkpoint_verified"])
            forced = [f.question_index for f in feedback]
            shadow, _, _ = run_task(w, task, "validation-shadow", .1, 0, steps=20, forced=forced, feedbacks=feedback, role="shadow")
            self.assertEqual(shadow["acquired_answers"], 0)
            w.finish(True); verify_block(out)
            with (out / "runs.jsonl").open("a") as f: f.write("{}\n")
            with self.assertRaises(ValueError): verify_block(out)

    def test_sequential_gate_uses_only_post_proposal_answers(self):
        scenes = tuple(Scene((), tuple((f"b{i}_{mask >> i & 1}", ()) for i in range(6)), (), "g") for mask in range(64))
        view = TaskView(BOOLEAN_SCHEMA, scenes)
        obj = LegacyAdapter(view, self.config, "gate-sequential", 1)
        completed = []
        for sequence in range(1, 145):
            mask = (sequence - 1) % 64
            result = obj.observe(Feedback(sequence, mask, bool(mask & 1) != bool(mask & 2), "teacher"))
            for event in result["events"]:
                if event["event"] == "challenge-completed":
                    self.assertEqual(event["fresh_responses"], event["sequence"] - event["started_after"])
                    self.assertLessEqual(event["fresh_responses"], 64)
                    completed.append(event)
        self.assertTrue(completed)
        self.assertTrue(any(e["accepted"] and e["fresh_responses"] < 32 for e in completed))

    def test_archive_is_bounded_and_only_preserves_supported_programs(self):
        task = core_tasks(0, "stable", self.config)[-1]
        config = {**self.config, "archive_mass": .01, "archive_streak": 2, "archive_limit": 2, "proposal_interval": 1000}
        learner = ActiveLearner(task.view, config, archive=True)
        for i in range(20):
            learner.observe(Feedback(i + 1, i % len(task.view.questions), bool(i % 3), "teacher"))
        self.assertLessEqual(len(learner.archive), 2)
        self.assertTrue(learner.archive)

    def test_parallel_blocks_preserve_non_timing_results(self):
        config = {**self.config, "steps": 2, "proposal_limit": 64, "expanded_limit": 8, "archive_limit": 2}
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / name for name in ("one", "three")]
            for root, workers in zip(roots, (1, 3)):
                (root / "blocks").mkdir(parents=True)
                _phase(root, "core", 2, {**config, "workers": workers}, float("inf"))
            def read(root):
                rows = []
                for path in sorted((root / "blocks").glob("*/runs.jsonl")):
                    for line in path.read_text().splitlines():
                        r = json.loads(line)
                        rows.append({k: v for k, v in r.items() if not k.endswith("seconds") and k != "cpu_before"})
                return rows
            self.assertEqual(read(roots[0]), read(roots[1]))


class ConceptTests(unittest.TestCase):
    def library(self, kind="typed"):
        lib = TemplateLibrary(kind)
        body = (A("edge", (0, 1)), A("tag0", (0, 2)), A("tag0", (1, 2)))
        for i, gate in enumerate((A("u", (0,)), A("u", (1,)), A("v", (1,)))):
            lib.record(P.make((C(A("out", (0, 1)), body + (gate,)),)), str(i), SCHEMA, .95)
        lib.mine(); return lib

    def test_typed_template_survives_predicate_renaming(self):
        body = (A("edge", (0, 1)), A("tag0", (0, 2)), A("tag0", (1, 2)))
        renamed = (A("b", (0, 1)), A("a", (0, 2)), A("a", (1, 2)))
        self.assertEqual(template(body, SCHEMA, True), template(renamed, {"b": ("object", "object"), "a": ("object", "value")}, True))

    def test_instantiation_matches_expansion_and_rust(self):
        lib = self.library(); self.assertTrue(lib.concepts)
        schema = {"p": ("object", "object"), "a": ("object", "value"), "u": ("object",)}
        definitions, _ = lib.instantiate(schema)
        from research_lab.discovery.programs import Scene
        scene = Scene(("x", "y"), (("p", (0, 1)), ("a", (0, 2)), ("a", (1, 2)), ("u", (0,)), ("u", (1,))), (0, 1), "g")
        self.assertTrue(definitions)
        for name, clause in definitions.items():
            imported = P.make((C(A("out", (0, 1)), (A(name, (0, 1)),)),), (name,))
            expanded = P.make((C(A("out", (0, 1)), clause.body),))
            a = BatchEvaluator((scene,), schema, definitions).evaluate(imported)
            b = BatchEvaluator((scene,), schema).evaluate(expanded)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(rust_audit(imported, scene, schema, definitions)["outcome"], a[0])

    def test_promotion_requires_two_distinct_positive_fresh_tasks(self):
        lib = self.library("validated"); key = lib.concepts[0]["id"]
        lib.validate_transfer(key, "a", 10, 32)
        self.assertEqual(lib.concepts[0]["status"], "provisional")
        with self.assertRaises(ValueError): lib.validate_transfer(key, "a", 10, 32)
        lib.validate_transfer(key, "b", -.1, 32)
        self.assertEqual(lib.concepts[0]["status"], "provisional")
        other = self.library("validated"); key = other.concepts[0]["id"]
        other.validate_transfer(key, "a", 2, 32); other.validate_transfer(key, "b", 2, 32)
        self.assertEqual(other.concepts[0]["status"], "promoted")
        restored = TemplateLibrary("validated"); restored.restore(json.loads(json.dumps(other.snapshot())))
        self.assertEqual(restored.instantiate(SCHEMA)[0], other.instantiate(SCHEMA)[0])

    def test_expansion_preserves_meaning_and_enforces_common_bounds(self):
        definition = C(A("c", (0, 1)), (A("edge", (0, 1)), A("tag0", (0, 2)), A("tag0", (1, 2))))
        p = P.make((C(A("out", (0, 1)), (A("c", (0, 1)), A("u", (0,)))),), ("c",))
        expanded = expanded_program(p, {"c": definition})
        validate(expanded, SCHEMA, 2)
        self.assertEqual(expanded.nodes, 5)
        # A helper must not smuggle extra variables into a supposedly three-variable clause.
        other = P.make((C(A("out", (0, 1)), (A("c", (0, 2)), A("edge", (2, 1)))),), ("c",))
        with self.assertRaises(ValueError): validate(expanded_program(other, {"c": definition}), SCHEMA, 2)

    def test_transfer_targets_are_fresh_and_renamed(self):
        tasks = transfer_tasks(0, configuration({"scene_count": 12}))
        self.assertEqual(len(tasks), 7)
        self.assertTrue(tasks[-1].mapping)
        self.assertNotIn("edge", tasks[-1].view.schema)
        for task in tasks[-2:]:
            self.assertEqual({task.outcome(s, 1) for s in task.evaluation}, {0., 1.})
        self.assertTrue(all(len(s.entities) >= 5 for s in tasks[-1].evaluation))

    def test_timing_selection_depends_only_on_budget(self):
        self.assertEqual(choose_count([4, 8, 12, 16], 100, 400, 3, .2), 8)
        self.assertEqual(choose_count([4, 8], float("inf"), 400, 3, .2), 4)


if __name__ == "__main__":
    unittest.main()
