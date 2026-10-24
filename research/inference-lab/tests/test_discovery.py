from dataclasses import asdict
import gzip
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_lab.discovery.expansion import CompositionalExpander, ExpansionView, SharedPatternLibrary, initial_programs
from research_lab.discovery.experiment import run, scores, verify
from research_lab.discovery.learner import DiscoveryLearner, QuestionPolicy
from research_lab.discovery.programs import Atom as A, BatchEvaluator, BudgetExhausted, Clause as C, Feedback, Program as P, Scene, reference_closure, rust_audit, validate
from research_lab.discovery.worlds import BOOLEAN_SCHEMA, SCHEMA, PublicTask, Teacher, clean_outcome, configuration, generate_curriculum


def boolean_public():
    scenes = tuple(Scene((), tuple((f"b{i}_{mask >> i & 1}", ()) for i in range(6)), (), f"group-{mask & 3}") for mask in range(64))
    return PublicTask(BOOLEAN_SCHEMA, scenes, scenes)


def compatibility():
    return (A("edge", (0, 1)), A("tag0", (0, 2)), A("tag0", (1, 2)))


def recursive_program(body):
    return P.make((C(A("h0", (0, 1)), body), C(A("h1", (0, 1)), (A("h0", (0, 1)),)),
                   C(A("h1", (0, 1)), (A("h0", (0, 2)), A("h1", (2, 1)))),
                   C(A("out", (0, 1)), (A("h1", (0, 1)),))))


class ProgramTests(unittest.TestCase):
    def test_boolean_language_and_alternatives_against_independent_truth(self):
        public = boolean_public()
        base = initial_programs(public.schema, 0)
        self.assertEqual(len(base), 730)
        engine = BatchEvaluator(public.questions, public.schema)
        for p in base[::17]:
            expected = [("out", ()) in reference_closure(p, s, public.schema) for s in public.questions]
            np.testing.assert_array_equal(engine.evaluate(p), expected)
        xor = P.make((C(A("out"), (A("b0_1"), A("b1_0"))), C(A("out"), (A("b0_0"), A("b1_1")))))
        np.testing.assert_array_equal(engine.evaluate(xor), [bool(mask & 1) != bool(mask & 2) for mask in range(64)])
        self.assertEqual(P.from_dict(xor.to_dict()), xor)

    def test_join_recursion_cycles_and_object_renaming(self):
        config = configuration({"curricula": 2, "scene_count": 12})
        task = generate_curriculum(0, config)["tasks"][-1]
        program = recursive_program(compatibility())
        engine = BatchEvaluator(task.public.evaluation, task.public.schema)
        predicted = engine.evaluate(program)
        for scene, value in zip(task.public.evaluation[:4], predicted[:4]):
            self.assertEqual(bool(value), ("out", scene.query) in reference_closure(program, scene, SCHEMA))
            audit = rust_audit(program, scene, SCHEMA)
            self.assertEqual(audit["outcome"], value)
            n = len(scene.entities)
            permutation = list(reversed(range(n)))
            facts = tuple((p, tuple(permutation[v] if sort == "object" else v for v, sort in zip(args, SCHEMA[p]))) for p, args in scene.facts)
            renamed = Scene(tuple(reversed(scene.entities)), facts, tuple(permutation[v] for v in scene.query), scene.group)
            other = BatchEvaluator((renamed,), SCHEMA).evaluate(program)[0]
            self.assertEqual(bool(other), bool(value))

    def test_types_bounds_and_explicit_negative_observations(self):
        bad = [P.make((C(A("out", (0, 1)), (A("u", (0,)),)),)),
               P.make((C(A("out", (0, 1)), (A("tag0", (0, 1)),)),)),
               P.make((C(A("edge", (0, 1)), (A("edge", (0, 1)),)),))]
        for program in bad:
            with self.assertRaises(ValueError):
                validate(program, SCHEMA, 2)
        with self.assertRaises(ValueError):
            validate(P.make(tuple(C(A("out"), (A(f"b{i}_1"),)) for i in range(5))), BOOLEAN_SCHEMA, 0)
        empty = P()
        self.assertFalse(BatchEvaluator(boolean_public().questions, BOOLEAN_SCHEMA).evaluate(empty).any())

    def test_search_budget_and_scene_rebuild(self):
        public = boolean_public()
        engine = BatchEvaluator(public.questions, BOOLEAN_SCHEMA, budget=64)
        p = P.make((C(A("out"), (A("b0_1"),)),))
        engine.evaluate(p)
        engine.evaluate(p)
        self.assertEqual(engine.context_evaluations, 64)
        with self.assertRaises(BudgetExhausted):
            engine.evaluate(P())
        facts = (("obj", (0,)), ("obj", (1,)), ("edge", (0, 1)))
        present = Scene(("a", "b"), facts, (0, 1), "group")
        removed = Scene(("a", "b"), facts[:2], (0, 1), "group")
        program = recursive_program((A("edge", (0, 1)),))
        np.testing.assert_array_equal(BatchEvaluator((present, removed), SCHEMA).evaluate(program), [True, False])


class ExpansionTests(unittest.TestCase):
    def test_data_guided_proposer_constructs_xor(self):
        public = boolean_public()
        observations = tuple(Feedback(i + 1, i, bool(i & 1) != bool(i & 2), "teacher") for i in range(64))
        engine = BatchEvaluator(public.questions, public.schema)
        view = ExpansionView(public.schema, 0, observations, initial_programs(public.schema, 0), engine, {}, 0.1, np.log(2))
        proposed = CompositionalExpander().propose(view, 2000)
        expected = np.array([o.label for o in observations])
        self.assertTrue(any(np.array_equal(engine.evaluate(p.program), expected) for p in proposed))
        self.assertTrue(any(p.strategy == "alternative-rules" for p in proposed))
        self.assertLessEqual(len(proposed), 2000)
        self.assertEqual(set(asdict(view.observations[0])), {"sequence", "question_index", "label", "source"})

    def test_expansion_requires_fresh_confirmation_and_changes_behavior(self):
        public = boolean_public()
        config = configuration({"steps": 192, "changes": []})
        learner = DiscoveryLearner(public, config, "adaptive")
        accepted = []
        for seq in range(1, 193):
            mask = (seq - 1) % 64
            update = learner.observe(Feedback(seq, mask, bool(mask & 1) != bool(mask & 2), "teacher"))
            for event in update["events"]:
                if event["event"] == "challenge-completed" and event["accepted"]:
                    accepted.append(event)
                    self.assertEqual(event["fresh_responses"], config["confirmation"])
                    self.assertGreater(event["prediction_gain"], 0)
                    self.assertGreater(event["penalized_gain"], config["promotion_margin"])
        self.assertTrue(any(e["candidate"] == "growth" for e in accepted))
        predicted = learner.probabilities() >= 0.5
        self.assertGreater(np.mean(predicted == np.array([bool(i & 1) != bool(i & 2) for i in range(64)])), 0.95)

    def test_library_pays_definition_cost_and_preserves_meaning(self):
        library = SharedPatternLibrary()
        programs = [P.make((C(A("out", (0, 1)), compatibility() + (gate,)),))
                    for gate in (A("u", (0,)), A("u", (1,)), A("v", (1,)))]
        self.assertEqual(library.record(programs[0], "task0"), [])
        self.assertEqual(library.record(programs[1], "task1"), [])
        admitted = library.record(programs[2], "task2")
        self.assertTrue(admitted)
        self.assertGreater(admitted[0]["description_savings_nodes"], 0)
        definition = next(c for c in library.concepts.values() if c.body == tuple(sorted(compatibility())))
        factored = P.make((C(A("out", (0, 1)), (A(definition.head.predicate, (0, 1)), A("u", (0,)))),), (definition.head.predicate,))
        task = generate_curriculum(0, configuration({"curricula": 2, "scene_count": 12}))["tasks"][1]
        original = BatchEvaluator(task.public.questions, SCHEMA).evaluate(programs[0])
        actual = BatchEvaluator(task.public.questions, SCHEMA, library.definitions).evaluate(factored)
        np.testing.assert_array_equal(original, actual)
        self.assertEqual(rust_audit(factored, task.public.questions[0], SCHEMA, library.definitions)["outcome"], actual[0])
        restored = SharedPatternLibrary()
        restored.restore(json.loads(json.dumps(library.snapshot())))
        self.assertEqual(restored.definitions, library.definitions)

    def test_learning_progress_uses_observed_improvement(self):
        public = boolean_public()
        policy = QuestionPolicy("progress", 4, exploration=0)
        for _ in range(16):
            policy.observe("group-0", 1.0)
            for group in ("group-1", "group-2", "group-3"):
                policy.observe(group, 0.69)
        for _ in range(16):
            policy.observe("group-0", 0.1)
            for group in ("group-1", "group-2", "group-3"):
                policy.observe(group, 0.69)
        question = policy.choose(np.full(64, 0.5), public.questions, 0.1)
        self.assertEqual(public.questions[question.index].group, "group-0")

    def test_pending_checkpoint_reproduces_future_updates(self):
        public = boolean_public()
        config = configuration({"steps": 96, "changes": []})
        learner = DiscoveryLearner(public, config, "adaptive")
        for seq in range(1, 41):
            mask = seq % 64
            learner.observe(Feedback(seq, mask, bool(mask & 1) != bool(mask & 2), "teacher"))
        state = json.loads(json.dumps(learner.snapshot()))
        restored = DiscoveryLearner(public, config, "adaptive", initialize_broad=False)
        restored.restore(state)
        for seq in range(41, 81):
            np.testing.assert_array_equal(learner.probabilities(), restored.probabilities())
            mask = seq % 64
            item = Feedback(seq, mask, bool(mask & 1) != bool(mask & 2), "teacher")
            self.assertEqual(learner.observe(item), restored.observe(item))
        self.assertEqual(learner.engine.context_evaluations, restored.engine.context_evaluations)


class WorldAndHarnessTests(unittest.TestCase):
    def test_reproducible_worlds_hidden_teacher_and_long_paths(self):
        config = configuration({"curricula": 2, "scene_count": 12})
        curriculum = generate_curriculum(0, config)
        self.assertEqual([t.to_dict() for t in curriculum["tasks"]], [t.to_dict() for t in generate_curriculum(0, config)["tasks"]])
        relay = curriculum["tasks"][-1]
        self.assertTrue(all(len(s.entities) <= 4 for s in relay.public.questions))
        self.assertTrue(all(len(s.entities) >= 5 for s in relay.public.evaluation))
        self.assertEqual(set(asdict(relay.public)), {"schema", "questions", "evaluation"})
        teachers = [Teacher(relay, 0.2, 7, 240) for _ in range(2)]
        for i in range(1, 241):
            a, b = (teacher.answer(index) for teacher, index in zip(teachers, (0, 1)))
            pa = clean_outcome(relay.public.questions[0], relay.targets[0])
            pb = clean_outcome(relay.public.questions[1], relay.targets[0])
            self.assertEqual(a.label != bool(pa), b.label != bool(pb))
            self.assertNotIn("target", asdict(a))
        teacher = Teacher(relay, 0.2, 8, 240)
        teacher.answer(0)
        saved = teacher.snapshot()
        other = Teacher(relay, 0.2, 99, 240)
        other.restore(saved)
        self.assertEqual(teacher.answer(1), other.answer(1))

    def test_expected_scores_account_for_irreducible_randomness(self):
        score = scores(np.full(4, 0.5), np.full(4, 0.5))
        self.assertEqual(score["balanced_accuracy"], 0.5)
        self.assertEqual(score["expected_brier"], 0.25)
        self.assertEqual(score["irreducible_brier"], 0.25)
        self.assertAlmostEqual(score["expected_log_loss"], np.log(2))

    def test_configuration_validation_and_component_selection(self):
        for overrides in ({"unknown": 1}, {"proposal_limit": 2001}, {"expanded_limit": 129},
                          {"noise_rates": [0.5]}, {"scene_count": 1}, {"seeds": [0, 0]}, {"variants": ["unknown"]},
                          {"components": {"world": "unknown"}}):
            with self.assertRaises(ValueError):
                configuration(overrides)

    def test_end_to_end_binding_replay_and_acquisition_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            run({"workers": 1, "curricula": 2, "steps": 36, "changes": [12, 24], "scene_count": 12,
                 "noise_rates": [0.1], "seeds": [0], "variants": ["small-reset", "adaptive-library"], "diagnostics": True}, output)
            manifest = verify(output)
            self.assertEqual(manifest["task_runs"], 26)
            self.assertEqual(manifest["teaching_turns"], 936)
            records = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
            for record in records:
                self.assertTrue(record["checkpoint_verified"])
                self.assertEqual(record["cumulative_acquisition_labels"], 36 * (record["task_position"] + 1))
                self.assertLessEqual(record["search_program_context_evaluations"], 1000000)
            with gzip.open(output / "trace.jsonl.gz", "rt") as stream:
                self.assertEqual(sum(1 for _ in stream), 936)
            self.assertTrue((output / "REPORT.html").exists())
            with self.assertRaises(ValueError):
                run({}, output)
            with (output / "concepts.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify(output)

    def test_parallel_curricula_preserve_non_timing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"curricula": 2, "steps": 8, "changes": [3, 6], "scene_count": 12,
                      "noise_rates": [0.1], "seeds": [0], "variants": ["small-reset"], "diagnostics": False}
            first, second = Path(tmp) / "sequential", Path(tmp) / "parallel"
            run({**config, "workers": 1}, first)
            run({**config, "workers": 2}, second)
            a = sorted((json.loads(line) for line in (first / "runs.jsonl").read_text().splitlines()), key=lambda r: r["run_id"])
            b = sorted((json.loads(line) for line in (second / "runs.jsonl").read_text().splitlines()), key=lambda r: r["run_id"])
            timing = {"wall_seconds", "cpu_seconds"}
            self.assertEqual([{k: v for k, v in r.items() if k not in timing} for r in a],
                             [{k: v for k, v in r.items() if k not in timing} for r in b])


if __name__ == "__main__":
    unittest.main()
