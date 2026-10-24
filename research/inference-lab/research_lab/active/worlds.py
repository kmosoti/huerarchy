"""Private experimental fixtures. Only TaskView crosses into learning components."""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from ..common import digest
from ..discovery.programs import Feedback, Scene
from ..discovery.worlds import configuration as legacy_config, generate_curriculum, diagnostic_tasks, clean_outcome, _scene
from ..learning.types import register
from ..learning.worlds import seed_for
from .types import TaskView


@dataclass(frozen=True)
class OracleTask:
    identifier: str
    family: str
    view: TaskView
    evaluation: tuple[Scene, ...]
    targets: tuple[dict, ...]
    changes: tuple[int, ...] = ()
    mapping: tuple[tuple[str, str], ...] = ()  # private surface -> oracle predicate map
    noise_schedule: tuple[float, ...] = ()

    def phase(self, sequence):
        return sum(sequence > boundary for boundary in self.changes)

    def outcome(self, scene, sequence):
        if self.mapping:
            names = dict(self.mapping)
            scene = replace(scene, facts=tuple((names[p], a) for p, a in scene.facts))
        target = self.targets[min(self.phase(sequence), len(self.targets) - 1)]
        if target["kind"] == "pair-combination":
            first = clean_outcome(scene, {**target, "kind": "pair"})
            predicate, side = target["gate2"]
            return first * float((predicate, (scene.query[side],)) in scene.facts)
        return clean_outcome(scene, target)

    def to_dict(self):
        return {"identifier": self.identifier, "family": self.family, "view": self.view.to_dict(),
                "evaluation": [s.to_dict() for s in self.evaluation], "targets": self.targets,
                "changes": self.changes, "mapping": self.mapping, "noise_schedule": self.noise_schedule}

    @classmethod
    def from_dict(cls, v):
        view = TaskView({k: tuple(t) for k, t in v["view"]["schema"].items()},
                        tuple(Scene.from_dict(s) for s in v["view"]["questions"]))
        return cls(v["identifier"], v["family"], view, tuple(Scene.from_dict(s) for s in v["evaluation"]),
                   tuple(v["targets"]), tuple(v["changes"]), tuple(tuple(x) for x in v["mapping"]), tuple(v["noise_schedule"]))


def _base(block, config, namespace="core"):
    c = legacy_config({"curricula": 2, "fixture_seed": seed_for("active-world", config["fixture_seed"], namespace, block),
                       "scene_count": config["scene_count"]})
    # All indices >=1 are changing in a two-curriculum generator. Vary the
    # index as well as the RNG so independent blocks cover different Boolean
    # operations and attribute motifs, not merely different scene instances.
    index = 1 + seed_for("active-structure", config["fixture_seed"], namespace, block) % 60
    return generate_curriculum(index, c)["tasks"]


def core_tasks(block, regime, config):
    rng = np.random.default_rng(seed_for("active-boundaries", config["fixture_seed"], block))
    durations = rng.permutation([60, 80, 100])
    boundaries = tuple(int(x) for x in np.cumsum(durations)[:2] * config["steps"] / 240)
    boundaries = tuple(sorted(set(max(1, min(config["steps"] - 1, x)) for x in boundaries)))
    result = []
    for i, t in enumerate(_base(block, config)):
        targets = t.targets if regime == "abc" else (t.targets[0], t.targets[1], t.targets[0]) if regime == "aba" else (t.targets[0],)
        result.append(OracleTask(f"b{block}/{regime}/{t.family}-{i}", t.family,
                                TaskView(t.public.schema, t.public.questions), t.public.evaluation,
                                targets, boundaries if regime != "stable" else ()))
    return tuple(result)


def controls(block, config):
    c = legacy_config({"fixture_seed": seed_for("active-controls", config["fixture_seed"], block)})
    result = [OracleTask(f"b{block}/diagnostic/{t.family}", t.family, TaskView(t.public.schema, t.public.questions),
                         t.public.evaluation, t.targets) for t in diagnostic_tasks(c)]
    task = core_tasks(block, "abc", config)[-1]
    result.append(replace(task, identifier=f"b{block}/diagnostic/noise-shift", family="noise-shift",
                          targets=(task.targets[0],) * 3, noise_schedule=(.05, .20, .05)))
    return tuple(result)


def _stable_task(t, name):
    return OracleTask(name, t.family, TaskView(t.public.schema, t.public.questions), t.public.evaluation, (t.targets[0],))


def renamed(task, rng):
    names = sorted(task.view.schema)
    shuffled = list(rng.permutation(len(names)))
    mapping = {p: f"r{int(i):02d}" for p, i in zip(names, shuffled)}
    def transform(s):
        return replace(s, facts=tuple(sorted((mapping[p], a) for p, a in s.facts)))
    return replace(task, view=TaskView({mapping[p]: t for p, t in task.view.schema.items()}, tuple(map(transform, task.view.questions))),
                   evaluation=tuple(map(transform, task.evaluation)), mapping=tuple((v, k) for k, v in mapping.items()))


def transfer_tasks(block, config):
    # All tasks share a hidden structural motif, but target surface names and scenes are fresh.
    base = _base(block, config, "transfer-source")
    motif = base[1].targets[0]
    sources = [_stable_task(t, f"b{block}/transfer/source-{i}") for i, t in enumerate(base[1:4])]
    validations = []
    for i in range(2):
        task = _base(block, config, f"transfer-validation-{i}")[1 + i]
        # Regenerate balanced scenes for the chosen common motif, using no source scenes.
        target = {**motif, "gate": ("u", 1) if i == 0 else ("v", 0)}
        validations.append(_new_pair(task, target, f"b{block}/transfer/validation-{i}", config))
    task = _base(block, config, "transfer-target")[1]
    combination = _new_pair(task, {**motif, "kind": "pair-combination", "gate": ("u", 0), "gate2": ("v", 1)},
                            f"b{block}/transfer/recombined", config)
    from ..discovery.worlds import _balanced_scenes
    rng = np.random.default_rng(seed_for("transfer-relay", config["fixture_seed"], block))
    target = {"kind": "relay", "tag": motif["tag"], "reverse": motif["reverse"]}
    relay = OracleTask(f"b{block}/transfer/relay", "relay", TaskView(task.public.schema,
        _balanced_scenes(rng, (target,), config["scene_count"], False, True)),
        _balanced_scenes(rng, (target,), config["scene_count"], True, True), (target,))
    return tuple(sources + validations + [renamed(combination, rng), renamed(relay, rng)])


def _new_pair(template, target, name, config):
    rng = np.random.default_rng(seed_for("active-pair", config["fixture_seed"], name))
    task = OracleTask(name, "pair", TaskView(template.public.schema, ()), (), (target,))
    def pool(evaluation):
        result = []
        for i in range(config["scene_count"]):
            for _ in range(20000):
                s = _scene(rng, int(rng.integers(6, 9) if evaluation else rng.integers(2, 6)))
                if bool(task.outcome(s, 1)) == bool(i % 2):
                    result.append(s)
                    break
            else:
                raise RuntimeError("failed to generate transfer scene")
        return tuple(result)
    return replace(task, view=TaskView(template.public.schema, pool(False)), evaluation=pool(True))


@register("active-teacher", "hidden", "Private target/noise schedule with paired response randomness")
class Teacher:
    def __init__(self, task, noise, seed, steps):
        self._task, self._noise, self._steps = task, noise, steps
        self._rng = np.random.default_rng(seed)
        self.sequence = 0

    def answer(self, index):
        if not 0 <= index < len(self._task.view.questions) or self.sequence >= self._steps:
            raise ValueError("invalid or exhausted teaching query")
        self.sequence += 1
        phase = self._task.phase(self.sequence)
        noise = self._task.noise_schedule[min(phase, len(self._task.noise_schedule) - 1)] if self._task.noise_schedule else self._noise
        outcome = self._rng.random() < self._task.outcome(self._task.view.questions[index], self.sequence)
        flipped = self._rng.random() < noise
        return Feedback(self.sequence, index, bool(outcome ^ flipped), "teacher")

    def snapshot(self):
        return {"task": digest(self._task.to_dict()), "sequence": self.sequence, "noise": self._noise,
                "steps": self._steps, "rng": self._rng.bit_generator.state}

    def restore(self, state):
        if (state["task"], state["noise"], state["steps"]) != (digest(self._task.to_dict()), self._noise, self._steps):
            raise ValueError("teacher checkpoint mismatch")
        self.sequence = state["sequence"]
        self._rng.bit_generator.state = state["rng"]
