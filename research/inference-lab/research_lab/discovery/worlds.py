"""Evaluator-owned procedural worlds. Public tasks contain no hidden mechanisms."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..learning.worlds import seed_for
from ..common import digest
from ..learning.types import REGISTRY, register
from .programs import Feedback, Scene


SCHEMA = {"obj": ("object",), "edge": ("object", "object"),
          "tag0": ("object", "value"), "tag1": ("object", "value"),
          "u": ("object",), "nu": ("object",), "v": ("object",), "nv": ("object",)}
BOOLEAN_SCHEMA = {f"b{i}_{value}": () for i in range(6) for value in (0, 1)}


@dataclass(frozen=True)
class PublicTask:
    schema: dict
    questions: tuple[Scene, ...]
    evaluation: tuple[Scene, ...]

    def to_dict(self):
        return {"schema": self.schema, "questions": [s.to_dict() for s in self.questions],
                "evaluation": [s.to_dict() for s in self.evaluation]}


@dataclass(frozen=True)
class HiddenTask:
    identifier: str
    family: str
    public: PublicTask
    targets: tuple[dict, ...]
    changes: tuple[int, ...]
    diagnostic: bool = False

    def phase(self, sequence):
        return sum(sequence > t for t in self.changes)

    def to_dict(self):
        return {"identifier": self.identifier, "family": self.family, "public": self.public.to_dict(),
                "targets": self.targets, "changes": self.changes, "diagnostic": self.diagnostic}


def clean_outcome(scene, target):
    facts = set(scene.facts)
    if target["kind"] == "coin":
        return 0.5
    if target["kind"] == "unsupported":
        # Four-bit parity needs eight conjunction clauses. The public language allows four.
        return float(sum((f"b{i}_1", ()) in facts for i in range(4)) % 2 == 1)
    if target["kind"] == "distraction":
        return float(("b1_1", ()) in facts) if ("b0_1", ()) in facts else 0.5
    if target["kind"] == "boolean":
        bits = [int((f"b{i}_1", ()) in facts) for i in range(6)]
        a, b, c = (bits[i] ^ flip for i, flip in zip(target["indices"], target["flips"]))
        operation = target["operation"]
        return float({"and": bool(a and b and c), "or": bool(a or b),
                      "xor": bool(a != b), "majority": a + b + c >= 2}[operation])
    x, y = scene.query
    tags = {args[0]: args[1] for pred, args in scene.facts if pred == target["tag"]}
    edges = {(args[0], args[1]) for pred, args in scene.facts if pred == "edge"}
    if target["reverse"]:
        edges = {(b, a) for a, b in edges}
    compatible = {(a, b) for a, b in edges if tags[a] == tags[b]}
    if target["kind"] == "pair":
        gate_predicate, gate_side = target["gate"]
        return float((x, y) in compatible and (gate_predicate, ((x, y)[gate_side],)) in facts)
    # Independent graph traversal, not rule-program execution.
    visited, frontier = {x}, [x]
    while frontier:
        vertex = frontier.pop()
        for a, b in compatible:
            if a == vertex and b not in visited:
                visited.add(b)
                frontier.append(b)
    # Paths must have positive length; source and destination are distinct in generated tasks.
    return float(y in visited)


def _scene(rng, objects, target=None, desired=None, relay=False, evaluation=False):
    n = objects
    tags = rng.integers(3, size=(2, n))
    edges = {(i, j) for i in range(n) for j in range(n) if i != j and rng.random() < 0.3}
    u, v = rng.integers(2, size=(2, n))
    query = (0, n - 1)
    if relay:
        # Short training chains and longer evaluation chains. Back edges introduce cycles,
        # but no forward shortcuts allow a fixed short-path rule to pass the long test.
        edges = {(i, i + 1) for i in range(n - 1)}
        edges |= {(i + 1, i) for i in range(n - 1) if rng.random() < 0.5}
        tag_index = int(target["tag"][-1])
        tags[tag_index, :] = int(rng.integers(3))
        if target["reverse"]:
            edges = {(b, a) for a, b in edges}
        if not desired:
            if rng.random() < 0.5:
                broken = int(rng.integers(n - 1))
                edges.discard((broken + 1, broken) if target["reverse"] else (broken, broken + 1))
            else:
                changed = int(rng.integers(n))
                tags[tag_index, changed] = (tags[tag_index, changed] + 1) % 3
    facts = [("obj", (i,)) for i in range(n)] + [("edge", edge) for edge in sorted(edges)]
    facts += [(f"tag{kind}", (i, int(tags[kind, i]))) for kind in range(2) for i in range(n)]
    facts += [("u" if u[i] else "nu", (i,)) for i in range(n)]
    facts += [("v" if v[i] else "nv", (i,)) for i in range(n)]
    # Pure renaming: identifiers never reveal a role or the outcome.
    permutation = rng.permutation(n)
    renamed = []
    for predicate, args in facts:
        renamed.append((predicate, tuple(int(permutation[a]) if sort == "object" else a for a, sort in zip(args, SCHEMA[predicate]))))
    names = tuple(f"object_{rng.bytes(4).hex()}" for _ in range(n))
    query = tuple(int(permutation[x]) for x in query)
    group = f"group-{int(u[0]) * 2 + int(v[-1])}"
    return Scene(names, tuple(sorted(renamed)), query, group)


def _balanced_scenes(rng, targets, count, evaluation, relay):
    scenes = []
    for index in range(count):
        target = targets[index % len(targets)]
        desired = bool((index // len(targets)) % 2)
        for _ in range(20000):
            n = int(rng.integers(5, 9) if evaluation else rng.integers(2, 5) if relay else rng.integers(2, 6))
            if evaluation and not relay:
                n = int(rng.integers(6, 9))
            scene = _scene(rng, n, target, desired, relay, evaluation)
            if bool(clean_outcome(scene, target)) == desired:
                scenes.append(scene)
                break
        else:
            raise RuntimeError("world generator could not create requested example")
    return tuple(scenes)


@register("discovery-world", "workshops", "Mechanism, pair, and relay teaching curricula")
def generate_curriculum(index, config):
    rng = np.random.default_rng(seed_for("discovery-fixture", config["fixture_seed"], index))
    changing = index >= config["curricula"] // 2
    changes = tuple(config["changes"]) if changing else ()
    phases = 1 + len(changes)
    motifs = [{"tag": "tag0" if (index + phase) % 2 == 0 else "tag1", "reverse": bool(phase == 2)} for phase in range(phases)]
    tasks = []
    operations = ("and", "or", "xor", "majority", "xor", "majority")
    boolean_targets = tuple({"kind": "boolean", "operation": operations[(index + phase) % len(operations)],
                             "indices": list(map(int, rng.choice(6, 3, replace=False))),
                             "flips": list(map(int, rng.integers(2, size=3)))} for phase in range(phases))
    shuffled = rng.permutation(64)
    boolean_scenes = []
    for mask in shuffled:
        facts = tuple((f"b{i}_{int(mask >> i & 1)}", ()) for i in range(6))
        boolean_scenes.append(Scene((), facts, (), f"group-{mask & 3}"))
    # Stratification is evaluator-only and stated in the report.
    for _ in range(10000):
        rng.shuffle(boolean_scenes)
        if all(len({clean_outcome(s, target) for s in part}) == 2
               for target in boolean_targets for part in (boolean_scenes[:32], boolean_scenes[32:])):
            break
    tasks.append(HiddenTask(f"c{index}/mechanism", "mechanism", PublicTask(BOOLEAN_SCHEMA, tuple(boolean_scenes[:32]), tuple(boolean_scenes[32:])), boolean_targets, changes))
    for task_index, gate in enumerate((("u", 0), ("u", 1), ("v", 1))):
        targets = tuple({"kind": "pair", **motif, "gate": gate} for motif in motifs)
        pool = _balanced_scenes(rng, targets, config["scene_count"], False, False)
        evaluation = _balanced_scenes(rng, targets, config["scene_count"], True, False)
        tasks.append(HiddenTask(f"c{index}/pair-{task_index}", "pair", PublicTask(SCHEMA, pool, evaluation), targets, changes))
    targets = tuple({"kind": "relay", **motif} for motif in motifs)
    pool = _balanced_scenes(rng, targets, config["scene_count"], False, True)
    evaluation = _balanced_scenes(rng, targets, config["scene_count"], True, True)
    tasks.append(HiddenTask(f"c{index}/relay", "relay", PublicTask(SCHEMA, pool, evaluation), targets, changes))
    return {"id": index, "changing": changing, "tasks": tuple(tasks)}


def diagnostic_tasks(config):
    rng = np.random.default_rng(seed_for("diagnostics", config["fixture_seed"]))
    masks = rng.permutation(64)
    scenes = tuple(Scene((), tuple((f"b{i}_{int(mask >> i & 1)}", ()) for i in range(6)), (), f"group-{mask & 3}") for mask in masks)
    public = PublicTask(BOOLEAN_SCHEMA, scenes[:32], scenes[32:])
    return tuple(HiddenTask(f"diagnostic/{kind}", kind, public, ({"kind": kind},), (), True) for kind in ("coin", "unsupported", "distraction"))


@register("discovery-teacher", "noisy", "Evaluator-owned targets with independently flipped teaching responses")
class Teacher:
    def __init__(self, task, noise, seed, steps):
        self.__task, self.__noise = task, noise
        self.__rng = np.random.default_rng(seed)
        self.__sequence, self.__steps = 0, steps

    def answer(self, question_index):
        if not 0 <= question_index < len(self.__task.public.questions) or self.__sequence >= self.__steps:
            raise ValueError("invalid question or exhausted teacher")
        self.__sequence += 1
        target = self.__task.targets[self.__task.phase(self.__sequence)]
        p = clean_outcome(self.__task.public.questions[question_index], target)
        # Always consume both draws: corresponding methods share flip/random schedules.
        latent = self.__rng.random() < p
        flip = self.__rng.random() < self.__noise
        return Feedback(self.__sequence, question_index, bool(latent ^ flip), f"teacher:{self.__task.identifier}")

    def snapshot(self):
        return {"schema": 1, "task_sha256": digest(self.__task.to_dict()),
                "sequence": self.__sequence, "rng": self.__rng.bit_generator.state}

    def restore(self, state):
        if (state["schema"] != 1 or state["task_sha256"] != digest(self.__task.to_dict())
                or not 0 <= state["sequence"] <= self.__steps):
            raise ValueError("teacher checkpoint mismatch")
        self.__sequence = state["sequence"]
        self.__rng.bit_generator.state = state["rng"]


DEFAULT = {"schema": 1, "fixture_seed": 42, "workers": 3, "curricula": 6, "steps": 240, "changes": [80, 160],
           "noise_rates": [0.0, 0.1, 0.2], "seeds": [0, 1, 2, 3, 4], "scene_count": 64,
           "assumed_noise": 0.1, "proposal_limit": 2000, "expanded_limit": 128,
           "search_context_budget": 1000000, "confirmation": 32, "minimum_observations": 32,
           "proposal_interval": 16, "fit_window": 64, "exploration": 0.1,
           "complexity_penalty": math.log(2), "promotion_margin": math.log(10),
           "components": {"world": "workshops", "teacher": "noisy", "expander": "compositional",
                          "library": "shared-patterns", "policy": "questions"},
           "diagnostics": True,
           "variants": ["small-reset", "broad-upfront", "adaptive", "adaptive-library", "library-random", "library-progress"]}


def configuration(overrides=None):
    config = {**DEFAULT, **(overrides or {})}
    if set(config) != set(DEFAULT) or config["schema"] != 1:
        raise ValueError("unknown discovery configuration or schema")
    if type(config["workers"]) is not int or not 1 <= config["workers"] <= 3:
        raise ValueError("workers must be between one and three")
    for key in ("curricula", "steps", "scene_count", "proposal_limit", "expanded_limit", "search_context_budget",
                "confirmation", "minimum_observations", "proposal_interval", "fit_window"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"invalid positive integer: {key}")
    if config["scene_count"] < 12 or config["proposal_limit"] > 2000 or config["expanded_limit"] > 128:
        raise ValueError("scene count too small or search exceeds public bounds")
    if sorted(set(config["changes"])) != config["changes"] or any(not 0 < t < config["steps"] for t in config["changes"]):
        raise ValueError("invalid change schedule")
    if not config["noise_rates"] or any(not math.isfinite(n) or not 0 <= n < 0.5 for n in config["noise_rates"]):
        raise ValueError("invalid feedback noise")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]) or any(type(s) is not int or s < 0 for s in config["seeds"]):
        raise ValueError("invalid seeds")
    if not 0 < config["assumed_noise"] < 0.5 or not 0 <= config["exploration"] <= 1:
        raise ValueError("invalid model noise or exploration")
    if any(not math.isfinite(config[k]) or config[k] < 0 for k in ("complexity_penalty", "promotion_margin")):
        raise ValueError("invalid complexity or confirmation penalty")
    if not config["variants"] or set(config["variants"]) - set(DEFAULT["variants"]) or len(set(config["variants"])) != len(config["variants"]):
        raise ValueError("unknown or duplicate variant")
    from . import expansion, learner  # explicit registration, no arbitrary imports from config
    if set(config["components"]) != {"world", "teacher", "expander", "library", "policy"}:
        raise ValueError("invalid discovery component selection")
    for category, name in config["components"].items():
        if name not in REGISTRY.get("discovery-" + category, {}):
            raise ValueError(f"unknown discovery component: {category}/{name}")
    return config
