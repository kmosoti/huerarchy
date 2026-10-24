"""Evaluator-owned worlds and a narrow, noisy teacher interface."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from ..common import digest
from .types import Observation, register


DEFAULT_CONFIG = {
    "schema": 1, "fixture_seed": 42, "dimensions": 6,
    "stable_worlds": 6, "changing_worlds": 6, "diagnostic": True,
    "steps": 240, "changes": [80, 160], "noise_rates": [0.0, 0.1, 0.2],
    "seeds": [0, 1, 2, 3, 4], "assumed_noise": 0.1, "window": 32, "exploration": 0.1,
    "components": {"teacher": "noisy-world", "memory": "sqlite", "detector": "page-hinkley",
                   "evaluator": "heldout-classification", "reasoners": ["direct", "rust-horn"]},
    "detector": {"minimum": 20, "tolerance": 0.05, "threshold": 5.0},
    "variants": [
        {"learner": learner, "policy": policy}
        for learner in ("full-history", "window", "drift-reset")
        for policy in ("random", "information-gain")
    ] + [{"learner": "frozen", "policy": "random"}, {"learner": "memory", "policy": "random"}],
}


def configuration(overrides=None):
    config = deepcopy(DEFAULT_CONFIG)
    overrides = overrides or {}
    if set(overrides) - set(config):
        raise ValueError(f"unknown configuration keys: {sorted(set(overrides) - set(config))}")
    config.update(overrides)
    integer_fields = ("fixture_seed", "dimensions", "stable_worlds", "changing_worlds", "steps", "window")
    if any(type(config[key]) is not int for key in integer_fields):
        raise ValueError("integer configuration fields must be integers")
    if config["schema"] != 1 or not 2 <= config["dimensions"] <= 8 or config["steps"] < 2 or config["window"] < 1:
        raise ValueError("invalid schema/dimensions/steps/window")
    if min(config["stable_worlds"], config["changing_worlds"]) < 0 or config["stable_worlds"] + config["changing_worlds"] < 1:
        raise ValueError("at least one ordinary world required")
    changes = config["changes"]
    if any(type(t) is not int or not 0 < t < config["steps"] for t in changes) or sorted(set(changes)) != changes:
        raise ValueError("changes must be sorted unique steps inside the experiment")
    if not config["noise_rates"] or any(not isinstance(n, (float, int)) or not math.isfinite(n) or not 0 <= n < 0.5 for n in config["noise_rates"]):
        raise ValueError("feedback noise must be in [0, 0.5)")
    if not config["seeds"] or any(type(s) is not int or s < 0 for s in config["seeds"]):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(config["seeds"])) != len(config["seeds"]) or len(set(config["noise_rates"])) != len(config["noise_rates"]):
        raise ValueError("duplicate seeds/noise rates")
    if not 0 < config["assumed_noise"] < 0.5 or not 0 <= config["exploration"] <= 1:
        raise ValueError("invalid learner noise/exploration")
    from .types import REGISTRY
    from . import components, evaluation  # register built-in plugins
    selected = config["components"]
    if set(selected) != {"teacher", "memory", "detector", "evaluator", "reasoners"}:
        raise ValueError("invalid component selection")
    for category in ("teacher", "memory", "detector", "evaluator"):
        if selected[category] not in REGISTRY.get(category, {}):
            raise ValueError(f"unknown {category} component")
    if (not selected["reasoners"] or len(set(selected["reasoners"])) != len(selected["reasoners"])
            or any(name not in REGISTRY["reasoner"] for name in selected["reasoners"])):
        raise ValueError("invalid reasoner selection")
    if not config["variants"]:
        raise ValueError("variants cannot be empty")
    seen = set()
    for variant in config["variants"]:
        if set(variant) != {"learner", "policy"} or any(variant[k] not in REGISTRY[k] for k in variant):
            raise ValueError("unknown learner/policy variant")
        key = (variant["learner"], variant["policy"])
        if key in seen:
            raise ValueError("duplicate component variant")
        seen.add(key)
    from .types import create
    create("detector", selected["detector"], **config["detector"])
    return config


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")


def evaluate_target(target, features):
    # Independent oracle implementation: does not call the learner or reasoner.
    if target["kind"] == "xor":
        a, b = target["features"]
        return bool((features >> a & 1) ^ (features >> b & 1))
    for condition in target["conditions"]:
        if (features >> condition["feature"] & 1) != condition["value"]:
            return False
    return True


def target_table(target, dimensions):
    return np.array([evaluate_target(target, x) for x in range(1 << dimensions)], dtype=bool)


def _target(rng, dimensions):
    # Each target needs at least two positive examples for disjoint question/test halves.
    indices = rng.choice(dimensions, size=int(rng.integers(1, min(3, dimensions - 1) + 1)), replace=False)
    return {"kind": "conjunction", "conditions": [
        {"feature": int(index), "value": int(rng.integers(2))} for index in sorted(indices)]}


def make_worlds(config):
    rng = np.random.default_rng(config["fixture_seed"])
    result = []
    total = config["stable_worlds"] + config["changing_worlds"] + int(config["diagnostic"])
    for index in range(total):
        diagnostic = index == total - 1 and config["diagnostic"]
        changing = not diagnostic and index >= config["stable_worlds"]
        changes = config["changes"] if changing else []
        targets = [{"kind": "xor", "features": [0, 1]}] if diagnostic else [_target(rng, config["dimensions"])]
        for _ in changes:
            previous = target_table(targets[-1], config["dimensions"])
            for attempt in range(10000):
                proposed = _target(rng, config["dimensions"])
                if np.mean(previous != target_table(proposed, config["dimensions"])) >= 0.25:
                    targets.append(proposed)
                    break
            else:
                raise RuntimeError("could not generate distinct world phases")
        # Evaluation-only stratification: both halves include both outcomes in each phase.
        for attempt in range(10000):
            shuffled = rng.permutation(1 << config["dimensions"])
            pool, held_out = np.split(shuffled, 2)
            if all(len(set(target_table(target, config["dimensions"])[part])) == 2
                   for target in targets for part in (pool, held_out)):
                break
        else:
            raise RuntimeError("could not create an evaluable train/test split")
        aliases = [f"v_{rng.bytes(3).hex()}" for _ in range(config["dimensions"])]
        result.append({"id": f"world-{index:02d}", "kind": "diagnostic" if diagnostic else "changing" if changing else "stable",
                       "attributes": aliases, "changes": list(changes), "targets": targets,
                       "question_pool": sorted(map(int, pool)), "held_out": sorted(map(int, held_out))})
    return result


def truth_at(world, sequence, dimensions):
    # A change after question 80 means question 81 uses the new target.
    phase = sum(sequence > boundary for boundary in world["changes"])
    return target_table(world["targets"][phase], dimensions)


@register("teacher", "noisy-world", "Membership labels with independent flips and hidden abrupt changes")
class NoisyTeacher:
    """Only answer() crosses into the learner's component orchestration."""
    def __init__(self, world, dimensions, noise, seed, steps):
        self.__world = world
        self.__dimensions = dimensions
        self.__flips = np.random.default_rng(seed).random(steps) < noise
        self.__steps, self.__sequence = steps, 0
        self.__pool = set(world["question_pool"])

    def answer(self, features):
        if features not in self.__pool or self.__sequence >= self.__steps:
            raise ValueError("question outside pool or exhausted teaching budget")
        self.__sequence += 1
        target = self.__world["targets"][sum(self.__sequence > t for t in self.__world["changes"])]
        label = evaluate_target(target, features) ^ bool(self.__flips[self.__sequence - 1])
        return Observation(self.__sequence, features, label, f"teacher:{self.__world['id']}")

    def schedule_hash(self):
        return digest(self.__flips.tolist())
