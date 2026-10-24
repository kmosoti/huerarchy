from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from ..common import digest
from ..discovery.programs import Scene


@dataclass(frozen=True)
class TaskView:
    """The entire learner input: no evaluation inputs, labels, phases, or oracle."""
    schema: dict
    questions: tuple[Scene, ...]

    def to_dict(self):
        return {"schema": self.schema, "questions": [s.to_dict() for s in self.questions]}

    @property
    def fingerprint(self):
        return digest(self.to_dict())


class Posterior(Protocol):
    def observe(self, likelihood): ...
    def weights(self): ...
    def snapshot(self): ...


class QueryPolicy(Protocol):
    def choose(self, conditional, weights): ...
    def snapshot(self): ...


DEFAULT = {
    "schema": 1, "fixture_seed": 51024, "steps": 240, "scene_count": 64,
    "noise_rates": [0.0, 0.1, 0.2], "seeds": [0, 1, 2], "workers": 3,
    "world_counts": [4, 8, 12, 16], "study_seconds": 21600,
    "pilot_seconds": 1800, "core_seconds": 12600, "transfer_seconds": 5400,
    "report_seconds": 1800, "timing_reserve": 0.2,
    "noise_grid": [0.01, 0.05, 0.10, 0.20, 0.35], "coin_prior": 0.05,
    "complexity_penalty": math.log(2), "proposal_limit": 2000,
    "expanded_limit": 128, "proposal_interval": 16, "search_context_budget": 1000000,
    "window": 64, "hazard": 0.01, "run_branches": 32, "exploration": 0.1,
    "archive_limit": 16, "archive_mass": 0.9, "archive_streak": 8,
    "concept_limit": 8, "concept_gain": math.log(20),
    "source_steps": 240, "validation_steps": 64, "validation_warmup": 32,
    "components": {"posterior": "finite", "policy": "information", "expander": "typed",
                   "library": "templates", "teacher": "hidden"},
}

CORE = {
    "eig-window": ("eig", "window", False),
    "epig-window": ("epig", "window", False),
    "eig-change": ("eig", "change", False),
    "epig-change": ("epig", "change", False),
    "random-window": ("random", "window", False),
    "epig-change-archive": ("epig", "change", True),
}
LIBRARIES = ("none", "exact", "typed", "validated")


def configuration(overrides=None):
    c = {**DEFAULT, **(overrides or {})}
    if set(c) != set(DEFAULT) or c["schema"] != 1:
        raise ValueError("unknown active configuration or schema")
    positive = ("steps", "scene_count", "workers", "study_seconds", "pilot_seconds", "core_seconds",
                "transfer_seconds", "report_seconds", "proposal_limit", "expanded_limit", "proposal_interval",
                "search_context_budget", "window", "run_branches", "archive_limit", "archive_streak",
                "concept_limit", "source_steps", "validation_steps", "validation_warmup")
    if any(type(c[k]) is not int or c[k] < 1 for k in positive):
        raise ValueError("active counts and budgets must be positive integers")
    if c["workers"] > 3 or c["scene_count"] < 12 or c["proposal_limit"] > 2000 or c["expanded_limit"] > 128:
        raise ValueError("active resource bounds exceeded")
    if c["archive_limit"] > c["expanded_limit"] or c["validation_warmup"] >= c["validation_steps"]:
        raise ValueError("invalid archive or validation bounds")
    for key in ("coin_prior", "hazard", "exploration", "timing_reserve", "archive_mass"):
        if not math.isfinite(c[key]) or not 0 <= c[key] < 1:
            raise ValueError(f"invalid probability: {key}")
    for key in ("noise_grid", "noise_rates"):
        if not c[key] or any(not math.isfinite(x) or not 0 <= x < .5 for x in c[key]):
            raise ValueError("invalid noise grid")
    if any(x == 0 for x in c["noise_grid"]):
        raise ValueError("model noise grid must be strictly positive")
    if not 0 < c["coin_prior"] < 1 or c["run_branches"] < 2:
        raise ValueError("coin prior and branch count must retain uncertainty")
    for key in ("seeds", "world_counts"):
        if not c[key] or len(set(c[key])) != len(c[key]) or any(type(x) is not int or x < (1 if key == "world_counts" else 0) for x in c[key]):
            raise ValueError("invalid seeds/world counts")
    if any(not math.isfinite(c[k]) or c[k] < 0 for k in ("complexity_penalty", "concept_gain")):
        raise ValueError("invalid evidence penalty")
    if c["components"] != DEFAULT["components"]:
        from ..learning.types import REGISTRY
        for key, value in c["components"].items():
            if key not in DEFAULT["components"] or value not in REGISTRY.get("active-" + key, {}):
                raise ValueError("unknown active component")
        if set(c["components"]) != set(DEFAULT["components"]):
            raise ValueError("missing active component")
    return c
