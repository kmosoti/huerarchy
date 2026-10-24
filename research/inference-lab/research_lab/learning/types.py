"""Public component contracts. Oracle truth is deliberately absent from these types."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Observation:
    sequence: int
    features: int
    label: bool
    source: str

    def __post_init__(self):
        if self.sequence < 1 or self.features < 0 or type(self.label) is not bool or not self.source:
            raise ValueError("invalid observation")


@dataclass(frozen=True)
class Hypothesis:
    identifier: int
    # None is constant false; () is constant true. Literal ±(feature_index + 1).
    literals: tuple[int, ...] | None

    def evaluate(self, features: int) -> bool:
        return self.literals is not None and all(
            bool(features & (1 << (abs(literal) - 1))) == (literal > 0)
            for literal in self.literals
        )


@dataclass(frozen=True)
class Prediction:
    probability: float
    evidence_count: int


@dataclass(frozen=True)
class Question:
    features: int
    policy: str
    reason: str
    expected_information_bits: float


@dataclass(frozen=True)
class LearningEvent:
    sequence: int
    kind: str
    payload: dict[str, Any]


class Teacher(Protocol):
    def answer(self, features: int) -> Observation: ...
    def schedule_hash(self) -> str: ...


class ExperienceMemory(Protocol):
    def append(self, event: LearningEvent) -> None: ...
    def events(self) -> list[LearningEvent]: ...
    def checkpoint(self, state: dict) -> None: ...
    def restore(self) -> dict: ...


class RuleLearner(Protocol):
    def observe(self, observation: Observation) -> dict: ...
    def probabilities(self) -> np.ndarray: ...
    def predict(self, features: int) -> Prediction: ...
    def explanation_candidate(self) -> tuple[Hypothesis, float] | None: ...
    def snapshot(self) -> dict: ...
    def restore(self, state: dict) -> None: ...


class QuestionPolicy(Protocol):
    def choose(self, probabilities: np.ndarray, candidates: tuple[int, ...], noise: float) -> Question: ...
    def snapshot(self) -> dict: ...
    def restore(self, state: dict) -> None: ...


class ChangeDetector(Protocol):
    def update(self, loss: float) -> bool: ...
    def reset(self) -> None: ...
    def snapshot(self) -> dict: ...
    def restore(self, state: dict) -> None: ...


class Reasoner(Protocol):
    def explain(self, hypothesis: Hypothesis, features: int, dimensions: int) -> dict: ...


class Evaluator(Protocol):
    def evaluate(self, probabilities: np.ndarray, labels: np.ndarray) -> dict: ...


REGISTRY: dict[str, dict[str, tuple[Any, str]]] = {}


def register(category, name, description):
    def decorator(factory):
        entries = REGISTRY.setdefault(category, {})
        if name in entries:
            raise ValueError(f"duplicate component: {category}/{name}")
        entries[name] = (factory, description)
        return factory
    return decorator


def create(category, name, **kwargs):
    try:
        factory = REGISTRY[category][name][0]
    except KeyError:
        raise ValueError(f"unknown component: {category}/{name}") from None
    return factory(**kwargs)


def catalog():
    return {category: {name: description for name, (_, description) in entries.items()}
            for category, entries in REGISTRY.items()}


def observation_dict(observation):
    return asdict(observation)
