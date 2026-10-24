from __future__ import annotations

from collections import deque
from dataclasses import asdict
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import zlib

import numpy as np

from ..common import ROOT, canonical
from .types import Hypothesis, LearningEvent, Observation, Prediction, Question, create, register


@lru_cache(maxsize=8)
def hypothesis_space(dimensions=6):
    if not 2 <= dimensions <= 8:
        raise ValueError("dimensions must be between 2 and 8")
    hypotheses = [Hypothesis(0, None)]
    for signs in itertools.product((-1, 0, 1), repeat=dimensions):
        literals = tuple(sign * (index + 1) for index, sign in enumerate(signs) if sign)
        hypotheses.append(Hypothesis(len(hypotheses), literals))
    table = np.array([[h.evaluate(x) for x in range(1 << dimensions)] for h in hypotheses], dtype=np.float64)
    prior = np.array([-len(h.literals or ()) * math.log(2) for h in hypotheses])
    prior -= np.logaddexp.reduce(prior)
    table.flags.writeable = prior.flags.writeable = False
    return tuple(hypotheses), table, prior


@register("detector", "page-hinkley", "Predictive log-loss change detector with explicit reset state")
class PageHinkley:
    def __init__(self, minimum=20, tolerance=0.05, threshold=5.0):
        if minimum < 1 or tolerance < 0 or threshold <= 0:
            raise ValueError("invalid change detector configuration")
        self.minimum, self.tolerance, self.threshold = minimum, tolerance, threshold
        self.reset()

    def reset(self):
        self.count, self.mean, self.cumulative, self.lowest = 0, 0.0, 0.0, 0.0

    def update(self, loss):
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("loss must be finite and nonnegative")
        self.count += 1
        self.mean += (loss - self.mean) / self.count
        self.cumulative += loss - self.mean - self.tolerance
        self.lowest = min(self.lowest, self.cumulative)
        return self.count >= self.minimum and self.cumulative - self.lowest > self.threshold

    def snapshot(self):
        return {"config": [self.minimum, self.tolerance, self.threshold],
                **{key: getattr(self, key) for key in ("count", "mean", "cumulative", "lowest")}}

    def restore(self, state):
        if (state["config"] != [self.minimum, self.tolerance, self.threshold] or state["count"] < 0
                or not all(math.isfinite(state[k]) for k in ("mean", "cumulative", "lowest"))):
            raise ValueError("invalid detector checkpoint")
        for key in ("count", "mean", "cumulative", "lowest"):
            setattr(self, key, state[key])


class BayesianLearner:
    def __init__(self, dimensions=6, noise=0.1, mode="full-history", window=32, detector=None):
        if not 0 < noise < 0.5 or window < 1:
            raise ValueError("noise must be in (0, 0.5), window must be positive")
        self.dimensions, self.noise, self.mode, self.window = dimensions, noise, mode, window
        self.hypotheses, self.table, self.prior = hypothesis_space(dimensions)
        self.log_weights = self.prior.copy()
        self.detector = detector
        self.recent = deque()
        self.count = self.resets = 0
        self._cache = None

    def _validate(self, observation):
        if observation.sequence != self.count + 1 or observation.features >= self.table.shape[1]:
            raise ValueError("observation must be in range and next in sequence")

    def _likelihood(self, observation):
        matches = self.table[:, observation.features] == observation.label
        return np.where(matches, math.log1p(-self.noise), math.log(self.noise))

    def weights(self):
        return np.exp(self.log_weights)

    def probabilities(self):
        if self._cache is None:
            # Small matrices: avoid starting a BLAS thread pool on every teaching turn.
            self._cache = np.clip(np.einsum("h,hx->x", self.weights(), self.table, optimize=False), 0, 1)
            self._cache.flags.writeable = False
        return self._cache

    def predict(self, features):
        if not 0 <= features < self.table.shape[1]:
            raise ValueError("feature assignment out of range")
        return Prediction(float(self.probabilities()[features]), self.count)

    def explanation_candidate(self):
        index = int(np.argmax(self.log_weights))
        return self.hypotheses[index], float(math.exp(self.log_weights[index]))

    def observe(self, observation):
        self._validate(observation)
        p = self.predict(observation.features).probability
        q = self.noise + (1 - 2 * self.noise) * p
        loss = -math.log(q if observation.label else 1 - q)
        reset = self.detector is not None and self.detector.update(loss)
        self.recent.append(observation)
        if reset:
            self.recent = deque(list(self.recent)[-8:])
            self.log_weights = self.prior.copy()
            for item in self.recent:
                self.log_weights += self._likelihood(item)
            self.detector.reset()
            self.resets += 1
        else:
            self.log_weights += self._likelihood(observation)
            if self.mode == "window" and len(self.recent) > self.window:
                self.log_weights -= self._likelihood(self.recent.popleft())
        keep = self.window if self.mode == "window" else 8
        while len(self.recent) > keep:
            self.recent.popleft()
        self.log_weights -= np.logaddexp.reduce(self.log_weights)
        self._cache = None
        self.count += 1
        return {"reset": bool(reset), "pre_update_observed_log_loss": loss, "resets": self.resets}

    def snapshot(self):
        return {"schema": 1, "kind": self.mode, "dimensions": self.dimensions, "noise": self.noise,
                "window": self.window, "count": self.count, "resets": self.resets,
                "log_weights": self.log_weights.tolist(), "recent": [asdict(o) for o in self.recent],
                "detector": self.detector.snapshot() if self.detector else None}

    def restore(self, state):
        if (state["schema"], state["kind"], state["dimensions"], state["noise"], state["window"]) != (
                1, self.mode, self.dimensions, self.noise, self.window):
            raise ValueError("checkpoint component configuration mismatch")
        weights = np.asarray(state["log_weights"], dtype=np.float64)
        if weights.shape != self.prior.shape or not np.all(np.isfinite(weights)) or abs(np.logaddexp.reduce(weights)) > 1e-8:
            raise ValueError("invalid posterior checkpoint")
        if (state["detector"] is None) != (self.detector is None) or state["count"] < 0:
            raise ValueError("invalid detector/count checkpoint")
        self.log_weights = weights.copy()
        self.recent = deque(Observation(**o) for o in state["recent"])
        self.count, self.resets = state["count"], state["resets"]
        if self.detector:
            self.detector.restore(state["detector"])
        self._cache = None


@register("learner", "full-history", "Bayesian conjunction learning from all feedback")
def full_history(**kwargs):
    return BayesianLearner(mode="full-history", **kwargs)


@register("learner", "window", "Bayesian conjunction learning from a recent observation window")
def rolling_window(**kwargs):
    return BayesianLearner(mode="window", **kwargs)


@register("learner", "drift-reset", "Bayesian conjunction learning with Page-Hinkley-triggered resets")
def drift_reset(detector_config=None, detector_component="page-hinkley", **kwargs):
    return BayesianLearner(mode="drift-reset", detector=create("detector", detector_component, **(detector_config or {})), **kwargs)


@register("learner", "frozen", "Fixed conjunction prior; feedback has no effect")
class FrozenLearner(BayesianLearner):
    def __init__(self, **kwargs):
        super().__init__(mode="frozen", **kwargs)

    def observe(self, observation):
        self._validate(observation)
        self.count += 1
        return {"reset": False, "resets": 0}


@register("learner", "memory", "Beta(1,1) outcome lookup; no generalization to unseen examples")
class MemoryLearner:
    def __init__(self, dimensions=6, noise=0.1, window=32):
        self.dimensions, self.noise, self.window = dimensions, noise, window
        self.count = 0
        self.positive = np.zeros(1 << dimensions, dtype=np.int64)
        self.total = np.zeros_like(self.positive)

    def probabilities(self):
        return (self.positive + 1) / (self.total + 2)

    def predict(self, features):
        if not 0 <= features < len(self.total):
            raise ValueError("feature assignment out of range")
        return Prediction(float(self.probabilities()[features]), self.count)

    def explanation_candidate(self):
        return None

    def observe(self, observation):
        if observation.sequence != self.count + 1 or observation.features >= len(self.total):
            raise ValueError("observation must be in range and next in sequence")
        self.positive[observation.features] += observation.label
        self.total[observation.features] += 1
        self.count += 1
        return {"reset": False, "resets": 0}

    def snapshot(self):
        return {"schema": 1, "kind": "memory", "dimensions": self.dimensions,
                "count": self.count, "positive": self.positive.tolist(), "total": self.total.tolist()}

    def restore(self, state):
        if (state["schema"], state["kind"], state["dimensions"]) != (1, "memory", self.dimensions):
            raise ValueError("checkpoint component mismatch")
        pos, total = np.asarray(state["positive"]), np.asarray(state["total"])
        if pos.shape != self.positive.shape or total.shape != self.total.shape or np.any(pos < 0) or np.any(pos > total) or total.sum() != state["count"]:
            raise ValueError("invalid memory checkpoint")
        self.positive, self.total, self.count = pos.copy(), total.copy(), state["count"]


def binary_entropy(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class Policy:
    def __init__(self, seed=0, exploration=0.1, kind="random"):
        if not 0 <= exploration <= 1:
            raise ValueError("exploration must be a probability")
        self.rng = np.random.default_rng(seed)
        self.exploration, self.kind = exploration, kind

    def choose(self, probabilities, candidates, noise):
        if not candidates or not 0 < noise < 0.5:
            raise ValueError("invalid question candidates/noise")
        indices = np.array(candidates, dtype=np.int64)
        if (np.any(indices < 0) or np.any(indices >= len(probabilities)) or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0) or np.any(probabilities > 1)):
            raise ValueError("invalid candidate predictions")
        # I(H; noisy Y | x) = H(Y | x) - H(noise), for deterministic hypotheses.
        gains = binary_entropy(noise + (1 - 2 * noise) * probabilities[indices]) - binary_entropy(noise)
        explore = self.kind == "random" or self.rng.random() < self.exploration
        choices = np.arange(len(indices)) if explore else np.flatnonzero(gains >= gains.max() - 1e-12)
        offset = int(self.rng.choice(choices))
        return Question(int(indices[offset]), self.kind,
                        "random exploration" if explore else "maximum expected hypothesis information",
                        float(max(0, gains[offset])))

    def snapshot(self):
        return {"schema": 1, "kind": self.kind, "exploration": self.exploration, "rng": self.rng.bit_generator.state}

    def restore(self, state):
        if (state["schema"], state["kind"], state["exploration"]) != (1, self.kind, self.exploration):
            raise ValueError("policy checkpoint mismatch")
        self.rng.bit_generator.state = state["rng"]


@register("policy", "random", "Uniform random membership questions, with repetition")
def random_policy(**kwargs):
    return Policy(kind="random", **kwargs)


@register("policy", "information-gain", "Expected hypothesis information, plus random exploration")
def informative_policy(**kwargs):
    return Policy(kind="information-gain", **kwargs)


@register("memory", "sqlite", "Append-only teaching events and compressed component checkpoints")
class SQLiteMemory:
    def __init__(self, connection, run_id):
        self.connection, self.run_id = connection, run_id
        connection.execute("CREATE TABLE IF NOT EXISTS events(run_id TEXT, sequence INTEGER, kind TEXT, payload TEXT, PRIMARY KEY(run_id, sequence, kind))")
        connection.execute("CREATE TABLE IF NOT EXISTS checkpoints(run_id TEXT PRIMARY KEY, state BLOB)")

    def append(self, event):
        self.connection.execute("INSERT INTO events VALUES(?,?,?,?)",
                                (self.run_id, event.sequence, event.kind, canonical(event.payload).decode()))

    def events(self):
        return [LearningEvent(seq, kind, json.loads(payload)) for seq, kind, payload in self.connection.execute(
            "SELECT sequence,kind,payload FROM events WHERE run_id=? ORDER BY sequence,kind", (self.run_id,))]

    def checkpoint(self, state):
        self.connection.execute("INSERT OR REPLACE INTO checkpoints VALUES(?,?)", (self.run_id, zlib.compress(canonical(state))))

    def restore(self):
        row = self.connection.execute("SELECT state FROM checkpoints WHERE run_id=?", (self.run_id,)).fetchone()
        if row is None:
            raise ValueError("checkpoint missing")
        return json.loads(zlib.decompress(row[0]))


def horn_program(hypothesis, features, dimensions):
    if not 0 <= features < 1 << dimensions:
        raise ValueError("feature assignment out of range")
    true_id, false_id = 2 * dimensions, 2 * dimensions + 1
    seeds = [2 * index + int(bool(features & (1 << index))) for index in range(dimensions)]
    if hypothesis.literals is None:
        rules = [(false_id, [])]
    else:
        if any(not 1 <= abs(lit) <= dimensions for lit in hypothesis.literals):
            raise ValueError("rule literal out of range")
        premises = [2 * (abs(lit) - 1) + int(lit > 0) for lit in hypothesis.literals]
        rules = [(true_id, premises)] + [(false_id, [premise ^ 1]) for premise in premises]
    return seeds, rules


@register("reasoner", "direct", "Direct evaluation of a candidate conjunction")
class DirectReasoner:
    def explain(self, hypothesis, features, dimensions):
        seeds, rules = horn_program(hypothesis, features, dimensions)
        return {"hypothesis_id": hypothesis.identifier, "conditional_on_hypothesis": True,
                "outcome": hypothesis.evaluate(features), "seeds": seeds, "rules": rules}


@register("reasoner", "rust-horn", "Existing Rust closure engine with explicit signed-feature atoms")
class RustReasoner:
    def explain(self, hypothesis, features, dimensions):
        seeds, rules = horn_program(hypothesis, features, dimensions)
        program = "\n".join([f"s {s}" for s in seeds] + [" ".join(map(str, ["r", head, *body])) for head, body in rules])
        with tempfile.TemporaryDirectory(prefix="learning-horn-") as tmp:
            path = Path(tmp) / "program.txt"
            path.write_text(program + "\n")
            output = json.loads(subprocess.check_output([str(ROOT / "target/release/bench"), "infer", str(path)], text=True))
        outcome = 2 * dimensions in output["fact_ids"]
        if outcome != hypothesis.evaluate(features) or (2 * dimensions + 1 in output["fact_ids"]) == outcome:
            raise AssertionError("Rust candidate-rule evaluation disagrees")
        return {"hypothesis_id": hypothesis.identifier, "conditional_on_hypothesis": True,
                "outcome": outcome, "program": program, "proof": output}
