from __future__ import annotations

import numpy as np

from ..learning.types import register


def entropy(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def information_scores(conditional, weights, kind="eig", reference=None):
    """Exact finite-mixture EIG/EPIG. Conditional draws are independent given the model."""
    p, w = np.asarray(conditional), np.asarray(weights)
    marginal = w @ p
    if kind == "eig":
        return np.maximum(0, entropy(marginal) - w @ entropy(p))
    if kind != "epig":
        raise ValueError("unknown information criterion")
    r = p if reference is None else np.asarray(reference)
    mr = w @ r
    joint = (p.T * w) @ r
    cells = (joint, marginal[:, None] - joint, mr[None, :] - joint,
             1 - marginal[:, None] - mr[None, :] + joint)
    independent = (marginal[:, None] * mr[None, :], marginal[:, None] * (1 - mr[None, :]),
                   (1 - marginal[:, None]) * mr[None, :], (1 - marginal[:, None]) * (1 - mr[None, :]))
    score = sum(np.maximum(c, 0) * np.log(np.maximum(c, 1e-15) / np.maximum(d, 1e-15))
                for c, d in zip(cells, independent))
    return np.maximum(0, score.mean(axis=1))


@register("active-policy", "information", "Random, EIG, and predictive-information questions over public inputs")
class InformationPolicy:
    def __init__(self, kind="eig", seed=0, exploration=.1):
        if kind not in {"random", "eig", "epig"}:
            raise ValueError("unknown policy")
        self.kind, self.exploration = kind, exploration
        self.rng = np.random.default_rng(seed)

    def choose(self, conditional, weights):
        if self.kind == "random" or self.rng.random() < self.exploration:
            return int(self.rng.integers(conditional.shape[1])), {"policy": self.kind, "exploration": True, "score_nats": None}
        scores = information_scores(conditional, weights, self.kind)
        choices = np.flatnonzero(scores >= scores.max() - 1e-12)
        index = int(self.rng.choice(choices))
        return index, {"policy": self.kind, "exploration": False, "score_nats": float(scores[index])}

    def snapshot(self):
        return {"kind": self.kind, "exploration": self.exploration, "rng": self.rng.bit_generator.state}

    def restore(self, state):
        if (state["kind"], state["exploration"]) != (self.kind, self.exploration):
            raise ValueError("policy configuration mismatch")
        self.rng.bit_generator.state = state["rng"]
