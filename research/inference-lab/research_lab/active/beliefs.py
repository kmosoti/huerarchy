from __future__ import annotations

from collections import deque
import math
import numpy as np

from ..learning.types import register


def normalize(values):
    # Stable log-sum-exp without a serial logaddexp reduction over every state.
    maximum = np.max(values)
    return values - (maximum + np.log(np.exp(values - maximum).sum()))


@register("active-posterior", "finite", "Finite noisy-program posterior with window or capped change-point branches")
class FinitePosterior:
    def __init__(self, prior, mode="window", window=64, hazard=.01, branches=32):
        if mode not in {"window", "change"} or not 0 <= hazard < 1 or branches < 2:
            raise ValueError("invalid posterior mode")
        self.prior = normalize(np.asarray(prior, dtype=float))
        self.mode, self.window, self.hazard, self.branches = mode, window, hazard, branches
        self.log_joint = self.prior[None, :].copy()
        self.runs = np.array([0], dtype=int)
        self.recent = deque()
        self.count = 0
        self.pruned_mass = 0.0

    def observe(self, log_likelihood):
        ll = np.asarray(log_likelihood)
        if ll.shape != self.prior.shape or not np.all(np.isfinite(ll)):
            raise ValueError("invalid likelihood")
        if self.mode == "window":
            self.recent.append(ll.copy())
            if len(self.recent) > self.window:
                self.recent.popleft()
            self.log_joint = normalize(self.prior + np.sum(self.recent, axis=0))[None, :]
            self.runs = np.array([min(self.count + 1, self.window)])
        elif not self.count:
            self.log_joint = normalize(self.prior + ll)[None, :]
            self.runs = np.array([1])
        elif self.hazard == 0:
            self.log_joint = normalize(self.log_joint + ll)
            self.runs += 1
        else:
            grow = self.log_joint + math.log1p(-self.hazard) + ll
            restart = self.prior + math.log(self.hazard) + ll
            joint = normalize(np.vstack((restart, grow)))
            runs = np.r_[1, self.runs + 1]
            if len(runs) > self.branches:
                mass = np.exp(joint).sum(axis=1)
                keep = np.r_[0, 1 + np.argsort(-mass[1:], kind="stable")[:self.branches - 1]]
                self.pruned_mass += max(0.0, 1.0 - float(mass[keep].sum()))
                joint, runs = normalize(joint[keep]), runs[keep]
            self.log_joint, self.runs = joint, runs
        self.count += 1

    def weights(self):
        return np.exp(self.log_joint).sum(axis=0)

    def next_weights(self):
        weights = self.weights()
        if self.mode == "change" and self.count:
            return (1 - self.hazard) * weights + self.hazard * np.exp(self.prior)
        return weights

    def run_weights(self):
        return np.exp(self.log_joint).sum(axis=1)

    def snapshot(self):
        return {"mode": self.mode, "window": self.window, "hazard": self.hazard, "branches": self.branches,
                "prior": self.prior.tolist(), "joint": self.log_joint.tolist(), "runs": self.runs.tolist(),
                "count": self.count, "recent": [x.tolist() for x in self.recent], "pruned_mass": self.pruned_mass}

    @classmethod
    def from_state(cls, state):
        obj = cls(state["prior"], state["mode"], state["window"], state["hazard"], state["branches"])
        # Re-normalizing a saved prior changes subsequent floating-point rounding.
        obj.prior = np.array(state["prior"], dtype=float)
        obj.log_joint = np.array(state["joint"])
        obj.runs = np.array(state["runs"], dtype=int)
        obj.count, obj.pruned_mass = state["count"], state["pruned_mass"]
        obj.recent = deque(np.array(x) for x in state["recent"])
        if (obj.log_joint.shape != (len(obj.runs), len(obj.prior)) or
                not np.all(np.isfinite(obj.log_joint)) or abs(np.exp(obj.log_joint).sum() - 1) > 1e-8):
            raise ValueError("invalid posterior checkpoint")
        return obj
