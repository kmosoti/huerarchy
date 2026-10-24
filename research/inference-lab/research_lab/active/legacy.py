"""Adapters preserve the historical predictor and isolate confirmation stopping rules."""
from dataclasses import asdict
import math
import numpy as np

from ..discovery.learner import DiscoveryLearner, QuestionPolicy, log_loss
from ..discovery.programs import Program
from ..discovery.worlds import PublicTask, configuration


class SequentialLegacy(DiscoveryLearner):
    def observe(self, feedback):
        # Compute exactly the historical comparison on the next fresh answer.
        # Only its stopping time changes; likelihood, complexity, and margin are identical.
        self.config["confirmation"] = 64
        if self.pending:
            p = self.pending
            q = self.noise + (1 - 2 * self.noise) * self.probabilities()[feedback.question_index]
            incumbent = p["incumbent_loss"] + log_loss(q, feedback.label)
            rp = p["reset_probability"][feedback.question_index]
            candidates = [(p["reset_loss"] + log_loss(self.noise + (1 - 2 * self.noise) * rp, feedback.label), p["incumbent_nodes"]),
                          (p["uncertain_loss"] + math.log(2), 0)]
            if p["growth"]:
                program = Program.from_dict(p["growth"]["program"])
                prediction = self.engine.evaluate(program)[feedback.question_index]
                candidates.append((p["growth_loss"] + log_loss(1 - self.noise if prediction else self.noise, feedback.label), program.nodes))
            if any(incumbent - loss > 0 and incumbent - loss - self.config["complexity_penalty"] * (nodes - p["incumbent_nodes"]) > self.config["promotion_margin"]
                   for loss, nodes in candidates):
                self.config["confirmation"] = p["seen"] + 1
        return super().observe(feedback)


class LegacyAdapter:
    def __init__(self, view, config, kind, seed, initialize=True, stored_config=None):
        self.view, self.kind = view, kind
        c = stored_config or configuration({"workers": 1, "proposal_limit": config["proposal_limit"],
            "expanded_limit": config["expanded_limit"], "search_context_budget": config["search_context_budget"],
            "proposal_interval": config["proposal_interval"], "fit_window": 32 if kind == "broad-window32" else 64,
            "confirmation": 8 if kind == "gate8" else 64 if kind == "gate-sequential" else 32})
        variant = "broad-upfront" if kind.startswith("broad") else "adaptive"
        cls = SequentialLegacy if kind == "gate-sequential" else DiscoveryLearner
        # Historical class requires an evaluation pool; this adapter supplies only teaching inputs.
        self.inner = cls(PublicTask(view.schema, view.questions, view.questions), c, variant, initialize_broad=initialize)
        self.policy = QuestionPolicy("information", seed, config["exploration"])

    def choose(self):
        q = self.policy.choose(self.inner.probabilities(), self.view.questions, self.inner.noise)
        return q.index, asdict(q)

    def observe(self, observation):
        r = self.inner.observe(observation)
        self.policy.observe(self.view.questions[observation.question_index].group, r["pre_update_log_loss"])
        p = r["pre_update_probability"]
        return {"pre_probability": p, "pre_observed_probability": self.inner.noise + (1 - 2 * self.inner.noise) * p,
                "pre_log_loss": r["pre_update_log_loss"], "events": r["events"]}

    def prediction_state(self):
        return (tuple(self.inner.programs), np.zeros(len(self.inner.programs)) if self.inner.uncertain else np.exp(self.inner.log_weights),
                float(self.inner.uncertain), {})

    def snapshot(self):
        return {"kind": self.kind, "config": self.inner.config, "learner": self.inner.snapshot(), "policy": self.policy.snapshot()}

    def restore(self, state):
        self.inner.restore(state["learner"])
        self.policy.restore(state["policy"])

    @property
    def engine(self):
        return self.inner.engine

    @property
    def programs(self):
        return self.inner.programs

    @property
    def budget_exhausted(self):
        return self.inner.budget_exhausted

    def best(self):
        return self.inner.best()
