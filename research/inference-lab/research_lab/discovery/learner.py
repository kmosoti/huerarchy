from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict
import math

import numpy as np

from ..common import digest
from ..learning.components import PageHinkley, binary_entropy
from ..learning.types import create, register
from . import expansion
from .expansion import ExpansionView, initial_programs
from .programs import BatchEvaluator, BudgetExhausted, Feedback, Program, Question


@register("discovery-policy", "questions", "Random, information-seeking, or prequential learning-progress questions")
class QuestionPolicy:
    def __init__(self, kind, seed, exploration=0.1):
        if kind not in {"random", "information", "progress"}:
            raise ValueError("unknown discovery question policy")
        self.kind, self.exploration = kind, exploration
        self.rng = np.random.default_rng(seed)
        self.losses = defaultdict(lambda: deque(maxlen=64))

    def choose(self, probabilities, scenes, noise):
        random = self.kind == "random" or self.rng.random() < self.exploration
        if random:
            return Question(int(self.rng.integers(len(scenes))), self.kind, "random exploration")
        q = noise + (1 - 2 * noise) * probabilities
        values = binary_entropy(q) - binary_entropy(noise)
        indices = np.arange(len(scenes))
        reason = "maximum expected hypothesis information"
        if self.kind == "progress":
            groups = sorted({scene.group for scene in scenes})
            progress = {}
            for group in groups:
                history = list(self.losses[group])
                progress[group] = 1.0 if len(history) < 32 else max(0, np.mean(history[-32:-16]) - np.mean(history[-16:]))
            best = max(progress.values())
            eligible = [g for g in groups if progress[g] >= best - 1e-12]
            group = str(self.rng.choice(eligible))
            indices = np.array([i for i, scene in enumerate(scenes) if scene.group == group])
            reason = "prequential prediction improvement in public question group"
        candidates = indices[values[indices] >= values[indices].max() - 1e-12]
        return Question(int(self.rng.choice(candidates)), self.kind, reason)

    def observe(self, group, loss):
        self.losses[group].append(float(loss))

    def snapshot(self):
        return {"schema": 1, "kind": self.kind, "exploration": self.exploration,
                "rng": self.rng.bit_generator.state, "losses": {k: list(v) for k, v in self.losses.items()}}

    def restore(self, state):
        if (state["schema"], state["kind"], state["exploration"]) != (1, self.kind, self.exploration):
            raise ValueError("policy checkpoint configuration mismatch")
        self.rng.bit_generator.state = state["rng"]
        self.losses = defaultdict(lambda: deque(maxlen=64), {k: deque(v, maxlen=64) for k, v in state["losses"].items()})


def log_loss(probability, label):
    return -math.log(max(1e-15, probability if label else 1 - probability))


class DiscoveryLearner:
    def __init__(self, public, config, variant, library=None, initialize_broad=True):
        self.public, self.config, self.variant = public, config, variant
        self.library = dict(library or {})
        self.noise = config["assumed_noise"]
        self.base = initial_programs(public.schema, len(public.questions[0].query))
        self.programs = list(self.base)
        self.extra = {}
        self.engine = BatchEvaluator(public.questions, public.schema, self.library, config["search_context_budget"])
        self.evaluation_engine = BatchEvaluator(public.evaluation, public.schema, self.library)
        self.history, self.active_history = deque(maxlen=config["fit_window"]), deque()
        self.count = self.resets = self.promotions = self.proposals = 0
        self.uncertain = False
        self.pending = None
        self.detector = PageHinkley()
        self.expander = create("discovery-expander", config["components"]["expander"])
        self.budget_exhausted = False
        self._matrices = {}
        self._rebuild()
        if variant == "broad-upfront" and initialize_broad:
            self._refresh_broad()

    def _matrix(self, evaluation=False):
        key = "evaluation" if evaluation else "questions"
        if key not in self._matrices:
            engine = self.evaluation_engine if evaluation else self.engine
            self._matrices[key] = np.array([engine.evaluate(p) for p in self.programs], dtype=np.float64)
        return self._matrices[key]

    def _costs(self):
        return np.array([self.config["complexity_penalty"] * p.nodes for p in self.programs])

    def _rebuild(self, history=None):
        self._matrices = {}
        self.log_weights = -self._costs()
        for observation in self.active_history if history is None else history:
            matches = self._matrix()[:, observation.question_index] == observation.label
            self.log_weights += np.where(matches, math.log1p(-self.noise), math.log(self.noise))
        self.log_weights -= np.logaddexp.reduce(self.log_weights)

    def probabilities(self, evaluation=False):
        if self.uncertain:
            count = len(self.public.evaluation if evaluation else self.public.questions)
            return np.full(count, 0.5)
        return np.clip(np.einsum("h,hx->x", np.exp(self.log_weights), self._matrix(evaluation), optimize=False), 0, 1)

    def best(self):
        index = int(np.argmax(self.log_weights))
        return self.programs[index], float(math.exp(self.log_weights[index]))

    def _refit(self, history):
        self.active_history = deque(history)
        self._rebuild()

    def _propose(self):
        order = np.argsort(-self.log_weights, kind="stable")
        # All base clauses remain available; parents carry no clean outcomes or target tags.
        parents = tuple(self.programs[i] for i in order[:24]) + self.base
        view = ExpansionView(self.public.schema, len(self.public.questions[0].query), tuple(self.history),
                             parents, self.engine, self.library, self.noise, self.config["complexity_penalty"])
        proposed = self.expander.propose(view, self.config["proposal_limit"])
        self.proposals += len(proposed)
        self.budget_exhausted |= self.engine.context_evaluations + len(self.public.questions) > self.engine.budget
        return proposed

    def _merge(self, incoming):
        for program in incoming:
            if program.identifier not in {p.identifier for p in self.base}:
                self.extra[program.identifier] = program
        def cost(program):
            pred = self.engine.evaluate(program)
            mismatch = sum(bool(pred[o.question_index]) != o.label for o in self.history)
            return (-mismatch * math.log(self.noise) - (len(self.history) - mismatch) * math.log1p(-self.noise)
                    + self.config["complexity_penalty"] * program.nodes, program.identifier)
        selected = sorted(self.extra.values(), key=cost)[:self.config["expanded_limit"]]
        self.extra = {p.identifier: p for p in selected}
        self.programs = list(self.base) + selected
        self._refit(list(self.history))

    def _refresh_broad(self):
        proposed = self._propose()
        try:
            self._merge([p.program for p in proposed])
        except BudgetExhausted:
            self.budget_exhausted = True

    def _launch(self):
        proposed = self._propose()
        current_ids = {p.identifier for p in self.programs}
        growth = next((p for p in proposed if p.program.identifier not in current_ids), None)
        # Freeze challenger structures and reset-model probabilities for the fresh confirmation block.
        reset_log = -self._costs()
        for observation in list(self.history)[-32:]:
            matches = self._matrix()[:, observation.question_index] == observation.label
            reset_log += np.where(matches, math.log1p(-self.noise), math.log(self.noise))
        reset_log -= np.logaddexp.reduce(reset_log)
        reset_probability = np.einsum("h,hx->x", np.exp(reset_log), self._matrix(), optimize=False)
        best, _ = self.best()
        self.pending = {"started_after": self.count, "seen": 0, "incumbent_loss": 0.0,
                        "reset_loss": 0.0, "uncertain_loss": 0.0, "growth_loss": 0.0,
                        "reset_probability": reset_probability.tolist(), "incumbent_nodes": best.nodes,
                        "growth": {"program": growth.program.to_dict(), "strategy": growth.strategy,
                                   "evidence_sequences": list(growth.evidence_sequences), "parent_ids": list(growth.parent_ids)} if growth else None}
        return {"event": "challenge-started", "growth": self.pending["growth"],
                "fresh_responses_required": self.config["confirmation"], "proposals_returned": len(proposed)}

    def observe(self, feedback):
        if feedback.sequence != self.count + 1 or not 0 <= feedback.question_index < len(self.public.questions):
            raise ValueError("out-of-order feedback or question outside public pool")
        before = self.probabilities()
        q = self.noise + (1 - 2 * self.noise) * float(before[feedback.question_index])
        loss = log_loss(q, feedback.label)
        alarm = self.detector.update(loss)
        events = []
        if self.pending:
            pending = self.pending
            pending["seen"] += 1
            pending["incumbent_loss"] += loss
            rp = pending["reset_probability"][feedback.question_index]
            pending["reset_loss"] += log_loss(self.noise + (1 - 2 * self.noise) * rp, feedback.label)
            pending["uncertain_loss"] += math.log(2)
            if pending["growth"]:
                program = Program.from_dict(pending["growth"]["program"])
                prediction = self.engine.evaluate(program)[feedback.question_index]
                pending["growth_loss"] += log_loss(1 - self.noise if prediction else self.noise, feedback.label)
        self.count += 1
        self.history.append(feedback)
        self.active_history.append(feedback)
        matches = self._matrix()[:, feedback.question_index] == feedback.label
        self.log_weights += np.where(matches, math.log1p(-self.noise), math.log(self.noise))
        if len(self.active_history) > self.config["fit_window"]:
            old = self.active_history.popleft()
            old_matches = self._matrix()[:, old.question_index] == old.label
            self.log_weights -= np.where(old_matches, math.log1p(-self.noise), math.log(self.noise))
        self.log_weights -= np.logaddexp.reduce(self.log_weights)
        if self.variant == "small-reset" and alarm:
            self._refit(list(self.history)[-8:])
            self.resets += 1
            self.detector.reset()
            events.append({"event": "immediate-reset", "sequence": self.count})
        if self.pending and self.pending["seen"] == self.config["confirmation"]:
            pending = self.pending
            candidates = [("reset", pending["reset_loss"], pending["incumbent_nodes"]),
                          ("uncertain", pending["uncertain_loss"], 0)]
            if pending["growth"]:
                program = Program.from_dict(pending["growth"]["program"])
                candidates.append(("growth", pending["growth_loss"], program.nodes))
            scored = [(pending["incumbent_loss"] - candidate_loss
                       - self.config["complexity_penalty"] * (nodes - pending["incumbent_nodes"]),
                       kind, pending["incumbent_loss"] - candidate_loss) for kind, candidate_loss, nodes in candidates]
            gain, winner, raw_gain = max(scored)
            accepted = gain > self.config["promotion_margin"] and raw_gain > 0
            event = {"event": "challenge-completed", "sequence": self.count, "started_after": pending["started_after"],
                     "candidate": winner, "accepted": accepted, "penalized_gain": gain, "prediction_gain": raw_gain,
                     "fresh_responses": pending["seen"], "growth": pending["growth"]}
            if accepted:
                self.uncertain = winner == "uncertain"
                if winner == "growth":
                    self._merge([Program.from_dict(pending["growth"]["program"])])
                    self.promotions += 1
                elif winner == "reset":
                    self._refit(list(self.history)[-32:])
                    self.resets += 1
                self.detector.reset()
            events.append(event)
            self.pending = None
        if self.variant == "broad-upfront":
            if self.count % self.config["proposal_interval"] == 0:
                self._refresh_broad()
        elif self.variant != "small-reset" and self.pending is None and self.count >= self.config["minimum_observations"]:
            if self.count % self.config["proposal_interval"] == 0 or alarm:
                events.append(self._launch())
        return {"pre_update_probability": float(before[feedback.question_index]), "pre_update_log_loss": loss,
                "alarm": bool(alarm), "events": events, "uncertain": self.uncertain}

    def snapshot(self):
        return {"schema": 1, "config_sha256": digest(self.config), "variant": self.variant,
                "programs": [p.to_dict() for p in self.programs], "base_ids": [p.identifier for p in self.base],
                "log_weights": self.log_weights.tolist(), "history": [asdict(o) for o in self.history],
                "active_history": [asdict(o) for o in self.active_history], "count": self.count,
                "resets": self.resets, "promotions": self.promotions, "proposals": self.proposals,
                "uncertain": self.uncertain, "pending": self.pending, "detector": self.detector.snapshot(),
                "budget_exhausted": self.budget_exhausted, "context_evaluations": self.engine.context_evaluations,
                "public_sha256": digest(self.public.to_dict()),
                "library_sha256": digest({name: Program.make((clause,)).to_dict() for name, clause in self.library.items()}),
                "prediction_cache": {k: v.tolist() for k, v in self.engine.cache.items()}}

    def restore(self, state):
        if (state["schema"], state["config_sha256"], state["variant"], state["public_sha256"]) != (
                1, digest(self.config), self.variant, digest(self.public.to_dict())):
            raise ValueError("learner checkpoint configuration/public task mismatch")
        if state["library_sha256"] != digest({name: Program.make((clause,)).to_dict() for name, clause in self.library.items()}):
            raise ValueError("learner checkpoint library mismatch")
        self.programs = [Program.from_dict(p) for p in state["programs"]]
        base_ids = set(state["base_ids"])
        self.extra = {p.identifier: p for p in self.programs if p.identifier not in base_ids}
        weights = np.asarray(state["log_weights"])
        if len(weights) != len(self.programs) or not np.all(np.isfinite(weights)) or abs(np.logaddexp.reduce(weights)) > 1e-8:
            raise ValueError("invalid posterior checkpoint")
        self.log_weights = weights.copy()
        self.history = deque((Feedback(**o) for o in state["history"]), maxlen=self.config["fit_window"])
        self.active_history = deque(Feedback(**o) for o in state["active_history"])
        for key in ("count", "resets", "promotions", "proposals", "uncertain", "pending", "budget_exhausted"):
            setattr(self, key, state[key])
        self.detector.restore(state["detector"])
        self.engine.context_evaluations = state["context_evaluations"]
        self.engine.cache = {k: np.asarray(v, dtype=bool) for k, v in state["prediction_cache"].items()}
        if any(v.shape != (len(self.public.questions),) for v in self.engine.cache.values()):
            raise ValueError("invalid prediction cache")
        self._matrices = {}
