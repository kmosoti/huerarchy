from __future__ import annotations

from dataclasses import asdict
import itertools
import math
import time
import numpy as np

from ..common import digest
from ..discovery.expansion import CompositionalExpander, ExpansionView, initial_programs, ordinary_atoms, program_for, fit_cost
from ..discovery.programs import Atom, Clause, BatchEvaluator, BudgetExhausted, Feedback, Program, ExpansionProposal, validate
from ..learning.types import register, create
from . import beliefs, policies


def expanded_program(program, library):
    """Hygienic expansion of extensional helpers; never share existential variables accidentally."""
    clauses = []
    for clause in program.clauses:
        used = {v for atom in (clause.head, *clause.body) for v in atom.variables}
        body = []
        for atom in clause.body:
            if atom.predicate not in library:
                body.append(atom)
                continue
            definition = library[atom.predicate]
            mapping = dict(zip(definition.head.variables, atom.variables))
            for variable in sorted({v for a in definition.body for v in a.variables} - set(mapping)):
                fresh = max(used, default=-1) + 1
                used.add(fresh); mapping[variable] = fresh
            body.extend(Atom(a.predicate, tuple(mapping[v] for v in a.variables)) for a in definition.body)
        clauses.append(Clause(clause.head, tuple(body)))
    return Program.make(clauses)


@register("active-expander", "typed", "Existing positive grammar plus equally available five-atom transfer compositions")
class TypedExpander:
    def propose(self, view, budget, transfer=False):
        ordinary = CompositionalExpander().propose(view, budget if not transfer else max(1, budget * 3 // 4))
        if not transfer or view.arity != 2:
            return ordinary
        extra = []
        atoms = ordinary_atoms(view.schema, view.arity)
        fragments = [(Atom(n, (0, 2)), Atom(n, (1, 2))) for n, t in sorted(view.schema.items()) if t == ("object", "value")]
        seen = {p.program.identifier for p in ordinary}
        try:
            for additions in itertools.combinations(atoms, 3):
                for fragment in fragments:
                    p = program_for(fragment + additions, view.arity)
                    if p.identifier in seen:
                        continue
                    try:
                        validate(p, view.schema, view.arity)
                    except ValueError:
                        continue
                    seen.add(p.identifier)
                    extra.append((fit_cost(p, view), ExpansionProposal(p, "typed-combination", tuple(o.sequence for o in view.observations), ())))
                    if len(ordinary) + len(extra) >= budget:
                        return ordinary + [v for _, v in sorted(extra, key=lambda x: (x[0], x[1].program.identifier))]
        except BudgetExhausted:
            pass
        return ordinary + [v for _, v in sorted(extra, key=lambda x: (x[0], x[1].program.identifier))]


class ActiveLearner:
    def __init__(self, view, config, mode="window", library=None, archive=False, transfer=False, initialize=True):
        self.view, self.config, self.mode = view, config, mode
        self.library, self.archive_enabled, self.transfer = dict(library or {}), archive, transfer
        self.engine = BatchEvaluator(view.questions, view.schema, self.library, config["search_context_budget"])
        self.base = initial_programs(view.schema, len(view.questions[0].query))
        self.programs = list(self.base)
        self.history = []
        self.archive, self.streak_id, self.streak = {}, None, 0
        self.proposals = self.rebuilds = 0
        self.budget_exhausted = False
        self.expander = create("active-expander", config["components"]["expander"])
        self._rebuild()
        if initialize:
            self.refresh()

    def _rebuild(self):
        self.matrix = np.array([self.engine.evaluate(p) for p in self.programs], dtype=float)
        grid = np.array(self.config["noise_grid"])
        self.conditional = np.vstack(((grid[None, :, None] + (1 - 2 * grid[None, :, None]) * self.matrix[:, None, :]).reshape(-1, self.matrix.shape[1]),
                                      np.full((1, self.matrix.shape[1]), .5)))
        prior = -self.config["complexity_penalty"] * np.array([p.nodes for p in self.programs], dtype=float)
        prior -= np.logaddexp.reduce(prior)
        prior = np.r_[np.repeat(prior + math.log1p(-self.config["coin_prior"]) - math.log(len(grid)), len(grid)),
                      math.log(self.config["coin_prior"])]
        self.posterior = create("active-posterior", self.config["components"]["posterior"], prior=prior, mode=self.mode,
                                window=self.config["window"], hazard=self.config["hazard"], branches=self.config["run_branches"])
        # Replay changes the searched approximation, never the already-recorded predictions.
        for o in self.history:
            p = self.conditional[:, o.question_index]
            self.posterior.observe(np.log(p if o.label else 1 - p))
        self.rebuilds += 1

    def question_weights(self):
        return self.posterior.next_weights()

    def program_weights(self, next_step=False):
        w = self.question_weights() if next_step else self.posterior.weights()
        return w[:-1].reshape(len(self.programs), -1).sum(axis=1), float(w[-1])

    def probabilities(self, next_step=False):
        w, coin = self.program_weights(next_step)
        return np.clip(w @ self.matrix + .5 * coin, 0, 1)

    def best(self):
        w, _ = self.program_weights()
        index = int(np.argmax(w))
        return self.programs[index], float(w[index])

    def refresh(self):
        weights, _ = self.program_weights()
        order = np.argsort(-weights, kind="stable")
        parents = tuple(self.programs[i] for i in order[:24]) + self.base
        view = ExpansionView(self.view.schema, len(self.view.questions[0].query), tuple(self.history[-64:]), parents,
                             self.engine, self.library, .1, self.config["complexity_penalty"])
        proposed = self.expander.propose(view, self.config["proposal_limit"], transfer=self.transfer)
        self.proposals += len(proposed)
        if self.library:
            within_bounds = []
            for proposal in proposed:
                try:
                    validate(expanded_program(proposal.program, self.library), self.view.schema, view.arity)
                except ValueError:
                    continue
                within_bounds.append(proposal)
            proposed = within_bounds
        base_ids = {p.identifier for p in self.base}
        extra = {p.identifier: p for p in self.programs if p.identifier not in base_ids}
        extra.update({p.program.identifier: p.program for p in proposed if p.program.identifier not in base_ids})
        extra.update(self.archive)
        try:
            ranked = sorted(extra.values(), key=lambda p: (fit_cost(p, view), p.identifier))
            protected = [p for p in self.archive.values() if p.identifier not in base_ids] if self.archive_enabled else []
            chosen = protected + [p for p in ranked if p.identifier not in {x.identifier for x in protected}][:self.config["expanded_limit"] - len(protected)]
            programs = list(self.base) + sorted(chosen, key=lambda p: p.identifier)
            if [p.identifier for p in programs] != [p.identifier for p in self.programs]:
                self.programs = programs
                self._rebuild()
        except BudgetExhausted:
            self.budget_exhausted = True
        self.budget_exhausted |= self.engine.context_evaluations + len(self.view.questions) > self.engine.budget

    def observe(self, observation):
        if observation.sequence != len(self.history) + 1 or not 0 <= observation.question_index < len(self.view.questions):
            raise ValueError("out-of-order observation")
        before = self.probabilities(next_step=True)[observation.question_index]
        p = self.conditional[:, observation.question_index]
        noisy = float(self.question_weights() @ p)
        loss = -math.log(max(1e-15, noisy if observation.label else 1 - noisy))
        self.history.append(observation)
        self.posterior.observe(np.log(p if observation.label else 1 - p))
        events = []
        best, mass = self.best()
        self.streak = self.streak + 1 if best.identifier == self.streak_id and mass >= self.config["archive_mass"] else int(mass >= self.config["archive_mass"])
        self.streak_id = best.identifier
        if self.archive_enabled and self.streak >= self.config["archive_streak"] and best.identifier not in self.archive:
            self.archive[best.identifier] = best
            if len(self.archive) > self.config["archive_limit"]:
                del self.archive[next(iter(self.archive))]
            events.append({"event": "archive-supported", "program": best.identifier, "mass": mass})
        if len(self.history) % self.config["proposal_interval"] == 0 and not self.budget_exhausted:
            start = time.process_time()
            self.refresh()
            events.append({"event": "candidate-refresh", "cpu_seconds": time.process_time() - start, "programs": len(self.programs)})
        return {"pre_probability": float(before), "pre_observed_probability": noisy, "pre_log_loss": loss, "events": events}

    def snapshot(self):
        return {"schema": 1, "config": digest(self.config), "view": self.view.fingerprint, "mode": self.mode,
                "library": digest({k: Program.make((v,)).to_dict() for k, v in self.library.items()}),
                "archive_enabled": self.archive_enabled, "transfer": self.transfer,
                "programs": [p.to_dict() for p in self.programs], "history": [asdict(o) for o in self.history],
                "posterior": self.posterior.snapshot(), "archive": [p.to_dict() for p in self.archive.values()],
                "streak_id": self.streak_id, "streak": self.streak, "proposals": self.proposals, "rebuilds": self.rebuilds,
                "budget_exhausted": self.budget_exhausted, "context_evaluations": self.engine.context_evaluations,
                "prediction_cache": {k: v.tolist() for k, v in self.engine.cache.items()}}

    def restore(self, state):
        current = self.snapshot()
        for key in ("schema", "config", "view", "mode", "library", "archive_enabled", "transfer"):
            if current[key] != state[key]:
                raise ValueError(f"learner checkpoint mismatch: {key}")
        self.programs = [Program.from_dict(p) for p in state["programs"]]
        self.history = [Feedback(**o) for o in state["history"]]
        self.engine.cache = {k: np.array(v, dtype=bool) for k, v in state["prediction_cache"].items()}
        self.engine.context_evaluations = state["context_evaluations"]
        self._rebuild()
        self.posterior = beliefs.FinitePosterior.from_state(state["posterior"])
        self.archive = {p.identifier: p for p in map(Program.from_dict, state["archive"])}
        for key in ("streak_id", "streak", "proposals", "rebuilds", "budget_exhausted"):
            setattr(self, key, state[key])

    def prediction_state(self):
        weights, coin = self.program_weights()
        return tuple(self.programs), weights.copy(), coin, dict(self.library)


def evaluate_state(state, engine):
    programs, weights, coin, _ = state
    return np.clip(sum(w * engine.evaluate(p) for p, w in zip(programs, weights)) + .5 * coin, 0, 1)
