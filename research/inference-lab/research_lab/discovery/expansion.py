from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, lru_cache
import itertools
import math

import numpy as np

from ..learning.types import register
from .programs import Atom, BudgetExhausted, Clause, ExpansionProposal, Program, target_atom, validate


def consistent(body):
    names = {(a.predicate, a.variables) for a in body}
    for name, variables in names:
        if name.startswith("b") and name[-2:] in ("_0", "_1") and (name[:-1] + str(1 - int(name[-1])), variables) in names:
            return False
        if name in ("u", "v") and ("n" + name, variables) in names:
            return False
    return True


def ordinary_atoms(schema, arity, library=None):
    atoms = []
    for name, sorts in sorted(schema.items()):
        if not sorts:
            atoms.append(Atom(name))
        elif sorts == ("object",):
            atoms.extend(Atom(name, (v,)) for v in range(arity))
        elif sorts == ("object", "object"):
            atoms.extend(Atom(name, pair) for pair in ((0, 1), (1, 0)))
    atoms.extend(Atom(name, pair) for name in sorted(library or {}) for pair in ((0, 1), (1, 0)))
    return tuple(atoms)


def imports_for(clauses, library):
    return tuple(sorted({a.predicate for c in clauses for a in c.body if a.predicate in library}))


def program_for(body, arity, library=None):
    clause = Clause(target_atom(arity), tuple(sorted(set(body))))
    return _program_for(clause, imports_for((clause,), library or {}))


@lru_cache(maxsize=16384)
def _program_for(clause, imports):
    return Program.make((clause,), imports)


def initial_programs(schema, arity):
    return _initial_programs(tuple(sorted(schema.items())), arity)


@lru_cache(maxsize=16)
def _initial_programs(schema_items, arity):
    schema = dict(schema_items)
    result = {Program().identifier: Program()}
    head = target_atom(arity)
    if arity == 0:
        # Exact original conjunction language, including constant outcomes.
        for signs in itertools.product((-1, 0, 1), repeat=6):
            body = tuple(Atom(f"b{i}_{int(sign > 0)}") for i, sign in enumerate(signs) if sign)
            program = Program.make((Clause(head, body),))
            result[program.identifier] = program
    else:
        atoms = ordinary_atoms(schema, arity)
        for size in range(1, 4):
            for body in itertools.combinations(atoms, size):
                if not consistent(body):
                    continue
                program = program_for(body, arity)
                try:
                    validate(program, schema, arity)
                except ValueError:
                    continue
                result[program.identifier] = program
    return tuple(result.values())


@dataclass(frozen=True)
class ExpansionView:
    # Only public schema, observed answers, and candidate explanations cross this boundary.
    schema: dict
    arity: int
    observations: tuple
    parents: tuple[Program, ...]
    evaluator: object
    library: dict
    noise: float
    complexity_penalty: float

    @cached_property
    def indices(self):
        return np.array([o.question_index for o in self.observations], dtype=np.int64)

    @cached_property
    def labels(self):
        return np.array([o.label for o in self.observations], dtype=bool)


def fit_cost(program, view):
    if not view.observations:
        return view.complexity_penalty * program.nodes
    predictions = view.evaluator.evaluate(program)
    mismatch = int(np.count_nonzero(predictions[view.indices] != view.labels))
    return (-mismatch * math.log(view.noise) - (len(view.observations) - mismatch) * math.log1p(-view.noise)
            + view.complexity_penalty * program.nodes)


@register("discovery-expander", "compositional", "Type-correct rule edits, alternatives, joins, and positive recursion")
class CompositionalExpander:
    def propose(self, view, budget=2000):
        budget = min(budget, 2000)
        sequences = tuple(o.sequence for o in view.observations)
        proposals, seen, single = [], set(), {}
        ceiling = max(1, int(budget * 0.4))

        def offer(program, strategy, retain=True, parents=()):
            if len(proposals) >= ceiling:
                return
            if program.identifier in seen:
                return
            seen.add(program.identifier)
            try:
                validate(program, view.schema, view.arity, view.library)
                cost = fit_cost(program, view)
            except ValueError:
                return
            if len(program.clauses) == 1 and program.clauses[0].head.predicate == "out":
                single[program.identifier] = (cost, program)
            if retain and len(proposals) < budget:
                proposals.append((cost, ExpansionProposal(program, strategy, sequences, tuple(parents))))

        # Preserve the clause vocabulary of prior candidate explanations.
        try:
            for parent in view.parents:
                for clause in parent.clauses:
                    if clause.head.predicate == "out" and all(a.predicate in view.schema or a.predicate in view.library for a in clause.body):
                        offer(program_for(clause.body, view.arity, view.library), "condition-revision", parents=(parent.identifier,))
            atoms = ordinary_atoms(view.schema, view.arity, view.library)
            ceiling = max(1, int(budget * 0.65))
            # Joins introduce a third variable through a typed, shared relationship.
            # The same construction is available for every schema; it never sees a world name.
            if view.arity == 2:
                fragments = []
                for name, sorts in sorted(view.schema.items()):
                    if sorts == ("object", "value"):
                        fragments.append((Atom(name, (0, 2)), Atom(name, (1, 2))))
                    elif sorts == ("object", "object"):
                        fragments.append((Atom(name, (0, 2)), Atom(name, (2, 1))))
                for name in sorted(view.library):
                    fragments.append((Atom(name, (0, 2)), Atom(name, (2, 1))))
                for fragment in fragments:
                    for size in range(3):
                        for extra in itertools.combinations(atoms, size):
                            body = tuple(sorted(set(fragment + extra)))
                            if consistent(body):
                                offer(program_for(body, view.arity, view.library), "shared-variable-join")
                # Imported concepts participate as ordinary relations, with no task-specific route.
                for name in sorted(view.library):
                    for pair in ((0, 1), (1, 0)):
                        for size in range(3):
                            for extra in itertools.combinations(atoms, size):
                                body = tuple(sorted(set((Atom(name, pair),) + extra)))
                                if consistent(body):
                                    offer(program_for(body, view.arity, view.library), "concept-reuse")
            # Single-clause growth; noisy failures rank candidates, never eliminate them as facts.
            ceiling = max(1, int(budget * 0.75))
            for size in range(0 if view.arity == 0 else 1, 5):
                for body in itertools.combinations(atoms, size):
                    if len(single) >= min(1300, budget):
                        break
                    if consistent(body):
                        offer(program_for(body, view.arity, view.library), "condition-revision")
            # Alternatives use clauses with useful positive coverage, instead of merely
            # combining the single rules that favor an all-negative majority.
            def coverage(item):
                _, program = item
                pred = view.evaluator.evaluate(program)
                values = pred[view.indices]
                return (3 * np.count_nonzero(values & ~view.labels)
                        - np.count_nonzero(values & view.labels) + 0.15 * program.nodes,
                        program.identifier)
            useful = [p for _, p in sorted(single.values(), key=coverage)[:12]]
            ceiling = max(1, int(budget * 0.95))
            for size in (2, 3):
                for group in itertools.combinations(useful, size):
                    clauses = tuple(c for p in group for c in p.clauses)
                    offer(Program.make(clauses, imports_for(clauses, view.library)), "alternative-rules", parents=tuple(p.identifier for p in group))
            if view.arity == 2:
                ceiling = budget
                # A generic positive recursion production, not a supplied reachability primitive.
                ranked = sorted(single.values(), key=lambda item: (item[0], item[1].identifier))[:48]
                for _, parent in ranked:
                    local = Clause(Atom("h0", (0, 1)), parent.clauses[0].body)
                    clauses = (local, Clause(Atom("h1", (0, 1)), (Atom("h0", (0, 1)),)),
                               Clause(Atom("h1", (0, 1)), (Atom("h0", (0, 2)), Atom("h1", (2, 1)))),
                               Clause(target_atom(2), (Atom("h1", (0, 1)),)))
                    offer(Program.make(clauses, imports_for(clauses, view.library)), "positive-recursion", parents=(parent.identifier,))
        except BudgetExhausted:
            pass
        # The cap is on returned proposals and their scored constructions, not on truth labels.
        return [proposal for _, proposal in sorted(proposals, key=lambda item: (item[0], item[1].program.identifier))[:budget]]


@register("discovery-library", "shared-patterns", "Versioned relational helpers admitted by whole-corpus compression")
class SharedPatternLibrary:
    def __init__(self, limit=8):
        self.limit = limit
        self.records = []
        self.concepts = {}
        self.metadata = []

    @property
    def definitions(self):
        return dict(self.concepts)

    def record(self, program, task_id):
        self.records.append((task_id, program))
        occurrences = defaultdict(list)
        for source, candidate in self.records:
            for index, clause in enumerate(candidate.clauses):
                if clause.head != target_atom(2) or any(a.predicate not in {"obj", "edge", "tag0", "tag1", "u", "nu", "v", "nv"} for a in clause.body):
                    continue
                for size in range(2, len(clause.body) + 1):
                    for body in itertools.combinations(clause.body, size):
                        if {0, 1}.issubset({v for a in body for v in a.variables}):
                            occurrences[body].append((source, candidate.identifier, index))
        accepted = []
        existing = {c.body for c in self.concepts.values()}
        ranked = []
        for body, uses in occurrences.items():
            uses = sorted(set(uses))
            savings = len(uses) * (len(body) - 1) - (1 + len(body))
            if len(uses) >= 3 and len({u[0] for u in uses}) >= 2 and savings > 0 and body not in existing:
                ranked.append((-savings, body, uses))
        for negative_savings, body, uses in sorted(ranked):
            if len(self.concepts) >= self.limit:
                break
            name = f"concept_{len(self.concepts) + 1:03d}"
            definition = Clause(Atom(name, (0, 1)), body)
            self.concepts[name] = definition
            metadata = {"name": name, "version": 1, "uses": len(uses), "source_tasks": sorted({u[0] for u in uses}),
                        "program_ids": sorted({u[1] for u in uses}), "description_savings_nodes": -negative_savings,
                        "definition_cost_nodes": 1 + len(body),
                        "definition": Program.make((definition,)).to_dict(), "conditional_on_source_programs": True}
            self.metadata.append(metadata)
            accepted.append(metadata)
        return accepted

    def snapshot(self):
        return {"schema": 1, "limit": self.limit, "records": [[source, p.to_dict()] for source, p in self.records],
                "metadata": self.metadata}

    def restore(self, state):
        if state["schema"] != 1 or state["limit"] != self.limit:
            raise ValueError("library checkpoint configuration mismatch")
        self.records = [(source, Program.from_dict(p)) for source, p in state["records"]]
        self.metadata = list(state["metadata"])
        self.concepts = {m["name"]: Program.from_dict(m["definition"]).clauses[0] for m in self.metadata}
