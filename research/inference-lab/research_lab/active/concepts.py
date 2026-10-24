from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import itertools
import math

from ..common import digest
from ..discovery.programs import Atom, Clause, Program, validate
from ..learning.types import register


def connected(body):
    variables = {v for a in body for v in a.variables}
    reached = {0}
    for _ in range(3):
        for atom in body:
            if reached.intersection(atom.variables):
                reached.update(atom.variables)
    return {0, 1}.issubset(variables) and variables <= reached


def template(body, schema, typed):
    if not typed:
        return tuple(sorted(body)), {a.predicate: schema[a.predicate] for a in body}
    groups = defaultdict(list)
    for name in sorted({a.predicate for a in body}):
        groups[tuple(schema[name])].append(name)
    choices, declarations = [], {}
    offset = 0
    for kinds, names in sorted(groups.items()):
        slots = [f"slot{i}" for i in range(offset, offset + len(names))]
        declarations.update({slot: kinds for slot in slots})
        choices.append([dict(zip(order, slots)) for order in itertools.permutations(names)])
        offset += len(names)
    forms = []
    for choices_ in itertools.product(*choices):
        mapping = {k: v for choice in choices_ for k, v in choice.items()}
        forms.append(tuple(sorted(Atom(mapping[a.predicate], a.variables) for a in body)))
    return min(forms), declarations


@register("active-library", "templates", "Typed relational templates with conditional provenance and fresh transfer validation")
class TemplateLibrary:
    def __init__(self, kind="typed", limit=8):
        self.kind, self.limit = kind, limit
        self.sources, self.concepts = [], []

    def record(self, program, task, schema, mass):
        self.sources.append({"task": task, "program": program.to_dict(), "schema": schema, "mass": mass})

    def mine(self):
        if self.kind == "none":
            return []
        occurrences = defaultdict(list)
        specifications = {}
        for source in self.sources:
            p = Program.from_dict(source["program"])
            for index, clause in enumerate(p.clauses):
                if clause.head.variables != (0, 1) or any(a.predicate not in source["schema"] for a in clause.body):
                    continue
                for size in range(2, min(4, len(clause.body)) + 1):
                    for body in itertools.combinations(clause.body, size):
                        if not connected(body):
                            continue
                        atoms, declarations = template(body, source["schema"], self.kind != "exact")
                        key = digest({"body": [asdict(a) for a in atoms], "types": declarations})[:20]
                        specifications[key] = (atoms, declarations)
                        occurrences[key].append((source["task"], p.identifier, index))
        ranked = []
        for key, uses_ in occurrences.items():
            uses = sorted(set(uses_))
            body, declarations = specifications[key]
            savings = len(uses) * (len(body) - 1) - (1 + len(body))
            if len(uses) >= 3 and len({u[0] for u in uses}) >= 2 and savings > 0:
                ranked.append((-savings, key, uses, body, declarations))
        self.concepts = [{"id": key, "version": 1, "body": [asdict(a) for a in body], "types": declarations,
                          "sources": uses, "savings_nodes": -saving, "definition_nodes": 1 + len(body),
                          "status": "provisional", "validation": [], "conditional_on_source_programs": True}
                         for saving, key, uses, body, declarations in sorted(ranked)[:self.limit]]
        return self.concepts

    def instantiate(self, schema, concept_ids=None, promoted_only=False):
        definitions, provenance = {}, {}
        for concept in self.concepts:
            if concept_ids is not None and concept["id"] not in concept_ids:
                continue
            if promoted_only and concept["status"] != "promoted":
                continue
            slots = sorted(concept["types"])
            options = [[slot] if self.kind == "exact" and slot in schema else
                       [] if self.kind == "exact" else [n for n, t in sorted(schema.items()) if tuple(t) == tuple(concept["types"][slot])]
                       for slot in slots]
            emitted = 0
            for values in itertools.product(*options):
                if len(set(values)) != len(values):
                    continue
                mapping = dict(zip(slots, values))
                body = tuple(Atom(mapping[a["predicate"]], tuple(a["variables"])) for a in concept["body"])
                name = "c_" + digest({"id": concept["id"], "bindings": mapping})[:16]
                clause = Clause(Atom(name, (0, 1)), body)
                validate(Program.make((Clause(Atom("out", (0, 1)), body),)), schema, 2)
                definitions[name] = clause
                provenance[name] = {"concept": concept["id"], "bindings": mapping}
                emitted += 1
                if emitted == 8:
                    break
        return definitions, provenance

    def validate_transfer(self, concept_id, task, gain, responses, gain_required=math.log(20)):
        c = next(c for c in self.concepts if c["id"] == concept_id)
        if task in {v["task"] for v in c["validation"]}:
            raise ValueError("duplicate validation task")
        c["validation"].append({"task": task, "fresh_log_score_gain": float(gain), "responses": responses})
        values = c["validation"]
        if len(values) >= 2 and all(v["fresh_log_score_gain"] > 0 for v in values) and sum(v["fresh_log_score_gain"] for v in values) >= gain_required:
            c["status"] = "promoted"

    def snapshot(self):
        return {"kind": self.kind, "limit": self.limit, "sources": self.sources, "concepts": self.concepts}

    def restore(self, state):
        if (state["kind"], state["limit"]) != (self.kind, self.limit):
            raise ValueError("library configuration mismatch")
        self.sources, self.concepts = state["sources"], state["concepts"]
