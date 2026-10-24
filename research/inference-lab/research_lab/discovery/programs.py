from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Protocol

import numpy as np

from ..common import ROOT, digest


@dataclass(frozen=True, order=True)
class Atom:
    predicate: str
    variables: tuple[int, ...] = ()


@dataclass(frozen=True, order=True)
class Clause:
    head: Atom
    body: tuple[Atom, ...]


@dataclass(frozen=True)
class Program:
    clauses: tuple[Clause, ...] = ()
    imports: tuple[str, ...] = ()

    @staticmethod
    def make(clauses=(), imports=()):
        return Program(tuple(sorted(set(Clause(c.head, tuple(sorted(set(c.body)))) for c in clauses))),
                       tuple(sorted(set(imports))))

    @cached_property
    def identifier(self):
        return digest(self.to_dict())[:20]

    @property
    def nodes(self):
        return sum(1 + len(c.body) for c in self.clauses)

    def to_dict(self):
        return {"clauses": [{"head": [c.head.predicate, list(c.head.variables)],
                             "body": [[a.predicate, list(a.variables)] for a in c.body]} for c in self.clauses],
                "imports": list(self.imports)}

    @staticmethod
    def from_dict(value):
        return Program.make([Clause(Atom(c["head"][0], tuple(c["head"][1])),
                                    tuple(Atom(p, tuple(v)) for p, v in c["body"])) for c in value["clauses"]], value["imports"])

    def text(self):
        def atom(a):
            return a.predicate + ("(" + ",".join("XYZ"[v] for v in a.variables) + ")" if a.variables else "")
        return "\n".join(atom(c.head) + (" :- " + ", ".join(atom(a) for a in c.body) if c.body else "") + "." for c in self.clauses) or "No rule predicts a positive outcome."


@dataclass(frozen=True)
class Scene:
    # Immutable public input. No target, phase, clean outcome, or noise parameters.
    entities: tuple[str, ...]
    facts: tuple[tuple[str, tuple[int, ...]], ...]
    query: tuple[int, ...]
    group: str

    def to_dict(self):
        return {"entities": self.entities, "facts": self.facts, "query": self.query, "group": self.group}

    @staticmethod
    def from_dict(v):
        return Scene(tuple(v["entities"]), tuple((p, tuple(args)) for p, args in v["facts"]), tuple(v["query"]), v["group"])


@dataclass(frozen=True)
class Question:
    index: int
    policy: str
    reason: str


@dataclass(frozen=True)
class Feedback:
    sequence: int
    question_index: int
    label: bool
    source: str


@dataclass(frozen=True)
class ExpansionProposal:
    program: Program
    strategy: str
    evidence_sequences: tuple[int, ...]
    parent_ids: tuple[str, ...]


class Expander(Protocol):
    def propose(self, view, budget: int) -> list[ExpansionProposal]: ...


class ConceptLibrary(Protocol):
    def record(self, program: Program, task_id: str) -> list[dict]: ...
    def snapshot(self) -> dict: ...
    def restore(self, state: dict) -> None: ...


class World(Protocol):
    def answer(self, question_index: int) -> Feedback: ...


def target_atom(arity):
    return Atom("out", tuple(range(arity)))


def validate(program, schema, arity, library=None):
    library = library or {}
    return dict(_validate(program, tuple(sorted(schema.items())), arity, tuple(sorted(library))))


@lru_cache(maxsize=16384)
def _validate(program, schema_items, arity, library_names):
    schema, library = dict(schema_items), set(library_names)
    if len(program.clauses) > 4 or sum(len(c.body) for c in program.clauses) > 12:
        raise ValueError("program exceeds clause/body budget")
    names = {c.head.predicate for c in program.clauses}
    if names & set(schema) or names & set(library) or len(names - {"out"}) > 2:
        raise ValueError("invalid or excessive invented predicates")
    if set(program.imports) - library:
        raise ValueError("unknown concept import")
    declarations = {**schema, "out": ("object",) * arity,
                    **{name: ("object", "object") for name in names - {"out"}},
                    **{name: ("object", "object") for name in program.imports}}
    for clause in program.clauses:
        if clause.head.variables != tuple(range(len(clause.head.variables))):
            raise ValueError("heads must use canonical distinct variables")
        body_variables = {v for a in clause.body for v in a.variables}
        if not set(clause.head.variables).issubset(body_variables):
            raise ValueError("unsafe rule: head variable is not bound")
        sorts = {}
        for atom in (clause.head, *clause.body):
            if atom.predicate not in declarations or len(atom.variables) != len(declarations[atom.predicate]):
                raise ValueError("undeclared predicate or invalid arity")
            for variable, kind in zip(atom.variables, declarations[atom.predicate]):
                if type(variable) is not int or not 0 <= variable < 3:
                    raise ValueError("at most three variables, no object constants")
                if variable in sorts and sorts[variable] != kind:
                    raise ValueError("variable type mismatch")
                sorts[variable] = kind
    return declarations


def definitions(program, library):
    return tuple(program.clauses) + tuple(library[name] for name in program.imports)


def reference_closure(program, scene, schema, library=None):
    library = library or {}
    declarations = validate(program, schema, len(scene.query), library)
    declarations.update({name: ("object", "object") for name in program.imports})
    facts = set(scene.facts)
    clauses = definitions(program, library)
    for _ in range(10000):
        added = set()
        for clause in clauses:
            bindings = [dict()]
            for atom in clause.body:
                following = []
                for binding in bindings:
                    for predicate, arguments in facts:
                        if predicate != atom.predicate:
                            continue
                        extended = dict(binding)
                        for variable, value in zip(atom.variables, arguments):
                            if variable in extended and extended[variable] != value:
                                break
                            extended[variable] = value
                        else:
                            following.append(extended)
                bindings = following
            added.update((clause.head.predicate, tuple(b[v] for v in clause.head.variables)) for b in bindings)
        added -= facts
        if not added:
            return facts
        facts |= added
    raise AssertionError("finite positive program failed to reach a fixed point")


class BatchEvaluator:
    """Boolean relational algebra over public scenes; independent of target generators."""
    def __init__(self, scenes, schema, library=None, budget=None):
        self.scenes, self.schema, self.library = tuple(scenes), schema, library or {}
        self.arity = len(scenes[0].query)
        self.batch, self.domain = len(scenes), 8
        self.budget = budget
        self.context_evaluations = 0
        self.cache, self.body_cache = {}, {}
        self.extensional = {name: np.zeros((self.batch,) + (8,) * len(types), dtype=bool) for name, types in schema.items()}
        for b, scene in enumerate(scenes):
            if len(scene.entities) > 8 or len(scene.query) != self.arity:
                raise ValueError("scene exceeds public bounds")
            for predicate, args in scene.facts:
                if predicate not in schema or len(args) != len(schema[predicate]) or any(not 0 <= a < 8 for a in args):
                    raise ValueError("invalid public fact")
                self.extensional[predicate][(b, *args)] = True

    def _clause(self, clause, tables):
        if not clause.body:
            return np.ones((self.batch,), dtype=bool)
        equation = ",".join("b" + "".join("ijk"[v] for v in atom.variables) for atom in clause.body)
        equation += "->b" + "".join("ijk"[v] for v in clause.head.variables)
        return np.einsum(equation, *[tables[a.predicate] for a in clause.body], optimize=False)

    def evaluate(self, program):
        if program.identifier in self.cache:
            return self.cache[program.identifier]
        if self.budget is not None and self.context_evaluations + self.batch > self.budget:
            raise BudgetExhausted("program-context evaluation budget exhausted")
        declarations = validate(program, self.schema, self.arity, self.library)
        clauses = definitions(program, self.library)
        self.context_evaluations += self.batch
        tables = dict(self.extensional)
        heads = {c.head.predicate for c in clauses}
        for name in heads:
            tables[name] = np.zeros((self.batch,) + (8,) * len(declarations[name]), dtype=bool)
        if "out" not in tables:
            tables["out"] = np.zeros((self.batch,) + (8,) * self.arity, dtype=bool)
        # Dependency scheduling avoids repeating nonrecursive relation joins.
        pending = set(heads)
        done = set(self.schema)
        while pending:
            ready = [name for name in sorted(pending) if all(a.predicate in done for c in clauses if c.head.predicate == name for a in c.body)]
            if not ready:
                # Remaining positive recursion converges over the finite relation tables.
                for _ in range(1 + len(pending) * 64):
                    changed = False
                    for clause in clauses:
                        if clause.head.predicate not in pending:
                            continue
                        derived = self._clause(clause, tables)
                        before = int(tables[clause.head.predicate].sum())
                        tables[clause.head.predicate] |= derived
                        changed |= int(tables[clause.head.predicate].sum()) != before
                    if not changed:
                        break
                else:
                    raise AssertionError("finite closure bound exceeded")
                break
            for name in ready:
                for clause in clauses:
                    if clause.head.predicate == name:
                        key = (clause.head.variables, clause.body)
                        extensional = all(a.predicate in self.schema for a in clause.body)
                        if extensional and key in self.body_cache:
                            derived = self.body_cache[key]
                        else:
                            derived = self._clause(clause, tables)
                            if extensional:
                                self.body_cache[key] = derived
                        tables[name] |= derived
                done.add(name)
                pending.remove(name)
        result = np.array([tables["out"][(b, *scene.query)] for b, scene in enumerate(self.scenes)], dtype=bool)
        result.flags.writeable = False
        self.cache[program.identifier] = result
        return result


class BudgetExhausted(RuntimeError):
    pass


def rust_audit(program, scene, schema, library=None):
    library = library or {}
    declarations = validate(program, schema, len(scene.query), library)
    clauses = definitions(program, library)
    propositions = set(scene.facts) | {("out", scene.query)}
    grounded = []
    for clause in clauses:
        variables = sorted({v for atom in (clause.head, *clause.body) for v in atom.variables})
        sorts = {v: kind for atom in (clause.head, *clause.body) for v, kind in zip(atom.variables, declarations[atom.predicate])}
        domains = {kind: sorted({arg for pred, args in scene.facts for arg, sort in zip(args, schema[pred]) if sort == kind}) for kind in ("object", "value")}
        for values in itertools.product(*(domains[sorts[v]] for v in variables)):
            binding = dict(zip(variables, values))
            def ground(atom):
                return atom.predicate, tuple(binding[v] for v in atom.variables)
            head, body = ground(clause.head), tuple(ground(a) for a in clause.body)
            propositions.add(head)
            propositions.update(body)
            grounded.append((head, body))
    ids = {fact: index for index, fact in enumerate(sorted(propositions))}
    lines = [f"s {ids[fact]}" for fact in sorted(scene.facts)]
    lines += [" ".join(map(str, ["r", ids[head], *[ids[a] for a in body]])) for head, body in grounded]
    with tempfile.TemporaryDirectory(prefix="discovery-proof-") as tmp:
        path = Path(tmp) / "program.txt"
        path.write_text("\n".join(lines) + "\n")
        result = json.loads(subprocess.check_output([str(ROOT / "target/release/bench"), "infer", str(path)], text=True))
    expected = reference_closure(program, scene, schema, library)
    actual = {fact for fact, index in ids.items() if index in set(result["fact_ids"])}
    if actual != expected:
        raise AssertionError("Python/Rust full closure disagreement")
    return {"conditional_on_program": True, "outcome": ("out", scene.query) in actual,
            "program": program.to_dict(), "source_facts": scene.facts,
            "fact_dictionary": [[index, predicate, args] for (predicate, args), index in ids.items()],
            "inference": result}
