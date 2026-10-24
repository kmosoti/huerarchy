from __future__ import annotations

from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np

from ..common import ROOT, canonical, digest, file_hash, peak_rss_bytes, write_json
from ..discovery.experiment import scores
from ..discovery.programs import BatchEvaluator, Program, rust_audit
from ..learning.components import SQLiteMemory
from ..learning.types import LearningEvent, create
from ..learning.worlds import seed_for
from . import concepts, worlds
from .learner import ActiveLearner, evaluate_state, expanded_program
from .legacy import LegacyAdapter
from .types import CORE, LIBRARIES, TaskView


def source_hashes():
    paths = list((ROOT / "research_lab/active").glob("*.py")) + list((ROOT / "tests").glob("test_active*.py"))
    paths += list((ROOT / "research_lab/discovery").glob("*.py"))
    paths += [ROOT / "research_lab/learning/components.py", ROOT / "research_lab/learning/types.py", ROOT / "research_lab/learning/worlds.py",
              ROOT / "research_lab/common.py", ROOT / "research_lab/__main__.py"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def learner_for(task, config, variant, seed, library=None, transfer=False, initialize=True, state=None):
    if variant.startswith(("broad", "gate")):
        return LegacyAdapter(task.view, config, variant, seed, initialize, state["config"] if state else None), None
    policy, mode, archive = CORE.get(variant, CORE["epig-change"])
    learner = ActiveLearner(task.view, config, mode, library, archive, transfer, initialize)
    query = create("active-policy", config["components"]["policy"], kind=policy, seed=seed, exploration=config["exploration"])
    return learner, query


def next_query(learner, policy):
    return learner.choose() if policy is None else policy.choose(learner.conditional, learner.question_weights())


def threshold_delay(values, begin, end):
    for i in range(begin, end - 9):
        if min(values[i:i + 10]) >= .9:
            return i + 10 - begin
    return None


class Writer:
    def __init__(self, output, config):
        self.output, self.config = Path(output), config
        self.output.mkdir(parents=True, exist_ok=False)
        self.connection = sqlite3.connect(self.output / "memory.sqlite")
        self.runs = (self.output / "runs.jsonl").open("w")
        self.trace = gzip.open(self.output / "trace.jsonl.gz", "wt")
        self.events = gzip.open(self.output / "events.jsonl.gz", "wt")
        self.concepts = (self.output / "concepts.jsonl").open("w")
        self.proofs = (self.output / "proofs.jsonl").open("w")
        self.tasks, self.records = {}, []
        self.proof_count = 0
        self.proof_keys = set()
        self.started, self.cpu_start = time.monotonic(), time.process_time()
        self.sources = source_hashes()
        write_json(self.output / "manifest.json", {"schema": 1, "status": "running", "sources": self.sources})

    def line(self, stream, value):
        stream.write(canonical(value).decode() + "\n")

    def finish(self, complete):
        for stream in (self.runs, self.trace, self.events, self.concepts, self.proofs):
            stream.close()
        self.connection.commit()
        if self.connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("experience database corruption")
        self.connection.close()
        with gzip.open(self.output / "worlds.json.gz", "wt") as stream:
            json.dump(self.tasks, stream)
        names = ("runs.jsonl", "trace.jsonl.gz", "events.jsonl.gz", "concepts.jsonl", "proofs.jsonl", "memory.sqlite", "worlds.json.gz")
        if self.sources != source_hashes():
            raise ValueError("source changed during experimental block")
        manifest = {"schema": 1, "status": "complete" if complete else "partial", "sources": self.sources,
                    "config_sha256": digest(self.config), "tasks": len(self.records),
                    "processed_answers": sum(r["answers"] for r in self.records),
                    "acquired_answers": sum(r["acquired_answers"] for r in self.records),
                    "cpu_seconds": time.process_time() - self.cpu_start,
                    "wall_seconds": time.monotonic() - self.started, "peak_process_rss_bytes": peak_rss_bytes(),
                    "proofs": self.proof_count, "outputs": {name: file_hash(self.output / name) for name in names}}
        write_json(self.output / "manifest.json", manifest)
        return manifest


def run_task(writer, task, variant, noise, seed, *, steps=None, library=None, transfer=False,
             forced=None, feedbacks=None, role="primary", acquisition_before=0, extra_cpu=0, extra_search=0, deadline=float("inf")):
    config = writer.config
    steps = steps or config["steps"]
    run_id = f"{task.identifier}/noise-{noise:g}/seed-{seed}/{variant}"
    writer.tasks[task.identifier] = task.to_dict()
    started, cpu = time.monotonic(), time.process_time()
    query_seed = seed_for("active-question", task.identifier, seed)
    learner, policy = learner_for(task, config, variant, query_seed, library, transfer)
    teacher = create("active-teacher", config["components"]["teacher"], task=task, noise=noise,
                     seed=seed_for("active-feedback", task.identifier, noise, seed), steps=steps)
    evaluator = BatchEvaluator(task.evaluation, task.view.schema, library)
    memory = SQLiteMemory(writer.connection, run_id)
    outcomes = [np.array([task.outcome(scene, i) for scene in task.evaluation]) for i in range(1, steps + 1)]
    accuracy, losses, observed, collected, briers, confidences, errors = [], [], [], [], [], [], []
    query_cpu = 0.0
    event_counts, last = {}, None
    for sequence in range(1, steps + 1):
        if time.time() >= deadline:
            break
        before = time.process_time()
        index, info = (forced[sequence - 1], {"policy": "shared-uniform"}) if forced is not None else next_query(learner, policy)
        query_cpu += time.process_time() - before
        feedback = feedbacks[sequence - 1] if feedbacks is not None else teacher.answer(index)
        if feedback.question_index != index or feedback.sequence != sequence:
            raise ValueError("shared validation stream mismatch")
        result = learner.observe(feedback)
        prediction = evaluate_state(learner.prediction_state(), evaluator)
        metrics = scores(prediction, outcomes[sequence - 1])
        confidence = np.maximum(prediction, 1 - prediction)
        correct = np.where(prediction >= .5, outcomes[sequence - 1], 1 - outcomes[sequence - 1])
        bins = np.minimum(9, (confidence * 10).astype(int))
        payload = {"feedback": asdict(feedback), "question": info, "pre_probability": result["pre_probability"],
                   "pre_observed_probability": result["pre_observed_probability"], "pre_log_loss": result["pre_log_loss"]}
        memory.append(LearningEvent(sequence, "observation", payload))
        for i, event in enumerate(result["events"]):
            event_counts[event["event"]] = event_counts.get(event["event"], 0) + 1
            memory.append(LearningEvent(sequence, f"decision-{i}", event))
            writer.line(writer.events, {"run_id": run_id, "sequence": sequence, **event})
        last = {"run_id": run_id, "sequence": sequence, **payload, **metrics,
                "confidence_counts": np.bincount(bins, minlength=10).tolist(),
                "confidence_sum": np.bincount(bins, confidence, minlength=10).tolist(),
                "correct_sum": np.bincount(bins, correct, minlength=10).tolist(),
                "coin_mass": learner.program_weights()[1] if isinstance(learner, ActiveLearner) else float(learner.inner.uncertain)}
        writer.line(writer.trace, last)
        accuracy.append(metrics["balanced_accuracy"])
        losses.append(metrics["expected_log_loss"])
        briers.append(metrics["expected_brier"])
        confidences.append(metrics["confident_coverage"])
        errors.append(metrics["confident_error_rate"])
        observed.append(result["pre_log_loss"])
        collected.append(feedback)
    state = {"schema": 1, "variant": variant, "config": config, "task": task.identifier, "noise": noise, "seed": seed,
             "transfer": transfer, "library": {k: Program.make((v,)).to_dict() for k, v in (library or {}).items()},
             "learner": learner.snapshot(), "policy": policy.snapshot() if policy else None, "teacher": teacher.snapshot()}
    memory.checkpoint(state)
    saved = memory.restore()
    restored, restored_policy = learner_for(task, config, variant, 0, library, transfer, False, saved["learner"])
    restored.restore(saved["learner"])
    if policy:
        restored_policy.restore(saved["policy"])
    a, b = learner.prediction_state(), restored.prediction_state()
    if [p.identifier for p in a[0]] != [p.identifier for p in b[0]] or not np.array_equal(a[1], b[1]) or a[2] != b[2]:
        raise AssertionError("checkpoint changed predictions")
    if next_query(learner, policy) != next_query(restored, restored_policy):
        raise AssertionError("checkpoint changed next question")
    best, mass = learner.best()
    proof_key = (task.family, role, bool(best.imports))
    if (writer.proof_count < 12 and proof_key not in writer.proof_keys and len(collected) == steps
            and (variant == "epig-change" or role in {"target", "validation", "shadow"})):
        proof = rust_audit(best, task.evaluation[0], task.view.schema, library)
        writer.line(writer.proofs, {"run_id": run_id, "mass": mass, "proof": proof})
        writer.proof_count += 1
        writer.proof_keys.add(proof_key)
    boundaries = [0, *[x for x in task.changes if x < len(accuracy)], len(accuracy)]
    phases = []
    for start, end in zip(boundaries, boundaries[1:]):
        success = threshold_delay(accuracy, start, end)
        phases.append({"start": start, "horizon": end - start, "delay": success,
                       "capped_cost": success if success is not None else end - start})
    def mean(values):
        return float(np.mean(values)) if values else None
    record = {"run_id": run_id, "task": task.identifier, "variant": variant, "role": role, "family": task.family,
              "noise": noise, "seed": seed, "answers": len(collected), "acquired_answers": 0 if feedbacks is not None else len(collected),
              "acquisition_before": acquisition_before, "cpu_before": extra_cpu, "search_before": extra_search,
              "status": "complete" if len(collected) == steps else "partial", "changes": task.changes,
              "accuracy": mean(accuracy), "final_accuracy": mean(accuracy[-20:]), "brier": mean(briers),
              "log_loss": mean(losses), "observed_log_loss": mean(observed), "confident_coverage": mean(confidences),
              "post32_brier": mean(briers[32:]), "post32_confident_coverage": mean(confidences[32:]),
              "phases": phases, "query_cpu_seconds": query_cpu, "cpu_seconds": time.process_time() - cpu,
              "wall_seconds": time.monotonic() - started, "search_contexts": learner.engine.context_evaluations,
              "evaluation_contexts": evaluator.context_evaluations, "budget_exhausted": learner.budget_exhausted,
              "events": event_counts, "program": best.to_dict(), "program_text": best.text(), "mass": mass,
              "program_nodes": best.nodes, "expanded_program_nodes": expanded_program(best, library or {}).nodes,
              "imported_concepts": list(best.imports), "checkpoint_verified": True,
              "coin_mass": last["coin_mass"] if last else None,
              "pruned_mass_sum": learner.posterior.pruned_mass if isinstance(learner, ActiveLearner) else None}
    writer.line(writer.runs, record)
    writer.runs.flush()
    writer.connection.commit()
    writer.records.append(record)
    if len(writer.records) % 50 == 0:
        print(f"active {writer.output.name}: {len(writer.records)} tasks recorded", flush=True)
    return record, collected, observed


def run_core_block(output, block, config, deadline=float("inf")):
    writer = Writer(output, config)
    complete = True
    for regime in ("stable", "abc", "aba", "diagnostic"):
        tasks = worlds.controls(block, config) if regime == "diagnostic" else worlds.core_tasks(block, regime, config)
        for task in tasks:
            variants = ["broad-upfront", "broad-window32", "eig-window", "epig-window", "eig-change", "epig-change", "random-window"]
            if regime == "aba":
                variants.append("epig-change-archive")
            if task.family in {"relay", "coin"}:
                variants += ["gate32", "gate8", "gate-sequential"]
            for noise in config["noise_rates"]:
                for seed in config["seeds"]:
                    for variant in variants:
                        if time.time() >= deadline:
                            complete = False
                            break
                        record, _, _ = run_task(writer, task, variant, noise, seed, deadline=deadline)
                        complete &= record["status"] == "complete"
                    if not complete:
                        break
                if not complete:
                    break
            if not complete:
                break
        if not complete:
            break
    return writer.finish(complete)


def run_transfer_block(output, block, config, deadline=float("inf")):
    writer = Writer(output, config)
    tasks = worlds.transfer_tasks(block, config)
    complete = True
    for noise in config["noise_rates"]:
        for seed in config["seeds"]:
            shared_labels, shared_cpu, shared_search = 0, 0.0, 0
            libraries = {kind: create("active-library", config["components"]["library"], kind=kind, limit=config["concept_limit"]) for kind in LIBRARIES}
            for task in tasks[:3]:
                forced = np.random.default_rng(seed_for("transfer-uniform", task.identifier, seed)).integers(len(task.view.questions), size=config["source_steps"]).tolist()
                record, _, _ = run_task(writer, task, "epig-change", noise, seed, steps=config["source_steps"],
                                        forced=forced, transfer=True, role="source", deadline=deadline)
                shared_labels += record["answers"]
                shared_cpu += record["cpu_seconds"]
                shared_search += record["search_contexts"]
                for lib in libraries.values():
                    lib.record(Program.from_dict(record["program"]), task.identifier, task.view.schema, record["mass"])
                if record["status"] != "complete":
                    complete = False
                    break
            if not complete:
                break
            for lib in libraries.values():
                lib.mine()
            validated = libraries["validated"]
            for task in tasks[3:5]:
                forced = np.random.default_rng(seed_for("transfer-uniform", task.identifier, seed)).integers(len(task.view.questions), size=config["validation_steps"]).tolist()
                record, feedbacks, baseline = run_task(writer, task, "validation-reference", noise, seed, steps=config["validation_steps"],
                    forced=forced, transfer=True, role="validation", deadline=deadline)
                shared_labels += record["answers"]
                shared_cpu += record["cpu_seconds"]
                shared_search += record["search_contexts"]
                if record["status"] != "complete":
                    complete = False
                    break
                for concept in validated.concepts:
                    if time.time() >= deadline:
                        complete = False
                        break
                    definitions, _ = validated.instantiate(task.view.schema, {concept["id"]})
                    shadow, _, observed = run_task(writer, task, "validation-" + concept["id"], noise, seed,
                        steps=config["validation_steps"], forced=forced, feedbacks=feedbacks, library=definitions,
                        transfer=True, role="shadow", deadline=deadline)
                    shared_cpu += shadow["cpu_seconds"]
                    shared_search += shadow["search_contexts"]
                    if shadow["status"] != "complete":
                        complete = False
                        break
                    warmup = config["validation_warmup"]
                    gain = sum(baseline[warmup:]) - sum(observed[warmup:])
                    validated.validate_transfer(concept["id"], task.identifier, gain, len(observed) - warmup, config["concept_gain"])
                if not complete:
                    break
            if not complete:
                break
            for kind, lib in libraries.items():
                writer.line(writer.concepts, {"block": block, "noise": noise, "seed": seed, "library": lib.snapshot()})
                for task in tasks[5:]:
                    definitions, provenance = lib.instantiate(task.view.schema, promoted_only=kind == "validated")
                    writer.line(writer.events, {"event": "library-instantiation", "task": task.identifier, "kind": kind,
                                                "noise": noise, "seed": seed, "instances": provenance})
                    record, _, _ = run_task(writer, task, "library-" + kind, noise, seed, library=definitions,
                        transfer=True, role="target", acquisition_before=shared_labels, extra_cpu=shared_cpu, extra_search=shared_search, deadline=deadline)
                    if record["status"] != "complete":
                        complete = False
                        break
                if not complete:
                    break
            if not complete:
                break
        if not complete:
            break
    return writer.finish(complete)


def verify_block(directory):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    if m.get("schema") != 1 or m.get("status") not in {"complete", "partial"}:
        raise ValueError("unfinished experimental block")
    for filename, expected in m["outputs"].items():
        if Path(filename).name != filename or file_hash(directory / filename) != expected:
            raise ValueError(f"changed artifact: {filename}")
    return m
