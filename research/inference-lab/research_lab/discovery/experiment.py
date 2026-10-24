from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np

from ..common import ROOT, canonical, digest, file_hash, peak_rss_bytes, profile_environment, write_json
from ..learning.components import SQLiteMemory
from ..learning.types import LearningEvent, create
from ..learning.worlds import seed_for
from . import expansion
from .learner import DiscoveryLearner
from .programs import Program, rust_audit
from .worlds import DEFAULT, Teacher, clean_outcome, configuration, diagnostic_tasks, generate_curriculum


RESEARCH = [
    {"title": "Learning MDL Logic Programs from Noisy Data", "year": 2024,
     "url": "https://www.cs.helsinki.fi/u/mjarvisa/papers/hnjc.aaai24.pdf",
     "use": "Noise-tolerant program search with an explicit complexity cost.",
     "limit": "Our bounded grammar and fresh-response gate are a separate implementation; no published performance result is transferred."},
    {"title": "Top-Down Synthesis for Library Learning (Stitch)", "year": 2023,
     "url": "https://arxiv.org/abs/2211.16605",
     "use": "Mine repeated symbolic structure to construct a reusable library.",
     "limit": "Shared-body mining here is simpler than Stitch, and compression alone does not establish correctness or new expressive power."},
    {"title": "Hypothesis Generation and Inductive Inference in Children and Language Models", "year": 2026,
     "url": "https://arxiv.org/abs/2605.24528",
     "use": "Generate hypotheses during learning instead of only reweighting an initial set.",
     "limit": "Recent box-task study; neither a reproduction nor evidence that this learner models children."},
    {"title": "Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring", "year": 2026,
     "url": "https://proceedings.iclr.cc/paper_files/paper/2026/hash/803b9c4a8e4784072fdd791c54d614e2-Abstract-Conference.html",
     "use": "Compare prequential learning progress with uncertainty-seeking questions.",
     "limit": "Our grouped loss-window policy is inspired by the idea, not the neural LPM method or its guarantees."},
    {"title": "DiscoveryWorld", "year": 2024, "url": "https://allenai.github.io/discoveryworld/",
     "use": "Separate task completion from discovering explanations and executing experiments.",
     "limit": "These smaller worlds are custom diagnostics, not DiscoveryWorld benchmark results."},
    {"title": "Formal Concept Analysis", "year": None,
     "url": "https://iccl.inf.tu-dresden.de/web/Formale_Begriffsanalyse/en",
     "use": "A possible future alternative proposer based on object-attribute closure.",
     "limit": "Deferred: exact observed implications are not reliable causal laws under noisy feedback."},
    {"title": "ADVENT: LLM-Driven Automatic Predicate Invention for ILP", "year": 2026,
     "url": "https://arxiv.org/abs/2607.01585", "use": "A future language-model proposer behind the same verification boundary.",
     "limit": "Deferred to keep pretrained knowledge outside this experiment; published poker results do not establish general concept discovery."},
]


def scores(probabilities, truth):
    prediction = probabilities >= 0.5
    positive, negative = float(truth.sum()), float((1 - truth).sum())
    balanced = ((float(np.sum(prediction * truth)) / positive + float(np.sum((~prediction) * (1 - truth))) / negative) / 2
                if positive and negative else None)
    confidence = np.maximum(probabilities, 1 - probabilities) >= 0.9
    error = np.where(prediction, 1 - truth, truth)
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return {"balanced_accuracy": balanced,
            "expected_brier": float(np.mean((probabilities - truth) ** 2 + truth * (1 - truth))),
            "irreducible_brier": float(np.mean(truth * (1 - truth))),
            "expected_log_loss": float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))),
            "confident_coverage": float(np.mean(confidence)),
            "confident_error_rate": float(np.mean(error[confidence])) if np.any(confidence) else None}


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    required = {"config.json", "worlds.json.gz", "runs.jsonl", "trace.jsonl.gz", "events.jsonl.gz",
                "memory.sqlite", "concepts.jsonl", "explanations.jsonl", "research.json", "environment.json"}
    if manifest.get("schema") != 1 or manifest.get("status") != "complete" or not required.issubset(manifest["outputs"]):
        raise ValueError("discovery run is incomplete")
    for filename, expected in manifest["outputs"].items():
        if Path(filename).name != filename or file_hash(directory / filename) != expected:
            raise ValueError(f"discovery artifact changed: {filename}")
    return manifest


def run(overrides=None, output=None, *, _indices=None, _include_diagnostics=None, _render=True):
    config = configuration(overrides)
    if config["workers"] > 1 and _indices is None:
        from .parallel import run_parallel
        return run_parallel(config, output)
    output = Path(output or ROOT / "results/discovery" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    if not (ROOT / "target/release/bench").exists():
        raise ValueError("build the Rust closure engine first: cargo build --release")
    output.mkdir(parents=True, exist_ok=True)
    indices = range(config["curricula"]) if _indices is None else _indices
    curricula = [create("discovery-world", config["components"]["world"], index=index, config=config) for index in indices]
    include_diagnostics = config["diagnostics"] if _include_diagnostics is None else _include_diagnostics
    diagnostics = diagnostic_tasks(config) if include_diagnostics else ()
    write_json(output / "config.json", config)
    write_json(output / "research.json", {"researched_on": "2026-10-24", "sources": RESEARCH,
               "contribution": "Controlled comparison of existing ideas; no claim that the underlying algorithms are novel."})
    with gzip.open(output / "worlds.json.gz", "wt") as stream:
        json.dump({"schema": 1, "curricula": [{"id": c["id"], "changing": c["changing"], "tasks": [t.to_dict() for t in c["tasks"]]} for c in curricula],
                   "diagnostics": [t.to_dict() for t in diagnostics]}, stream)
    write_json(output / "environment.json", profile_environment())
    source_paths = sorted((ROOT / "research_lab/discovery").glob("*.py")) + sorted((ROOT / "tests").glob("test_discovery*.py"))
    source_paths += [ROOT / "research_lab/__main__.py", ROOT / "research_lab/common.py", ROOT / "research_lab/learning/components.py", ROOT / "research_lab/learning/types.py", ROOT / "research_lab/learning/worlds.py"]
    manifest = {"schema": 1, "status": "running", "config_sha256": digest(config),
                "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths},
                "rust_reasoner_sha256": file_hash(ROOT / "target/release/bench"), "outputs": {}}
    write_json(output / "manifest.json", manifest)
    connection = sqlite3.connect(output / "memory.sqlite")
    total = (len(curricula) * 5 + len(diagnostics)) * len(config["noise_rates"]) * len(config["seeds"]) * len(config["variants"])
    count = proofs = turns = 0
    start, cpu_start = time.perf_counter(), time.process_time()
    try:
        with (output / "runs.jsonl").open("w") as runs, gzip.open(output / "trace.jsonl.gz", "wt") as trace, \
                gzip.open(output / "events.jsonl.gz", "wt") as event_stream, \
                (output / "concepts.jsonl").open("w") as concept_stream, (output / "explanations.jsonl").open("w") as explanations:
            sequences = [(str(c["id"]), c["changing"], c["tasks"]) for c in curricula]
            sequences += [("diagnostics", False, diagnostics)] if diagnostics else []
            for curriculum_id, changing, tasks in sequences:
                for noise in config["noise_rates"]:
                    for seed in config["seeds"]:
                        for variant in config["variants"]:
                            library = create("discovery-library", config["components"]["library"])
                            library_enabled = variant in {"adaptive-library", "library-random", "library-progress"}
                            cumulative_labels = cumulative_search = 0
                            for task_index, task in enumerate(tasks):
                                run_id = f"{task.identifier}/noise-{noise:g}/seed-{seed}/{variant}"
                                task_start, task_cpu = time.perf_counter(), time.process_time()
                                learner = DiscoveryLearner(task.public, config, variant, library.definitions if library_enabled else {})
                                kind = "random" if variant == "library-random" else "progress" if variant == "library-progress" else "information"
                                policy = create("discovery-policy", config["components"]["policy"], kind=kind,
                                                seed=seed_for("discovery-questions", task.identifier, seed), exploration=config["exploration"])
                                teacher = create("discovery-teacher", config["components"]["teacher"], task=task, noise=noise,
                                                 seed=seed_for("discovery-teacher", task.identifier, noise, seed), steps=config["steps"])
                                memory = SQLiteMemory(connection, run_id)
                                truths = [np.array([clean_outcome(scene, target) for scene in task.public.evaluation]) for target in task.targets]
                                balanced, brier, coverage = [], [], []
                                alarms = accepted_growth = accepted_reset = accepted_uncertain = 0
                                observations_loss = 0.0
                                questions_seen = set()
                                event_counts = {}
                                for sequence in range(1, config["steps"] + 1):
                                    probabilities = learner.probabilities()
                                    question = policy.choose(probabilities, task.public.questions, config["assumed_noise"])
                                    feedback = teacher.answer(question.index)
                                    update = learner.observe(feedback)
                                    policy.observe(task.public.questions[question.index].group, update["pre_update_log_loss"])
                                    evaluated = scores(learner.probabilities(True), truths[task.phase(sequence)])
                                    balanced.append(evaluated["balanced_accuracy"])
                                    brier.append(evaluated["expected_brier"])
                                    coverage.append(evaluated["confident_coverage"])
                                    alarms += update["alarm"]
                                    observations_loss += update["pre_update_log_loss"]
                                    questions_seen.add(question.index)
                                    payload = {"observation": asdict(feedback), "question": asdict(question),
                                               "pre_update_probability": update["pre_update_probability"], "uncertain": update["uncertain"]}
                                    memory.append(LearningEvent(sequence, "teaching", payload))
                                    for event in update["events"]:
                                        event_counts[event["event"]] = event_counts.get(event["event"], 0) + 1
                                        memory.append(LearningEvent(sequence, event["event"], event))
                                        event_stream.write(canonical({"run_id": run_id, **event}).decode() + "\n")
                                        if event["event"] == "challenge-completed" and event["accepted"]:
                                            accepted_growth += event["candidate"] == "growth"
                                            accepted_reset += event["candidate"] == "reset"
                                            accepted_uncertain += event["candidate"] == "uncertain"
                                    trace.write(canonical({"run_id": run_id, "sequence": sequence, **payload,
                                                "pre_update_teacher_log_loss": update["pre_update_log_loss"],
                                                "alarm": update["alarm"], "active_programs": len(learner.programs), **evaluated}).decode() + "\n")
                                best, mass = learner.best()
                                old_library_size = len(library.concepts)
                                mined = library.record(best, task.identifier) if library_enabled else []
                                for concept in mined:
                                    concept_stream.write(canonical({"run_id": run_id, "curriculum": curriculum_id, "variant": variant,
                                                                   "noise": noise, "seed": seed, **concept}).decode() + "\n")
                                state = {"schema": 1, "config_sha256": digest(config), "learner": learner.snapshot(),
                                         "policy": policy.snapshot(), "library": library.snapshot(), "teacher": teacher.snapshot()}
                                memory.checkpoint(state)
                                saved = memory.restore()
                                restored = DiscoveryLearner(task.public, config, variant, learner.library, initialize_broad=False)
                                restored.restore(saved["learner"])
                                restored_policy = create("discovery-policy", config["components"]["policy"], kind=kind, seed=0, exploration=config["exploration"])
                                restored_policy.restore(saved["policy"])
                                if not np.array_equal(restored.probabilities(), learner.probabilities()):
                                    raise AssertionError("discovery checkpoint changed predictions")
                                if policy.choose(learner.probabilities(), task.public.questions, config["assumed_noise"]) != restored_policy.choose(
                                        restored.probabilities(), task.public.questions, config["assumed_noise"]):
                                    raise AssertionError("discovery checkpoint changed next question")
                                if (proofs < 48 and variant == "adaptive-library" and seed == config["seeds"][0]
                                        and noise == config["noise_rates"][0]):
                                    for scene in task.public.evaluation[:2]:
                                        audit = rust_audit(best, scene, task.public.schema, learner.library)
                                        explanations.write(canonical({"run_id": run_id, "posterior_mass": mass,
                                            "readable_program": best.text(), "audit": audit}).decode() + "\n")
                                        proofs += 1
                                cumulative_labels += config["steps"]
                                cumulative_search += learner.engine.context_evaluations
                                record = {"run_id": run_id, "curriculum": curriculum_id, "changing": changing,
                                          "task": task.identifier, "family": task.family, "diagnostic": task.diagnostic,
                                          "task_position": task_index, "noise": noise, "seed": seed, "variant": variant,
                                          "steps": config["steps"], "balanced_accuracy_auc": float(np.mean(balanced)),
                                          "final_balanced_accuracy": float(np.mean(balanced[-20:])),
                                          "expected_brier": float(np.mean(brier)), "confident_coverage": float(np.mean(coverage)),
                                          "teacher_prequential_log_loss": observations_loss / config["steps"],
                                          "alarms": int(alarms), "growth_promotions": accepted_growth,
                                          "confirmed_resets": accepted_reset, "uncertainty_admissions": accepted_uncertain,
                                          "immediate_resets": learner.resets - accepted_reset,
                                          "event_counts": event_counts, "proposals_returned": learner.proposals,
                                          "search_program_context_evaluations": learner.engine.context_evaluations,
                                          "evaluation_program_context_evaluations": learner.evaluation_engine.context_evaluations,
                                          "budget_exhausted": learner.budget_exhausted,
                                          "unique_questions": len(questions_seen), "available_at_task_start": old_library_size,
                                          "imported_concepts": len(best.imports),
                                          "new_concepts": len(mined), "available_concepts": len(library.concepts),
                                          "cumulative_acquisition_labels": cumulative_labels,
                                          "cumulative_acquisition_search_contexts": cumulative_search,
                                          "best_program": best.to_dict(), "best_program_text": best.text(), "best_mass": mass,
                                          "checkpoint_verified": True, "wall_seconds": time.perf_counter() - task_start,
                                          "cpu_seconds": time.process_time() - task_cpu}
                                runs.write(canonical(record).decode() + "\n")
                                runs.flush()
                                connection.commit()
                                count += 1
                                turns += config["steps"]
                            if count % 30 == 0 or count == total:
                                print(f"discovery: {count}/{total} tasks; {time.perf_counter() - start:.1f}s", flush=True)
    finally:
        connection.close()
    if count != total:
        raise AssertionError("missing discovery tasks")
    manifest.update(status="complete", task_runs=count, teaching_turns=turns, rust_proofs=proofs,
                    elapsed_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu_start,
                    peak_rss_bytes=peak_rss_bytes(), energy_joules=None)
    names = ("config.json", "worlds.json.gz", "runs.jsonl", "trace.jsonl.gz", "events.jsonl.gz", "memory.sqlite",
             "concepts.jsonl", "explanations.jsonl", "research.json", "environment.json")
    manifest["outputs"] = {name: file_hash(output / name) for name in names}
    write_json(output / "manifest.json", manifest)
    if _render:
        from .report import render
        render(output)
    return output
