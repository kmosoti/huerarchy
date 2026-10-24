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
from . import components
from .types import LearningEvent, create
from .worlds import configuration, make_worlds, seed_for, truth_at


def make_learner(config, name):
    kwargs = {"dimensions": config["dimensions"], "noise": config["assumed_noise"], "window": config["window"]}
    if name == "drift-reset":
        kwargs["detector_config"] = config["detector"]
        kwargs["detector_component"] = config["components"]["detector"]
    return create("learner", name, **kwargs)


def recovery_delays(accuracy, changes, horizon, threshold=0.9, consecutive=10):
    result = []
    for change, end in zip(changes, changes[1:] + [horizon]):
        recovered = None
        for offset in range(change, end - consecutive + 1):
            if min(accuracy[offset:offset + consecutive]) >= threshold:
                # Report when recovery is confirmed, not the first point of a future window.
                recovered = offset + consecutive - change
                break
        result.append({"after_question": change, "confirmed_recovery_delay": recovered,
                       "censoring_horizon": end - change})
    return result


def verify_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("status") != "complete":
        raise ValueError("learning experiment is incomplete")
    required = {"config.json", "worlds.json", "runs.jsonl", "trace.jsonl.gz", "memory.sqlite", "explanations.jsonl", "environment.json"}
    if not required.issubset(manifest["outputs"]):
        raise ValueError("missing required learning artifacts")
    for filename, expected in manifest["outputs"].items():
        if Path(filename).name != filename or file_hash(directory / filename) != expected:
            raise ValueError(f"learning artifact changed: {filename}")
    return manifest


def run(overrides=None, output=None):
    config = configuration(overrides)
    config_sha = digest(config)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = ROOT / "results/learning" / f"{stamp}-{config_sha[:8]}"
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; completed experiments are preserved")
    use_rust = "rust-horn" in config["components"]["reasoners"]
    if use_rust and not (ROOT / "target/release/bench").exists():
        raise ValueError("build the existing Rust reasoner first: cargo build --release")
    output.mkdir(parents=True, exist_ok=True)
    worlds = make_worlds(config)
    write_json(output / "config.json", config)
    write_json(output / "worlds.json", {"schema": 1, "worlds": worlds,
               "split_policy": "Random halves, resampled until both outcomes occur in both halves in every phase; labels are evaluator-only"})
    write_json(output / "environment.json", profile_environment())
    sources = sorted((ROOT / "research_lab/learning").glob("*.py")) + [ROOT / "research_lab/__main__.py", ROOT / "research_lab/common.py"]
    sources += sorted((ROOT / "tests").glob("test_learning*.py"))
    manifest = {"schema": 1, "status": "running", "config_sha256": config_sha,
                "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                "rust_reasoner_sha256": file_hash(ROOT / "target/release/bench") if use_rust else None,
                "components": "built-in-registry-v1", "outputs": {}}
    write_json(output / "manifest.json", manifest)
    connection = sqlite3.connect(output / "memory.sqlite")
    evaluator = create("evaluator", config["components"]["evaluator"])
    run_count = 0
    proof_count = 0
    began = time.perf_counter()
    cpu_began = time.process_time()
    total_runs = len(worlds) * len(config["noise_rates"]) * len(config["seeds"]) * len(config["variants"])
    try:
        with (output / "runs.jsonl").open("w") as metrics_stream, \
                gzip.open(output / "trace.jsonl.gz", "wt", encoding="utf-8") as trace_stream, \
                (output / "explanations.jsonl").open("w") as evidence_stream:
            for world in worlds:
                for noise in config["noise_rates"]:
                    for seed in config["seeds"]:
                        for variant in config["variants"]:
                            run_id = f"{world['id']}/noise-{noise:g}/seed-{seed}/{variant['learner']}/{variant['policy']}"
                            learner = make_learner(config, variant["learner"])
                            policy_seed = seed_for("questions", world["id"], seed)
                            policy = create("policy", variant["policy"], seed=policy_seed, exploration=config["exploration"])
                            teacher = create("teacher", config["components"]["teacher"], world=world, dimensions=config["dimensions"], noise=noise,
                                             seed=seed_for("noise", world["id"], noise, seed), steps=config["steps"])
                            memory = create("memory", config["components"]["memory"], connection=connection, run_id=run_id)
                            pool = tuple(world["question_pool"])
                            held_out = np.array(world["held_out"])
                            accuracy, briers, losses, coverage, pre_briers, alarms = [], [], [], [], [], []
                            confident_errors = confident_count = flip_count = 0
                            prequential_observed_loss = 0.0
                            unique_questions = set()
                            learn_ns = policy_ns = 0
                            start, cpu_start = time.perf_counter(), time.process_time()
                            for sequence in range(1, config["steps"] + 1):
                                clean = truth_at(world, sequence, config["dimensions"])
                                probabilities = learner.probabilities()
                                pre_briers.append(float(np.mean((probabilities[held_out] - clean[held_out]) ** 2)))
                                tick = time.perf_counter_ns()
                                question = policy.choose(probabilities, pool, config["assumed_noise"])
                                policy_ns += time.perf_counter_ns() - tick
                                observation = teacher.answer(question.features)
                                p = float(probabilities[question.features])
                                q = config["assumed_noise"] + (1 - 2 * config["assumed_noise"]) * p
                                observed_loss = -math.log(q if observation.label else 1 - q)
                                prequential_observed_loss += observed_loss
                                flip_count += observation.label != bool(clean[question.features])
                                unique_questions.add(question.features)
                                tick = time.perf_counter_ns()
                                update = learner.observe(observation)
                                post = learner.probabilities()
                                learn_ns += time.perf_counter_ns() - tick
                                evaluated = evaluator.evaluate(post[held_out], clean[held_out])
                                accuracy.append(evaluated["balanced_accuracy"])
                                briers.append(evaluated["brier"])
                                losses.append(evaluated["log_loss"])
                                coverage.append(evaluated["confident_coverage"])
                                confident_errors += evaluated["confident_errors"]
                                confident_count += evaluated["confident_count"]
                                if update["reset"]:
                                    alarms.append(sequence)
                                # Learner memory contains observations, never clean labels or phase boundaries.
                                memory.append(LearningEvent(sequence, "teaching", {
                                    "observation": asdict(observation), "question": asdict(question), "update": update}))
                                trace = {"run_id": run_id, "sequence": sequence, "question": asdict(question),
                                         "observation": asdict(observation), "update": update,
                                         "pre_update_question_probability": p,
                                         "pre_update_observed_log_loss": observed_loss,
                                         "pre_update_heldout_brier": pre_briers[-1], **evaluated}
                                trace_stream.write(canonical(trace).decode() + "\n")
                                if (proof_count < 12 and world is worlds[0]
                                        and noise == config["noise_rates"][0] and seed == config["seeds"][0]
                                        and sequence in {1, config["steps"] // 2, config["steps"]}):
                                    candidate = learner.explanation_candidate()
                                    if candidate is None:
                                        continue
                                    hypothesis, mass = candidate
                                    explanations = {name: create("reasoner", name).explain(hypothesis, question.features, config["dimensions"])
                                                    for name in config["components"]["reasoners"]}
                                    if len({e["outcome"] for e in explanations.values()}) != 1:
                                        raise AssertionError("reasoner plugin disagreement")
                                    evidence_stream.write(canonical({"run_id": run_id, "sequence": sequence,
                                        "posterior_mass": mass,
                                        "candidate": asdict(hypothesis), "reasoners": explanations}).decode() + "\n")
                                    proof_count += 1
                            state = {"schema": 1, "config_sha256": config_sha, "sequence": config["steps"],
                                     "teacher_schedule_sha256": teacher.schedule_hash(),
                                     "learner": learner.snapshot(), "policy": policy.snapshot()}
                            memory.checkpoint(state)
                            saved = memory.restore()
                            restored = make_learner(config, variant["learner"])
                            restored.restore(saved["learner"])
                            restored_policy = create("policy", variant["policy"], seed=policy_seed, exploration=config["exploration"])
                            restored_policy.restore(saved["policy"])
                            if not np.array_equal(restored.probabilities(), learner.probabilities()):
                                raise AssertionError("checkpoint changed predictions")
                            if restored_policy.choose(restored.probabilities(), pool, config["assumed_noise"]) != policy.choose(
                                    learner.probabilities(), pool, config["assumed_noise"]):
                                raise AssertionError("checkpoint changed next question")
                            elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu_start
                            connection.commit()
                            record = {"run_id": run_id, "world": world["id"], "world_kind": world["kind"],
                                      "noise": noise, "seed": seed, **variant, "steps": config["steps"],
                                      "balanced_accuracy_auc": float(np.mean(accuracy)),
                                      "final_balanced_accuracy": float(np.mean(accuracy[-20:])),
                                      "heldout_brier": float(np.mean(briers)), "heldout_log_loss": float(np.mean(losses)),
                                      "pre_update_heldout_brier": float(np.mean(pre_briers)),
                                      "confident_coverage": float(np.mean(coverage)),
                                      "confident_error_rate": confident_errors / confident_count if confident_count else None,
                                      "observed_prequential_log_loss": prequential_observed_loss / config["steps"],
                                      "noise_flips": int(flip_count), "noise_schedule_sha256": teacher.schedule_hash(),
                                      "unique_questions": len(unique_questions), "alarms": alarms,
                                      "recovery": recovery_delays(accuracy, world["changes"], config["steps"]),
                                      "wall_seconds": elapsed, "cpu_seconds": cpu_elapsed,
                                      "learner_ns": learn_ns, "policy_ns": policy_ns,
                                      "checkpoint_verified": True}
                            metrics_stream.write(canonical(record).decode() + "\n")
                            run_count += 1
                        if run_count % 40 == 0 or run_count == total_runs:
                            print(f"learning: {run_count}/{total_runs} runs; {time.perf_counter() - began:.1f}s", flush=True)
    finally:
        connection.close()
    if run_count != total_runs:
        raise AssertionError("incomplete experiment")
    manifest.update(status="complete", runs=run_count, teaching_turns=run_count * config["steps"],
                    proof_checks=proof_count, elapsed_seconds=time.perf_counter() - began,
                    cpu_seconds=time.process_time() - cpu_began, peak_rss_bytes=peak_rss_bytes(),
                    energy_joules=None, energy_note="No exposed energy sensor; CPU time is not energy")
    manifest["outputs"] = {name: file_hash(output / name) for name in (
        "config.json", "worlds.json", "runs.jsonl", "trace.jsonl.gz", "memory.sqlite", "explanations.jsonl", "environment.json")}
    write_json(output / "manifest.json", manifest)
    from .report import render
    render(output)
    return output
