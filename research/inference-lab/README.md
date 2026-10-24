# Compact state research pilot

A reproducible CPU experiment combining exact event state, positive Horn closure,
native ternary embeddings, and versioned state deltas. Original records and their
provenance remain authoritative; retrieval cannot authorize an action.

Open [REPORT.html](REPORT.html) for measured findings, methodology and limits.
[evidence/claim-audit.csv](evidence/claim-audit.csv) audits 32 research claims.

## Reproduce

The inspected environment has a four-core CPU quota, 16 GiB RAM, network access,
GCC/G++, Python 3.12 and `uv`. No GPU, driver change, paid API, or external inference
service is required. Setup downloads a pinned 367 MB model and builds two native
CPU runtimes. Outbound requests use the environment's inherited proxy and trust.

```bash
cd /workspace/inference-lab
bash scripts/setup.sh
bash scripts/run.sh
```

On another workspace, keep this project next to a writable `scratch/` directory.
Setup selects local Rust tools without modifying your shell profile. Upstream
BitNet and its llama.cpp submodule are pinned; the GGUF artifact is SHA-256 checked.
The runtime build requires npm for the source-pinned server UI dependency; this
cloud environment supplies it. Prebuilt, floating UI downloads are disabled.

Use individual stages after setup:

```bash
source scripts/env.sh
python -m research_lab profile
python -m research_lab generate --events 100000 --nodes 5000 --queries 10000 --seed 42
python -m research_lab bench --variant exact --repeats 5
python -m research_lab bench --variant index-build --repeats 5
python -m research_lab bench --variant closure --repeats 5
python -m research_lab bench --variant communication --repeats 5
python -m research_lab bench --variant retrieval --repeats 5
python -m research_lab bench --variant instruction-set --repeats 5
python -m research_lab bench --variant vector-kernels --repeats 5
python -m research_lab audit
python -m research_lab report
python scripts/package.py
```

Run native builds before timing; avoid concurrent build/benchmark processes.
The embedding server is temporary, listens only on loopback and is stopped after
each stage. Corpus/query embeddings are cached with model, source and fixture
fingerprints; five warm encoding repetitions are always measured afresh.

## Workload and semantics

The default fixture contains 100,000 events, 5,000 nodes and 100 snapshots. It has
2,500 queries each for exact reconstruction, bounded multi-hop reachability,
held-out paraphrase retrieval and simulated restart decisions. Replica groups
contain five nodes and require three healthy replicas after a restart; any unknown
group health gives an unknown decision. No real actions are executed.

The cyclic dependency graph has bounded 25-node components. Rules propagate
impact from unhealthy nodes. Counts, absence and quorum policy are evaluated
against the authoritative snapshot outside the monotone Horn engine. Changed or
removed source facts are handled by rebuilding each snapshot's closure.

Rust exposes `Program::new(rules).derive(seeds, proofs)` as a fact iterator and
`finish()` for a complete outcome. Initialization eagerly consumes seeds; rule
propagation is lazy. Evidence records one first-discovery rule per fact. The native
`bench infer PROGRAM` interface accepts lines `s FACT_ID` and
`r CONCLUSION PREMISE...`, then returns fact IDs, proof references and counters.

Ground-rule delta rounds and relational semi-naive joins are separate experiments.
There is no claim that the ground evaluator implements general Datalog.

## Measurements and guarantees

Exact checks use independent event replay, graph search and decision references.
Tests check closure axioms, duplicate/circular/unconditional rules, proof ordering,
early iterator termination, stale facts, codec round trips, atomic failure,
unknown IDs, context mismatch, out-of-order delivery and replay.

SQLite FTS5, FP32-vector, INT8-storage and reciprocal-rank-fusion retrieval use the
same documents and relevance labels. INT8 is **output vector storage**, separate
from model-weight ternarization. The original NumPy comparison decodes it to FP32.
The follow-up Rust comparison keeps the INT8 corpus packed and converts eight
values at a time in AVX2 registers, with a scalar fallback. It uses the same top-50
heap and one CPU thread for both Rust FP32 and INT8. Search latency excludes query
encoding, which is measured separately with batched requests.

Run `vector-kernels` after `retrieval` to reuse its verified embedding cache; it
does not rerun the model. Each worker reads a little-endian `VEC1` matrix: 4-byte
magic, dtype (`u32`: 0=FP32, 1=INT8), row count (`u32`), column count (`u32`), then
row-major values. INT8 codes use signed two's-complement bytes. Row scales cancel
for cosine; inverse norms require four additional bytes per corpus row.

The native benchmark checks 32 sampled queries against an FP64 score/top-k oracle,
records all per-query rankings, and measures each worker's RSS independently from
the Python controller and model. Both native paths receive the same FP32 queries.
Scheduled worker CPU time includes search and result formatting/output; elapsed
search timing excludes output. Queries remain FP32, so this is mixed FP32/INT8
arithmetic rather than an integer-only dot product.

The search worker is also usable directly after the matrix files are generated:

```bash
target/release/vector-search data/kernel-corpus-int8.vec data/kernel-queries.vec 50
```

Communication counts application payloads for full JSON, full CBOR and CBOR deltas,
with and without zstd-3, including context bootstrap and recovery. All variants
share the same dictionary cost. This is exact replication of the current health
projection, not compression of the complete event history, and not a LAN test.
Tasks require matching context, epoch and state digest before decisions.

Energy is null with an explicit reason when no sensor is exposed. CPU seconds are
not substituted for joules. AVX2 measurements on this Xeon do not predict laptop
performance. No FP16 model baseline, GPU LoRA, full MoTE training or distributed
RPC result is claimed.

## Artifacts

- `results/*.jsonl`: raw per-repetition measurements and correctness checks.
- `results/summary.json`, `summary.csv`: report-ready results.
- `results/environment.json`, `build-manifest.json`, `model.json`: runtime pins,
  flags, model hashes and observed capabilities.
- `results/manifest.json`: hashes of outputs and experiment source files.
- `results/suite-inputs.json`: binds completed suites to the fixture and raw result
  hashes; report generation rejects stale or mixed results.
- `data/manifest.json`: deterministic fixture hashes and configuration.
- `evidence/claim-audit.csv`: source-linked claim assessments, distinct from
  replicated measurements.

Large generated data, models, the virtual environment and build artifacts are
excluded by `.gitignore`. The code has no Rust crate dependencies; Python package
versions are pinned in `requirements.lock`.

## Online learning and curiosity

The [measured default run](results/learning/default/REPORT.html) contains 1,560
runs and 374,400 teaching turns, with all final checkpoints verified.

The separate `learning` experiment tests how a learner acquires and revises rules
under noisy feedback and unannounced world changes. It runs on the CPU, without a
language model. Its supplied attributes and conjunction rule family are explicit
inductive biases; it does not begin without prior structure.

```bash
source scripts/env.sh
python -m research_lab learning components
python -m research_lab learning run --smoke --output results/learning/smoke
bash scripts/run-learning.sh --output results/learning/default
python -m research_lab learning report --run results/learning/default
```

Use a new or empty output directory for every experiment. The full default
configuration is in `configs/learning/default.json`; `learning config` prints the
same defaults. `--config PATH` accepts a complete configuration or top-level
overrides. Nested component and detector selections replace their respective
objects in full. Existing pilot fixtures, results, and benchmark commands are
independent of this experiment.

The default has six stable and six changing worlds, plus a separate XOR diagnostic
that the hypothesis family cannot express. Each world has six binary attributes,
32 available question examples, and 32 held-out examples. The evaluator resamples
the split until both outcomes occur in both halves in every phase. Changing worlds
switch rules after questions 80 and 160; each consecutive target differs on at
least 25% of the complete truth table. Those are deliberately detectable abrupt
changes, not a test of arbitrary or gradual drift.

At feedback-error rates 0%, 10%, and 20%, five seeds compare full-history,
32-observation-window, and drift-reset Bayesian learners with random or
information-seeking questions. Frozen-prior and exact-memory baselines complete
the eight conditions. Corresponding runs share teacher flip schedules and label
budgets; repeated questions receive independent flips. Learners assume 10% noise,
so actual error rates are never passed to them. Candidate hypotheses are all 730
signed conjunctions and constant outcomes. Their initial weights are proportional
to `2 ** (-condition_count)`.

The curiosity policy maximizes expected hypothesis information, using 10% random
exploration. Drift detection monitors pre-update noisy-response log loss with
Page–Hinkley (20 observations minimum, tolerance 0.05, threshold 5), resets to the
prior, and replays the latest eight observations. These settings are experimental
defaults, not claims of optimality. Probabilities describe beliefs under the
specified model and can be poorly calibrated when the model is wrong.

### Replaceable components

`research_lab.learning.types` defines protocols for teachers, experience memory,
rule learners, question policies, change detectors, reasoners, and evaluators.
Factories register by category and name with `@register(category, name,
description)`. A new implementation supplies that protocol, accepts the factory
arguments used by the harness, and is explicitly imported for registration.
Configurations select learner/policy pairs and the teacher, memory, detector,
reasoner list, and evaluator. There is no automatic discovery or execution of
untrusted plugin code.

The initial interface operates on complete binary feature assignments encoded as
integer bit masks. Observations carry a sequence, feature assignment, noisy label,
and source. Learners expose predictions, updates, serializable state, and an
optional candidate explanation. Question policies receive predictions and allowed
examples, never target rules. Expanding this representation or adding a language
bridge is a later component increment.

Rust's existing Horn reasoner remains unchanged. Negative feature values become
explicit positive atoms; absence is never treated as false. Proofs explain a
consequence **conditional on a candidate rule**, not the truth of that rule.

### Learning evidence

Each experiment produces a standalone `REPORT.html`, JSON/CSV summaries, per-run
metrics, a compressed per-turn trace, SQLite experience memory and checkpoints,
conditional proof examples, environment data, and a SHA-256 artifact manifest.
The SQLite teaching log contains observations, questions, and learning updates.
Hidden targets are in the evaluator's separate world fixture; clean evaluation
labels and phase boundaries are never fed back to learning components.

Scoring separates latent clean outcomes from pre-update prediction of noisy
teacher responses. Reports include held-out balanced accuracy, Brier/log scores,
confidence, adaptation and censored recovery, false alarms in stable worlds, and
world-level paired bootstrap intervals for question-policy comparisons. Raw runs
also retain unique question counts, policy/learner timing, and CPU time. Peak RSS
covers the controller; energy is unavailable. Every final checkpoint must
reproduce predictions and its next question before a run is accepted.

An XOR diagnostic exposes limited representational capacity. A reset can change
beliefs but cannot make one conjunction represent XOR. Results from that world
are reported separately. The experiment makes no claim of natural-language
understanding, discovered concepts, or neural weight adaptation.

## Discovery worlds and explanation expansion

The [discovery report](results/discovery/default/REPORT.html) adds mechanism,
pair, and relay workshops. It tests constructing richer explanations and reusing
relational helpers. The [research note](evidence/discovery-research.md) identifies
the primary sources, our design choices, and which proposed approaches remain
deferred.

```bash
source scripts/env.sh
python -m research_lab discovery components
python -m research_lab discovery run --smoke --output results/discovery/NEW_SMOKE
bash scripts/run-discovery.sh --output results/discovery/NEW_FULL_RUN
python -m research_lab discovery report --run results/discovery/NEW_FULL_RUN
python scripts/analyze-discovery.py results/discovery/NEW_FULL_RUN
python scripts/plot-discovery.py results/discovery/NEW_FULL_RUN
python scripts/replay-discovery.py results/discovery/NEW_FULL_RUN --run-id c0/relay/noise-0/seed-0/adaptive-library
```

Default settings are in `configs/discovery/default.json`. The full suite uses six
curricula, each containing one mechanism task, three pair tasks, and a relay task.
Half change their rules after responses 80 and 160. All tasks receive 240 teaching
responses, at error rates 0%, 10%, and 20%, across five seeds and six conditions.
Separate coin, mixed random/learnable, and unsupported parity controls are also
run: 2,970 task runs and 712,800 teaching responses in total. Three independent
CPU workers preserve the order of tasks within each curriculum; a test confirms
that worker count does not change non-timing results.

`research_lab.discovery.programs` defines public scenes, queries, feedback, rule
programs, and component protocols. Programs use up to four clauses, twelve body
conditions, three typed variables per clause, and two local helper predicates.
No object-identifier constants, generated objects, or arbitrary Python execution
are allowed. All recursion is positive over finite scenes of at most eight
objects. Negative observable values have explicit facts; absence is not evidence
of a negative world observation. An unproved model output is a conditional
negative prediction under that complete finite program.

The starting language contains a single conjunction. The compositional expander
constructs edits, alternatives, shared-variable joins, and positive recursion
without receiving the hidden target or world-family identifier. Each round scores
at most 2,000 returned candidates; 128 expanded candidates can be retained in
addition to the original starting bank. Strategy quotas reserve space for
alternatives and recursion. Candidate generation has an overall budget of one
million unique program–public-scene evaluations per task; exhausted budgets are
reported. Context counters do not count every scalar operation or posterior
update; CPU/wall timings include those additional costs.

The adaptive learner fits proposals using the latest 64 observations and starts
confirmation after at least 32. A challenger must improve prediction on 32 fresh
responses, exceeding its added size penalty plus `ln(10)`. A reset and an
uninformative predictor compete with growth. The penalty is `ln(2)` per added
head/body node. These settings are declared heuristics, not calibrated statistical
guarantees. The broad-upfront baseline can use rich productions immediately and
refreshes candidates directly; it is bounded search, not an exhaustive oracle.

Relational helpers require three occurrences across at least two tasks, with
positive description savings after paying for the definition. A library holds at
most eight immutable definitions. Each helper includes its source programs and
tasks. Available helpers and helpers actually imported by the final explanation
are reported separately. Every condition pays for the earlier curriculum lessons;
cumulative acquisition labels and search costs accompany later task scores.

Components register explicitly under `discovery-expander`, `discovery-library`,
`discovery-policy`, `discovery-teacher`, and `discovery-world`. The configuration's
`components` object selects those factories. New implementations follow the
public protocols/factory arguments and are explicitly imported for registration;
configurations do not load arbitrary code. Programs and helper definitions can be
inspected as structured data or readable clauses. Core learning and question
selection are independent of a language model.
The first world contract uses six named binary inputs or the shared object
measurement schema; other input vocabularies need an adapter or a new starting
language. Configurable components do not imply unrestricted representation learning.
The curriculum factory accepts `(index, config)` and returns evaluator-owned
tasks. The teacher factory accepts `(task, noise, seed, steps)` and exposes only
`answer()` to the teaching loop. Expanders accept an observed `ExpansionView`;
policies accept `(kind, seed, exploration)`. The public world protocol describes
the answering interface, while the curriculum factory constructs its private
test fixtures.

Results contain per-task metrics, compressed teaching traces and expansion events,
SQLite experience logs and restorable component checkpoints, mined definitions,
conditional Rust proof audits, and SHA-256 manifests. The NumPy evaluator, Python
reference closure, and existing Rust engine are checked independently. Report
comparisons resample whole curricula, rather than treating seeds and related pair
tasks as independent evidence. The few synthetic curricula limit generalization.
The optional figure-export step uses pinned Matplotlib to generate standalone SVG
and PNG charts and embeds them into the verified report. Re-rendering the report
can be followed by the same export command to restore those figure links.
The replay command reads the verified SQLite checkpoint and reconstructs the
learner's original library version, which can precede definitions mined at the
task boundary. It previews the next question without obtaining another label.
The analysis step adds per-run learning delays, censored recovery, and an explicit
discussion of negative results. These postprocessing scripts bind their own source
hashes separately from the source files that produced the training traces.
Alarm episodes are false-to-true detector transitions. The raw `alarms` metric
counts alarm-positive responses, which can be consecutive while a detector remains
above threshold; it is not a count of independent change events.

## Active probabilistic learning and concept transfer

The `active` suite extends the discovery experiment with a finite mixture over
symbolic programs and feedback-noise rates, an explicit random-outcome model,
prediction-oriented questions (EPIG), capped Bayesian change-point inference,
and typed relational templates. Existing discovery configurations and results
remain separate. No language model, embedding model, neural training, GPU, or
external vector database is part of this suite.

```bash
source scripts/env.sh
python -m research_lab active components
python -m research_lab active run --smoke --output results/active/smoke
bash scripts/run-active.sh --output results/active/default
python -m research_lab active report --run results/active/default
python scripts/analyze-active.py results/active/default
python scripts/check-active.py results/active/default
python -m research_lab active replay --run results/active/default --id RUN_ID
```

Choose a fresh `--output` path when rerunning either study command; existing study
directories are never overwritten. The report, analysis, check, and replay commands
operate on saved results without obtaining new teaching answers.

`configs/active/default.json` declares a six-hour study ceiling after correctness
checks: 30 minutes for a development-fixture timing pilot, 3.5 hours for core
comparisons, 1.5 hours for transfer, and 30 minutes for report generation. Three
worker processes each use one numerical-computation thread. Sample sizes are
chosen from 4, 8, 12, or 16 independent world blocks using pilot CPU timings only,
with a 20% reserve and an extra factor of two for transfer-shadow uncertainty.
If even the smallest planned size does not fit, it is attempted under the same
deadline; unfinished blocks are explicit and excluded from paired comparisons.
Fewer than eight independent worlds is labeled exploratory. The pilot uses a
different fixture seed and is never included in the study statistics.
The recorded full run uses `configs/active/study.json`, which subtracts 199 seconds
spent on a discarded development pilot from the six-hour ceiling. Its learning
settings match the default configuration.

Each core block contains mechanism, pair, and relay tasks with stable rules,
A→B→C changes, and A→B→A returns, plus random, mixed, unsupported-parity, and
noise-only-shift controls. Fixed feedback-noise conditions are 0%, 10%, and 20%,
with three response-randomness seeds. All compared conditions within a task use
the same hidden noise draws; active policies can ask different questions.
No learner receives evaluation inputs, private mechanisms, true noise rates, or
change times. The evaluator consumes a read-only prediction snapshot.

The starting bank and 128 expanded candidates are bounded. Programs retain the
original positive-rule semantics: at most four clauses, twelve body atoms, three
typed variables per clause, and two local helper predicates. Ternary weights and
matrix operations are unrelated to these symbolic atoms. Probability updates use
noise rates `[.01, .05, .10, .20, .35]` and a 5% initial random-outcome probability.
Candidate search retains the original fixed-10%-noise ranking heuristic so that
query and update comparisons share the same proposer. Fresh candidates are
provisional predictions, not accepted facts. Rebuilding a bank replays each
observation once and leaves historical pre-answer predictions unchanged.

The window updater uses 64 answers. The change-point updater has hazard 0.01 and
at most 32 run-length branches, including the newest branch; discarded branch
mass is recorded. A separate archive variant reserves sixteen expanded slots for
programs with at least 90% mass over eight successive answers. Neither the archive
nor change inference establishes that a rule is objectively correct. Pure random
outcomes and an inadequate hypothesis language can be observationally ambiguous.

EIG and EPIG integrate the same finite noisy-model mixture. EPIG's reference
distribution is uniform over the permitted teaching inputs, including repeated
independent answers when appropriate. Long-chain evaluation inputs are absent.
Both policies retain 10% random exploration. An independent gate comparison uses
the historical adaptive learner with fixed 32-answer, fixed 8-answer, and sequential
confirmation; likelihood, complexity, and the `ln(10)` acceptance margin are held
constant. Sequential gates expire after 64 fresh answers.

Transfer uses three shared source tasks (240 answers each), two validation tasks
(32 preparation plus 32 fresh answers each), and two target tasks (240 answers
each). Targets rename predicate symbols and recombine compatible links with new
endpoint conditions, or require relay composition. Source and validation queries
are shared uniform streams; target queries use EPIG with change inference.

Libraries compare no reuse, exact-body compression, typed-template compression,
and typed templates with predictive validation. Templates contain two to four
connected atoms, at least three uses over two tasks, and positive description
savings after charging for the definition. At most eight templates and eight
type-correct instantiations per template are retained. Predicate mappings are
searched from public types, never supplied by the oracle. Symbolic imports are
hygienically expanded and checked against the same language bounds before they
enter the learner. This conservative check can reject factorizations that would
require a larger language after expansion.

Each candidate in the validated library is tested by a shadow learner on the
same answers as a reference learner. Promotion needs positive fresh log-score
gain in each of two tasks and total gain above `ln(20)`. A shared answer is acquired
once; every shadow's CPU and search cost is still measured. Library definitions
are frozen before the target tasks. Source programs can be wrong; compression
and provisional reuse alone do not validate them.

The registry exposes `active-posterior`, `active-policy`, `active-expander`,
`active-library`, and `active-teacher`. `TaskView` contains only schema and teaching
scenes. Components provide probability/prediction updates, question selection,
proposals, or concept operations plus snapshot/restore as applicable. Components
are explicitly imported and selected by registered names, not dynamically loaded
arbitrary files. Each hypothesis executes independently; incompatible hypotheses'
facts are never unioned into a common proof.
The current replay adapter expects posterior replacements to preserve the finite
mixture checkpoint format; a different state representation needs its own restore
adapter. Policy, proposal, library, and teacher implementations follow their
registered component contracts.

The report includes all conditions, cluster-bootstrap contrasts, censored recovery
costs, confidence reliability, and transfer costs. Full traces, pre-answer scores,
decision events, private reproducibility fixtures, immutable source hashes,
conditional Rust proofs, and compressed SQLite checkpoints live in per-world
block directories. Replay verifies hashes and previews the next question without
acquiring an answer. Independent random seeds are repetitions within a world;
statistical intervals resample complete world blocks. Multiple reported contrasts
are descriptive, without a familywise significance guarantee.
`accuracy` averages held-out balanced accuracy over every teaching step;
`final_accuracy` averages the final twenty answers. Success requires at least
90% accuracy for ten consecutive answers. Unsuccessful attempts retain their
available horizon in capped learning and recovery costs.
The analysis command adds plain-language findings and recovery curves with censored
failures, a concept-admission breakdown, and `noise-breakdown.csv` for all conditions.
It also includes `relay-query-coverage.csv`, an explicitly post-hoc description of
teaching-graph sizes and distinct questions, added after inspecting the completed
core results. This diagnostic does not establish a cause or change the experiment.
Re-run it after regenerating the base report. The check command verifies
artifact hashes, SQLite integrity and counts, representative restored prediction
caches against fresh computation, conditional Rust closures, and local report links.
Both postprocessing scripts record their own source hashes separately from the
frozen experimental source files. Worker-local syntax/validation caches can be
warm across conditions; learned predictions and observations are never shared.
Search-evaluation counts and full CPU costs are reported separately.

Research sources and predeclared targets are in
[`evidence/active-research.md`](evidence/active-research.md). Research targets can
fail while the software remains correct; final-test results are not used to retune
the study. Energy is unmeasured; memory is reported as the largest process RSS,
not summed simultaneous memory.
