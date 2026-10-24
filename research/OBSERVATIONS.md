# Online symbolic learning: observations from 2026-10-24

The implementation learns executable rules from feedback, tracks competing
explanations, selects questions, detects possible changes, and tests reusable
concepts. This study does not establish an overall advantage over the existing
learner. Keeping uncertainty helped one failure mode while learning and adaptation
became worse on several chain tasks.

## What was implemented

The research project combines a Rust positive Horn closure engine with Python
learning and experiment components. The closure engine calculates consequences
of a supplied rule set; evidence-based learning determines how much to trust the
candidate rules. A finite Bayesian mixture tracks programs and feedback-noise
rates. EIG/EPIG policies select questions, change-point inference tracks possible
rule changes, and a typed template library proposes reusable relational concepts.
SQLite checkpoints, event logs, proof records, and SHA-256 manifests preserve the
experiment state.

This is online symbolic learning in bounded candidate languages. There is no
neural weight training or unrestricted natural-language teaching in this suite.
Earlier storage, retrieval, learning, and discovery experiments are preserved in
the same project with separate reports and configurations.

## Experiment and verification

- 24 complete world blocks: eight core and sixteen transfer; no partial blocks.
- 12,954 task runs and verified checkpoints.
- 2,990,592 newly acquired teaching answers; 3,008,640 processed answers including
  validation reuse.
- Approximately 4 hours 6 minutes elapsed study time within the six-hour ceiling.
- Original correctness checks: 70 Python tests and eight Rust tests passed.
- All checkpoints round-tripped during execution. Final evidence checks verified
  artifact hashes, database integrity/counts, frozen experimental sources, and
  representative restored predictions and next-question selections. There are
  182 conditional proof records; sampled Python/Rust closure results agree.

Worlds are small generated environments whose true rules are known to the
evaluator. They test simple mechanisms, pairs, and signal relays, with stable
rules, successive changes, and returns to an earlier rule. Feedback noise is
0%, 10%, or 20%. Controls include random outcomes and rules outside the supported
language. Learners cannot see evaluation inputs or change times. Long relay
evaluations use chains beyond the teaching examples. Paired intervals resample
whole world blocks; repeated seeds are not treated as independent worlds.

## Findings

| Question | Existing baseline | New EPIG/change learner | Interpretation |
| --- | --- | --- | --- |
| At least 90% confidence on random outcomes, after 32 answers | 83.0% of predictions | 6.7% | Much less unsupported confidence |
| Probability error on random outcomes (Brier; lower is better) | 0.460 | 0.301 | Improved, but fair-coin optimum is 0.250 and the predeclared target was at most 0.270 |
| Recoveries after successive new relay rules | 111/144 | 89/144 | New combination adapted less successfully |
| Capped answers to recovery, successive relay rules | 66.1 | 69.4 | New combination was slower; failures retain the available horizon |
| Recoveries when a previous relay rule returned | 97/144 | 83/144 | Change inference did not deliver the expected recovery benefit |
| Stable relay accuracy averaged over learning | 92.4% | 83.2% | New learner learned more slowly; final accuracy was 100% versus 97.4% |

Other variants matter: sequential and eight-answer gates met the random-outcome
calibration target but learned relays poorly. A shorter-memory version of the old
baseline recovered faster on changing relays, with some loss during stable
learning. The new learner improved mean accuracy on changing pair tasks. No
variant should be called a universal winner from these experiments.

For renamed pair transfer tasks, provisional typed concepts reduced capped
target-only teaching cost from 23.2 to 16.7 answers. Including source learning
and validation changed total cost from 871.2 to 864.7: approximately 0.75% saved.
The paired whole-world 95% bootstrap interval for the difference was -9.72 to
-3.59 answers. Search cost and CPU time did not improve on that pair comparison.

On relay transfer, provisional concepts improved mean accuracy from 71.5% to
74.8%, but the interval for that difference included zero. Total teaching cost
fell about 1.35%, also with an interval including zero. Search work fell about
1.31%; measured CPU time increased. This is tentative transfer evidence.

The library produced 422 concept records across repeated variants and runs,
not 422 globally distinct concepts. None passed the two-task predictive admission
gate. The validated library consequently matched the no-library predictions and
learning cost. Exact matching also failed to reuse concepts across renamed
predicates. Provisional typed imports appeared in 223 of 288 target programs.

## Limits and next hypotheses

These findings concern bounded synthetic worlds, finite rule grammars, and this
implementation/configuration. Correct software does not guarantee useful
learning. The tests do not establish why the new relay learner was worse or why
no concept passed admission.

A post-hoc trace analysis found that EPIG did not avoid the largest permitted
teaching graphs. It repeated more questions early than EIG, but that observation
does not establish a cause. The analysis is labeled post-hoc in the report and
did not change the learning configuration.

Useful next experiments would isolate question selection from belief updating,
compare change handling with the shorter-memory baseline, and test whether
concept acquisition pays off over many more independent target tasks. Any tuning
should use development worlds followed by a fresh held-out study. Do not silently
replace the recorded negative results with a tuned rerun.

## Evidence

- [Interactive browser report](inference-lab/results/active/default/REPORT.html)
  and [analysis JSON](inference-lab/results/active/default/analysis.json).
- [Summary](inference-lab/results/active/default/summary.json),
  [noise breakdown](inference-lab/results/active/default/noise-breakdown.csv),
  [selection record](inference-lab/results/active/default/selection.json), and
  [verification receipt](inference-lab/results/active/default/verification.json).
- [Frozen study configuration](inference-lab/configs/active/study.json),
  [research references](inference-lab/evidence/active-research.md), and
  [archive retrieval and verification](ARTIFACTS.md).
