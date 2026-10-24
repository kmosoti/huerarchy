# Active symbolic learning study

The study implements a bounded synthesis of established methods. It does not
claim a new inference algorithm or reproduce the following papers' benchmarks.

* [Prediction-Oriented Bayesian Active Learning](https://proceedings.mlr.press/v206/bickfordsmith23a.html),
  Bickford Smith et al., AISTATS 2023: expected predictive information gain.
  Our reference distribution is uniform over permitted teaching inputs. No held-out
  long-path input is used in question selection. Conditional labels are independent
  given a finite program/noise state; the random-outcome model has zero conditional
  information about its independent future draws.
* [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742),
  Adams and MacKay, 2007: posterior uncertainty over regime duration. Our implementation
  bounds computation by retaining 32 branches, always including the newest branch.
  The change prior is 0.01 per observation and is independent of private schedules.
  Retaining earlier programs is a separate archive experiment.
* [DreamCoder](https://www.pldi21.org/poster_pldi.355.html), Ellis et al., PLDI 2021:
  reusable symbolic components and learned search. We implement typed template
  mining and symbolic retrieval; there is no neural search model or dreaming loop.
* [Learning MDL Logic Programs from Noisy Data](https://ojs.aaai.org/index.php/AAAI/article/view/28925),
  Hocquette et al., AAAI 2024: noise-aware evaluation of symbolic programs.
  Our atom-count penalty is an MDL-inspired proxy measured in nats, not an actual
  prefix code or measured compressed-byte length. No individual noisy error causes
  logical elimination of a hypothesis.
* [Building Intelligent Agents with Neuro-Symbolic Concepts](https://arxiv.org/abs/2505.06191),
  Mao, Tenenbaum, and Wu: typed, compositional concepts. The present implementation
  establishes symbolic definitions and provenance without neural representations.

## Preregistered interpretation

The primary goals are 20% lower capped recovery cost against historical broad-upfront,
without more than two percentage points lost on stable accuracy; random-control
Brier error at most 0.27 and >=90% confidence coverage at most 10% after 32 answers;
and 20% lower transfer acquisition or search costs without degraded accuracy or
probability error. These are research targets, never conditions for silently tuning
on evaluation results. All compared variants, failures, censored recoveries, and
budget stops must be reported.

For the executable target assessment, transfer costs include source and validation
acquisition, and reliability permits at most 0.02 absolute accuracy loss, 0.01 Brier
increase, and 0.02 nats log-loss increase. These margins are fixed before the study.
Passing an in-sample target is distinguished from a supported generalization claim.

Finite searched banks, the noise grid, the stochastic candidate, supplied recursion
productions, and typed matching are inductive biases. A high posterior probability
does not establish correctness outside this model class. Randomness and an inadequate
explanation language are not always identifiable from the available experiments.
Closure proofs establish consequences of a candidate definition, not that the
definition describes an external world. Concurrent worker RSS is not inferred by
summing per-process lifetime peaks; energy is unmeasured.
