# Research basis for the discovery experiment

Researched 24 October 2026. These are ingredients for a new controlled comparison,
not claims that this project invented the underlying algorithms or reproduced
their published results. The next experiment uses no pretrained language model.

| Research | Useful idea | Application and limit |
|---|---|---|
| [Learning MDL Logic Programs from Noisy Data, AAAI 2024](https://www.cs.helsinki.fi/u/mjarvisa/papers/hnjc.aaai24.pdf) | Balance a program's predictive fit against its size when examples can be wrong. | Use a declared size penalty during bounded search. Our fresh-response confirmation gate is an experimental design choice, not an optimality guarantee. |
| [Stitch: Top-Down Synthesis for Library Learning, POPL 2023](https://arxiv.org/abs/2211.16605) | Extract recurring program structure into reusable definitions. | Mine shared relational bodies and pay for the definition before accepting compression. This simpler miner does not implement Stitch's full algorithm. |
| [Hypothesis Generation and Inductive Inference in Children and Language Models, May 2026](https://arxiv.org/abs/2605.24528) | Combine uncertainty about evidence with generating hypotheses during interaction. | Test generated programs rather than only an initial fixed list. Findings from the paper's box task do not establish a general cognitive mechanism. |
| [Learning Progress Monitoring, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/803b9c4a8e4784072fdd791c54d614e2-Abstract-Conference.html) | Reward improvements in prediction to avoid fixation on irreducible randomness. | Compare grouped prequential loss improvements with random and information-seeking questions. This is inspired by LPM, not a reproduction of its dual-network method or theoretical guarantees. |
| [DiscoveryWorld, NeurIPS 2024](https://allenai.github.io/discoveryworld/) | Evaluate discovery as a cycle of explanations, experiments, and new knowledge. | Build small diagnostic worlds to isolate failures. Scores here are not DiscoveryWorld benchmark scores. |
| [Alchemy](https://github.com/google-deepmind/dm_alchemy) | Procedurally change hidden causal structure and inspect how agents investigate it. | An external benchmark adapter remains a later component; no Unity or Alchemy dependency is added. |

## The experimental proposal

Failures have competing explanations: an earlier lesson may be wrong, the world
may have changed, or the learner's rule language may be inadequate. A larger
program should compete with a reset and with remaining uncertain. All candidates
must face new teaching responses, and labels used to construct a proposal cannot
also count as its confirmation evidence.

Concept compression and expressive expansion are different. Replacing three
conditions with a named helper preserves an existing program's meaning. A new
alternative rule or a recursive definition can express a different function.
The report measures both, and tests whether compressed helpers help later tasks.

The mechanism workshop tests new combinations of conditions. The pair workshop
tests relationships with shared variables. The relay workshop tests composing
those relationships repeatedly through a finite graph. Positive recursion has
a finite fixed point; closure proofs explain consequences conditional on a
candidate program, not whether that program is a true description of the world.

## Deferred alternatives

[Formal Concept Analysis](https://iccl.inf.tu-dresden.de/web/Formale_Begriffsanalyse/en)
offers a principled route from object–attribute tables to concepts and their
hierarchy. An alternative proposer could use closed patterns, but exact observed
implications must not be treated as causal laws under noisy feedback.

[ADVENT, July 2026](https://arxiv.org/abs/2607.01585) combines language-model
predicate proposals with deductive execution checks and a reusable knowledge
pool. [UniPred, revised September 2026](https://arxiv.org/abs/2512.17992) combines
foundation-model suggestions with predicates learned from low-level data. Both
are relevant future proposal sources; they introduce pretrained knowledge and
additional compute. They are excluded from this learner-generated baseline.

Hidden-property discovery, action-sequence learning, raw visual perception,
natural-language teaching, and neural weight updates are separate later steps.
