# Provenance notes for AELLO-C8-S7

Derived from this slot's seed/contract.yaml record. The compiled checker set and the corpus layout
below were authored for this slot; they are not the batch template.

## Corpus

Google Speech Commands v0.02 (Warden 2018), 35 keywords.

One second of 16 kHz audio per utterance. The graded split is subsampled to a 1.00:1 balance -- 100
utterances for each of the 35 keywords, 3500 in all -- rather than reweighted at scoring time, so
the metric the instruction names is plain top-1 accuracy and needs no per-class weight to be
honest. The released corpus is not balanced: the ten command words carry roughly four times the
clips of the auxiliary words, and grading unbalanced accuracy on it would reward a prior.

## Corpus status: NOT LANDED

`environment/data/` and `tests/heldout/` carry no bytes in this tree. `seed/build/land_corpora.py`
has not run for this slot. Every path, roster, count and bound the compiled checks read is
therefore authored in `solution/grounding.yaml` under `corpus_layout`, and stated as authored
rather than as inspected. Two consequences, both recorded rather than absorbed:

- No check here was written against bytes anyone has read. Each one that needs a file it cannot
  find reports that it could not find it; none passes over an absent corpus.
- `keyword_roster`, `graded_utterance_count` and `utterances_per_keyword` are the layout the landed
  archives must match. A disagreement at Phase 1 is a corpus-landing failure to repair, not a
  licence to re-pin the graded split after seeing which utterances a model gets right.

## What is authored and what is measured

Authored at Phase 0: the 35-word roster and its lexicographic order, the balanced graded count, the
one-second utterance length, the 20.0 ms per-utterance latency bound and its maximum statistic, the
admitted and rejected latency-source rosters, the declarable front-end envelope, the 0.25 share cap,
the six vendored background clips and the 0-20 dB mixing range, and the fifteen compiled checks.
These are published task parameters: instruction.md states every one of them to the agent, they
bound HOW an inference may be obtained and measured rather than what score it earns, and none of
them is a ramp constant. The latency bound in particular is a specification and not a measurement --
`CTL-OVERBUDGET` cannot target a model that exceeds a bound that does not yet exist. Its calibration
against the pinned verifier class is owed at Phase 1: if the shipped baseline cannot meet it there,
the bound is repaired at landing and the repair is recorded, because repairing it after seeing agent
scores would be choosing the gate to fit the results.

Measured at Phase 1/2 and deliberately null here: `floor`, `knee`, `dichotomizing_threshold`,
`reward_gate_pass_threshold`, and `graded_axis`. The consequence is stated rather than hidden --
this bundle is COMPLETE but NOT YET GRADEABLE, and
`tests/test_output.py::test_latency_gated_keyword_accuracy_ramp` calls `require_measured` as its
first statement, before it reads anything, until the measurement wave writes those constants. That
ordering is load-bearing: a check that read the submission first would let a missing-file error mask
the unmeasured signal, and the verifier would then report `gradeable=true` with `r_det=0.0` --
asserting the agent scored zero where the honest claim is that the slot cannot yet be scored.

## The split that keeps both measurements honest

The graded utterances ship agent-visible and unlabelled; their keyword labels ship only under
`tests/heldout/`. The asymmetry is forced by the two objectives rather than chosen for secrecy:
latency is a property of running the model over the graded audio, so the audio cannot be hidden, and
accuracy is a property of the labels, so the labels cannot be shown. The same coupling is why the
front-end digest is recorded on both the timed pass and the scored pass and checked for equality --
a latency measured on one configuration and an accuracy measured on another are two measurements of
two models, and nothing downstream could tell.

## Measurement tier

`gpu` -- trains a neural net over an audio corpus. Grading itself is CPU-only: the verifier reads
the harness's recorded per-utterance wall-clock and the delivered label vector and re-runs no model,
which is why `[verifier.environment]` binds `gpus = 0`. The per-utterance bound is stated against
that CPU class, not against the H100 the agent trains on.

## Supersession

`supersedes: null`. This slot has no predecessor generation; it is authored new in this batch.
