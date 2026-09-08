# Provenance notes for AELLO-C8-S11

Derived from this slot's seed/contract.yaml record. The compiled checker set and the corpus layout
below were authored for this slot; they are not the batch template.

## Corpus

vendored activation traces from the C8-S1 CIFAR-100 reference at pinned seeds.

Three traces, one per graded block shape, captured at seeds 20260814, 20260815 and 20260816 --
continuing the C8-S1 shard seed so the lineage is readable from the seed alone. The traces are the
input distribution the kernel is timed on, and they are the reason the graded quantity is a
measurement rather than a prediction: a depthwise-separable block timed on synthetic zeros or on one
re-used draw reports its cache residency, not its kernel.

## Corpus status: NOT LANDED

`environment/data/` and `tests/heldout/` carry no bytes in this tree. `seed/build/land_corpora.py`
has not run for this slot. Every path, shape, seed and bound the compiled checks read is therefore
authored in `solution/grounding.yaml` under `corpus_layout`, and stated as authored rather than as
inspected. Two consequences, both recorded rather than absorbed:

- No check here was written against bytes anyone has read. Each one that needs a file it cannot find
  reports that it could not find it; none passes over an absent corpus.
- `graded_block_shapes` and `pinned_trace_seeds` are the roster the landed archive must match. A
  disagreement at Phase 1 is a corpus-landing failure to repair, not a licence to re-pin the graded
  set after seeing which shape a kernel wins on.

## What is authored and what is measured

Authored at Phase 0: the three pinned block shapes, the pinned trace seeds, the fidelity envelope
(2e-3 absolute, 5e-3 relative against the vendored fp32 reference), the measurement methodology
parameters -- warmup floor 25, repetition floor 100, the admitted statistic and clock-source rosters,
the 5 percent stage-sum tolerance -- and the twelve compiled checks. These are published task
parameters: instruction.md states every one of them to the agent, they bound HOW a latency may be
obtained rather than what score it earns, and none of them is a ramp constant. The envelope's
calibration against the landed reference outputs is owed at Phase 1; if the landed reference makes it
vacuous or unclearable it is re-authored there and the change is recorded.

Measured at Phase 1/2 and deliberately null here: `floor`, `knee`, `dichotomizing_threshold`,
`reward_gate_pass_threshold`, and `graded_axis`. The consequence is stated rather than hidden --
this bundle is COMPLETE but NOT YET GRADEABLE, and
`tests/test_output.py::test_envelope_gated_speedup_ramp` raises `ConstantUnmeasured` as its first
statement, before it reads anything, until the measurement wave writes those constants. That
ordering is load-bearing: a check that read the submission first would let a missing-file error mask
the unmeasured signal, and the verifier would then report `gradeable=true` with `r_det=0.0` --
asserting the agent scored zero where the honest claim is that the slot cannot yet be scored.

## The split that keeps the envelope a gate

The activation traces ship agent-visible; the vendored fp32 reference OUTPUTS and the reference
kernel's measured latency ship only under `tests/heldout/`. The reason is specific to a kernel task
rather than a general secrecy habit: if the correct outputs were visible, the fastest conforming
submission would be a table lookup with no kernel in it, and the fidelity envelope would gate
nothing. The agent checks its own fidelity by running the shipped baseline on the same trace, which
is exactly what the vendored reference is.

## Measurement tier

`gpu` -- grades a GPU kernel's measured latency. Grading itself is CPU-only: the verifier reads the
harness's recorded per-shape wall-clock and never re-runs a kernel, which is why `[verifier.environment]`
binds `gpus = 0`.

## Supersession

`supersedes: null`. This slot has no predecessor generation; it is authored new in this batch.
