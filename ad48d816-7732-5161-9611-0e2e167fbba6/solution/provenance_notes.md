# Provenance notes for AELLO-C8-S13

Derived from this slot's seed/contract.yaml record; the compiled checker set and the deliverable
layout were authored against the landed corpus rather than generated from the contract alone.

## Corpus

CIFAR-100-C (Hendrycks & Dietterich 2019), 19 corruptions at 5 severities.

Landed by `seed/build/land_c8s13.py`. One fixed draw of 200 CIFAR-100 test images is corrupted
into every one of the 95 (corruption, severity) cells, so the 19000-row corrupted grid is 200
images seen 95 ways and the private label vector is 95 tilings of one 200-entry vector. Two
consequences are carried into the checker set: a 200-entry label leak is a complete leak, and the
macro average over cells weights all 19 corruptions equally at each severity by construction.

## The deliverable covers two folds

The graded artifact is `(29000, 100)`: the 19000 corrupted rows then the 10000 clean graded rows.
The clean graded fold ships without labels precisely so the clean-accuracy tolerance gate has an
input. seed/contract.yaml names that gate as the mitigation for the uniform-posterior shortcut,
and a corrupted-only deliverable would leave it reading nothing -- which is the same as not having
it, since a uniform posterior genuinely does lower expected calibration error and nothing inside
the corrupted objective punishes it.

## What is authored and what is measured

Authored at Phase 0: the delivery shape and stacking order, the pinned graded binning, the
compiled structural checks, the red lines and the two obligations. Twelve of the fourteen checks
run against delivered bytes today.

Measured at Phase 1/2 and deliberately null here: `floor`, `knee`, `dichotomizing_threshold`,
`reward_gate_pass_threshold`, `clean_accuracy_floor`, and `graded_axis`. The consequence is stated
rather than hidden -- this bundle is COMPLETE but NOT YET GRADEABLE, and both
`tests/test_output.py::test_clean_accuracy_tolerance` and
`tests/test_output.py::test_corruption_macro_ece_ramp` raise `ConstantUnmeasured` until the
measurement wave writes those constants. `solution/recompute.py` refuses to emit unless the ramp
check's body opens with `require_measured`.

## Obligations

`CK-BINFAM` is carried from seed/contract.yaml, verified there, and now has a compiled body rather
than a lookup into the graded run's own account of itself.

`ck_binpin` is authored HERE, against the recorded `checker_obligations_shortfall`: seed/contract.yaml
requires CTL-GRADEDBINS be rejected and records that no obligation enforced it. The contract still
carries that shortfall paragraph until it is amended outside this bundle, so the gap is closed in
the checker before it is closed in the record.

## Measurement tier

`gpu` -- trains a neural net over an image, audio or dense-label corpus.

## Supersession

`supersedes: null`. This slot has no predecessor generation; it is authored new in this batch.
