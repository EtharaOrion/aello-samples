# Provenance notes for AELLO-C8-S8

Derived from this slot's seed/contract.yaml record. The canonical carriers are regenerated from
solution/grounding.yaml by solution/recompute.py; nothing below is hand-maintained separately.

## Corpus

ADE20K scene parsing (Zhou et al. 2017), 150 stuff and object classes.

Single visible-light scene photographs on heterogeneous pixel grids -- not tiles, not a multi-band
stack. Each scene keeps its own height and width from the shipped imagery through to the graded
mask, which is why the delivered geometry is a per-image-id table rather than one shape for the
split.

## What this slot withholds, and what it does not

The corpus ships one annotation: `environment/data/sparse_labels.npz`, carrying a class id on a
bound fraction of training pixels and the ignore sentinel 255 everywhere else. The dense annotation
of the private split is never in the agent image, and red line 1 forbids the graded run to READ it
rather than merely to write over it. Those are different claims and the bundle checks them
separately: `dense_private_labels_unread` reads the run's own path log, and
`checker_tree_and_graded_masks_untouched` reads the harness's guarded-write record.

The supervision fraction itself is a task parameter fixed when the shard lands. It is deliberately
absent from every authored carrier: the compiled checks measure a declared consumption against the
delivered annotation's own labelled support, so there is no authored copy of the number for the
corpus to drift away from.

## What is authored and what is measured

Authored at Phase 0: the label domain and its ignore sentinel, the per-scene delivery geometry, the
run-record declarations the checks grade, the two red lines and the eleven compiled checks. Every
one of these runs against delivered bytes today.

Measured at Phase 1/2 and deliberately null here: `floor`, `knee`, `dichotomizing_threshold`,
`reward_gate_pass_threshold`, `graded_axis`, every control's score and every lever's ablation worth.
The consequence is stated rather than hidden -- this bundle is COMPLETE but NOT YET GRADEABLE, and
`tests/test_output.py::test_sparse_supervised_miou_ramp` calls `require_measured` as its first
statement and raises `ConstantUnmeasured` until the measurement wave writes those constants.

That ordering is deliberate and load-bearing. Were the check to read the submission first, a
missing artifact would raise before the unmeasured signal was ever emitted, and the verifier would
report `gradeable: true` with `r_det: 0.0` -- asserting that the agent scored zero when the truth is
that the slot cannot yet be measured at all.

## Measurement tier

`gpu` -- trains a neural net over an image, audio or dense-label corpus.

## Supersession

`supersedes: null`. This slot has no predecessor generation; it is authored new in this batch.
