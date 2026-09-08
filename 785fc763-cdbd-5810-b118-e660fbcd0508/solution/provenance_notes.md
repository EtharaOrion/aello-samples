# Provenance notes for AELLO-C8-S12

Derived from this slot's seed/contract.yaml record. The canonical carriers are regenerated from
solution/grounding.yaml by solution/recompute.py; nothing below is hand-maintained separately.

## Corpus

CIFAR-100N (Wei et al. 2022), real human annotations over CIFAR-100 images.

One human annotation per training image over the CIFAR-100 fine label set. The annotations are
wrong on a substantial share of the pool and wrong in a STRUCTURED way -- the flip rate differs
class by class and the mass concentrates inside the 20 coarse superclasses, because an annotator
who mislabels a maple tree reaches for another tree. That is what separates this corpus from
synthetically corrupted CIFAR-100 and it is why a symmetric noise model is the wrong model here
rather than a coarser one.

## What this slot withholds, and what it does not

The graded split is CLEAN. The training labels are NOISY. The two label distributions therefore
differ, and every quantity the agent can measure on its own data is measured against the wrong one
of them -- which is the whole task and the reason
`prediction_prior_closer_to_balanced_split_than_to_noisy_prior` and
`noisy_train_accuracy_below_human_agreement_rate` are compiled checks rather than advice.

The clean CIFAR-100 fine labels for the training images exist and are public, so red line 1
forbids the graded run to READ them rather than merely to write over them. Those are different
claims and the bundle checks them separately: `clean_training_labels_unread` reads the run's own
path log, and `checker_tree_and_clean_split_untouched` reads the harness's guarded-write record.
A third check, `retained_subset_not_the_clean_label_ordering`, closes the route that neither log
would see, by measuring the retained subset itself against the verifier's agreement bitmap.

The human agreement rate and the bound pool size are properties of the delivered shard, not Phase 0
literals. They are deliberately absent from every authored carrier: the compiled checks count them
from the delivered bytes, so there is no authored copy of either number for the corpus to drift
away from.

## What is authored and what is measured

Authored at Phase 0: the 100-class fine label domain and its 20 coarse superclasses, the corpus
layout the shard must land at, the run-record declarations the checks grade, the two red lines and
the thirteen compiled checks. Every one of these runs against delivered bytes today.

Measured at Phase 1/2 and deliberately null here: `floor`, `knee`, `dichotomizing_threshold`,
`reward_gate_pass_threshold`, `graded_axis`, the calibrated purity bound of the clean-peek
detector, every control's score and every lever's ablation worth. The consequence is stated rather
than hidden -- this bundle is COMPLETE but NOT YET GRADEABLE, and
`tests/test_output.py::test_clean_split_balanced_accuracy_ramp` calls `require_measured` as its
first statement and raises `ConstantUnmeasured` until the measurement wave writes those constants.

That ordering is deliberate and load-bearing. Were the check to read the submission first, a
missing artifact would raise before the unmeasured signal was ever emitted, and the verifier would
report `gradeable: true` with `r_det: 0.0` -- asserting that the agent scored zero when the truth
is that the slot cannot yet be measured at all.

## Corpus staging

`environment/data/` and `tests/heldout/` have not landed in this bundle. The paths every compiled
check reads are authored in `solution/grounding.yaml` under `corpus_layout`, and each check reports
an absent artifact by name -- `graded_labels_unavailable`, `agreement_bitmap_unavailable`,
`shipped_noisy_annotation_unavailable`, `coarse_map_unavailable`. None of them passes vacuously and
none of them raises.

## Measurement tier

`gpu` -- trains a neural net over an image, audio or dense-label corpus

## Supersession

`supersedes: null`. This slot has no predecessor generation; it is authored new in this batch.
