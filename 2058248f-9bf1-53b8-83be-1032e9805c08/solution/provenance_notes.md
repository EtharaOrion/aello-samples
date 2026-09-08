# Provenance notes for A5-02

## Corpus

BigEarthNet-S2 v1.0 (Sumbul et al., bigearth.net), Community Data License Agreement Permissive
1.0, over Copernicus Sentinel-2 L2A imagery under the Copernicus open data terms. Acquisition at
build time from https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz with the Zenodo mirror
as fallback, pinned by sha256. The shipped training shard is a fixed 80,000-patch subset at
120x120x12 uint16 (approximately 2.8 GiB) under the official 19-class nomenclature, with the
recommended cloud, shadow and snow exclusions applied. Country attribution is derived from the
official patch-to-tile metadata, so the held-out split is a genuine geographic holdout over two of
the ten countries rather than a random partition. A fixed private permutation of the 12 spectral
band indices and of the 19 label indices is applied identically to the shipped shard and to both
graded splits, which is what makes any pretrained band ordering worthless here.

## The corpus has not landed in this tree

`environment/` currently carries the Dockerfile and the starter only. The shard, the 60-run sweep
archive and the held-out digests are not present, so `corpus_layout` in `solution/grounding.yaml`
AUTHORS the paths this slot needs rather than reading them off delivered bytes, and says so.
Nothing under `environment/data/` or `tests/heldout/` was created or modified. No compiled check
depends on the corpus being present: each one reads the score matrix, the export manifest, the
parameter blob or the run record, and the facts that would otherwise be read off the corpus --
held-out and archive digests -- are read from the run record's own digest ledger instead.

## What is authored and what is measured

Authored at Phase 0: the deliverable layout and the export manifest schema, the twenty-three
compiled checks, the operator envelope, the three deployment ceilings, the accounted-FLOP budget,
the two divergence bounds and the closed reason set. Every one of those is a property of the task
envelope and every one runs against delivered bytes today.

Measured at Phase 1/2 and deliberately null here:

- `floor` and `knee`, the in-distribution macro AP band;
- `shift_floor` and `shift_knee`, the held-out-country macro AP band. These are a **separate**
  band rather than aliases of the first two. The two splits are different distributions, and
  normalizing both on one band would assert that they are the same;
- `dichotomizing_threshold` and `reward_gate_pass_threshold`;
- `host_flops_per_second`, `host_kernel_launch_seconds` and `host_hbm_bytes_per_second`, the three
  coefficients of the analytic latency model, measured single-tenant on the grading host and
  rotated within a disclosed plus or minus 8 percent band between attempts.

The consequence is stated rather than hidden: this bundle is COMPLETE but NOT YET GRADEABLE.
`tests/test_output.py` raises `ConstantUnmeasured` from `macro_ap_product_ramp` and from
`sized_against_the_pinned_host_constants` until the measurement wave writes those constants, and
`tests/verifier.py` reports `gradeable: false` with `r_det: null`. Null is not zero. Zero asserts
that the agent failed; null asserts that the slot is not yet measurable.

## Measurement tier

`gpu` -- trains a from-scratch classifier over multispectral imagery.

## Supersession

`supersedes: null`. This slot has no predecessor generation; it is authored new in this batch.
