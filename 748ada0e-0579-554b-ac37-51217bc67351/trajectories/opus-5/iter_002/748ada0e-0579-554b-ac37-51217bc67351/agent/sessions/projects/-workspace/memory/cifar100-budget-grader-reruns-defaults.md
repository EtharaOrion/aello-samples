---
name: cifar100-budget-grader-reruns-defaults
description: "In the CIFAR-100 fixed-step-budget task, the grader re-trains from the shipped code, so tuned hyperparameters must live in argparse DEFAULTS, not CLI flags."
metadata: 
  node_type: memory
  type: project
  originSessionId: 8226c946-6e6b-4702-84cd-cc85201a32e3
  modified: 2026-09-03T05:42:22.641Z
---

In the CIFAR-100 "fixed step budget" refinement-loop task (`environment/starter/train.py`),
the graded pipeline is re-run from scratch with the canonical command:

    python3 train.py --instance i20k --seed 0 \
        --graded ../data/graded_images.npz --out /submission/preds_i20k.npy

Any hyperparameter passed only as a CLI flag during local sweeps is therefore **discarded**
at grading time. Attempt 1 (2026-09-02) tuned width 96 / wd 2e-3 on the command line but
left the file's defaults at width 64 / wd 5e-4, so the graded run used the untuned config.

**Why:** local val read 0.7065 while the graded reward implied balanced accuracy ~0.614.
Re-measuring the old defaults locally gave 0.6691, confirming the defaults were what ran.

**How to apply:** after any sweep, bake the winning values into `ap.add_argument(...)`
defaults and re-run the bare canonical command to confirm it reproduces the tuned config.
Also verify `--pick`-style selection logic actually fires — a `str.replace` patch that
targets a two-line `add_argument` silently no-ops when the source has it on one line.
Related: [[cifar100-short-budget-regularisation-hurts]]
