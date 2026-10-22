# CIFAR-10 to 94% on a single H100

94.80% on the CIFAR-10 test split at `t_star = 21.32 s`, 95.06% after 52 epochs
(27.8 s end to end). Trained from scratch on the CIFAR-10 train split only.

* `PREREGISTRATION.md` -- the selection and promotion rule, written before any
  sweep was run.
* `RESULTS.md` -- what was measured: noise floor, screen, confirmation,
  ablations, and the graded ladder's per-epoch test curve.

## Layout

| file | role |
|---|---|
| `model.py` | the network, the frozen ZCA whitening init, input canonicalisation and in-`forward` 6-way TTA |
| `train.py` | training loop; `--split holdout` for decisions, `--split full` for the graded ladder |
| `prep_data.py` | one-time cache of the PNG folders into uint8 `.npy` |
| `sweep.py` | noise floor + one-axis-at-a-time screen |
| `stage2.py` | combination / budget-scaling runs, then the 3-seed confirmation |
| `ablate.py` | leave-one-out ablations around the promoted config |
| `seedtest.py` | 10-seed paired re-test of the two finalists (post-hoc, reporting only) |
| `verify_ladder.py` | re-scores a ladder through the verifier's exact pipeline |
| `tests/test_export.py` | the two export contracts (see below) |

## Reproduce

```bash
python prep_data.py                       # ~20 s, once

# the graded run (writes ckpt_epoch_0001..0052 + hardlinks checkpoint.pt)
python train.py --split full --save 1 --outdir /workspace/periodic_checkpoints \
    --epochs 52 --lr 0.6 --momentum 0.9 --wd 0.004 --bias_scaler 1.0 --cutout 8

python verify_ladder.py /workspace/periodic_checkpoints   # independent re-score
python tests/test_export.py /workspace/checkpoint.pt      # export contracts

# how the config was chosen (all on the 45k/5k split of *train*)
python sweep.py all && python stage2.py && python ablate.py
```

Two fresh processes running the graded command produced bit-identical ladders
(52/52 epochs), so the commands above reproduce the reported numbers, not merely
similar ones.

## The two export contracts

Both are asserted by `tests/test_export.py`, because both are silent failure
modes rather than crashes:

1. **The verifier's inputs are already standardised.** `Net.canonicalize` is the
   identity on them and only rescales `[0,1]` / `[0,255]` input. Normalising
   twice would still look like 94% under a `[0,1]` evaluation while collapsing
   under the verifier.
2. **`torch.jit.script` shares parameter storage with the eager module.** The
   loop scripts once and re-saves that same `ScriptModule` each epoch; the test
   mutates weights *after* scripting and asserts the change is serialised, so a
   ladder of 52 identical stale checkpoints cannot slip through.
