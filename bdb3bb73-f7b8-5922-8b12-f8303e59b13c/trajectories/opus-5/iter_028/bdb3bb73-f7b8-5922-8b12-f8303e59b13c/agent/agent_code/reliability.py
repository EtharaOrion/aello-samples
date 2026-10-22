"""Post-hoc robustness check of the PROMOTED config on the full train split.

This re-runs the exact promoted recipe under fresh seeds and reports, for each,
the first epoch whose checkpoint clears 94% on the real test split. It is a
reliability measurement only: the graded ladder stays the seed-0 run that was
already written. Picking whichever seed happened to score best here would be
selecting on the test split, so it is deliberately not done.
"""
import glob
import os
import shutil
import time

import torch

from train import get_cfg, load_split, train
from verify import cache_batches, make_loader, score

SCRATCH = "/workspace/scratch_ckpt"

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    tx, ty = load_split("train", "cuda")
    batches = cache_batches(make_loader())

    for seed in (1, 2):
        shutil.rmtree(SCRATCH, ignore_errors=True)
        os.makedirs(SCRATCH, exist_ok=True)
        cfg = get_cfg(epochs=24, seed=seed, lr_end=0.02, cutout=12)
        t0 = time.time()
        train(cfg, tx, ty, ckpt_dir=SCRATCH, log=lambda *a: None)
        wall = time.time() - t0

        paths = sorted(glob.glob(f"{SCRATCH}/ckpt_epoch_*.pt"))
        mt = {int(p[-7:-3]): os.path.getmtime(p) for p in paths}
        first, accs = None, {}
        for p in paths:
            ep = int(p[-7:-3])
            if ep < 14:            # crossing never occurs this early
                continue
            accs[ep] = score(p, batches)
            if accs[ep] >= 0.94 and first is None:
                first = ep
        n_cross = sum(a >= 0.94 for a in accs.values())
        t_star = ((mt[first] - mt[1]) * first / (first - 1)) if first else None
        print(f"seed {seed}: wall={wall:.1f}s  final={accs[24] * 100:.2f}  "
              f"first>=94% at epoch {first}  t*={t_star:.2f}s  "
              f"({n_cross} checkpoints >= 94%)", flush=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)
