"""Train one full run and write the checkpoint ladder.

    python run.py --out /workspace/periodic_checkpoints --epochs 20

Writes ckpt_epoch_0001.pt .. ckpt_epoch_NNNN.pt (one genuine pass over the
50k train split each) plus a run_log.json next to them.
"""
import argparse
import json
import os
import shutil
import time

import torch

import airbench as A

T_PROCESS_START = time.perf_counter()

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/scratch_ckpt/run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="cfg overrides, e.g. lr=1.6 epochs=18")
    ap.add_argument("--final", default="", help="also hardlink last ckpt here")
    ap.add_argument("--holdout", type=int, default=0, help="hold out N train images for eval")
    args = ap.parse_args()

    cfg = dict(A.DEFAULT_CFG)
    for kv in args.set:
        k, v = kv.split("=", 1)
        assert k in cfg, f"unknown cfg key {k}"
        cfg[k] = type(cfg[k])(eval(v)) if not isinstance(cfg[k], tuple) else eval(v)

    xtr, ytr = A.load_split("train")
    eval_sets = None
    if args.holdout:
        g = torch.Generator().manual_seed(1234)  # fixed split, independent of run seed
        perm = torch.randperm(len(ytr), generator=g)
        hold, keep = perm[:args.holdout], perm[args.holdout:]
        eval_sets = {"holdout": (xtr[hold], ytr[hold])}
        xtr, ytr = xtr[keep], ytr[keep]

    data = A.GpuTrainData(xtr, ytr)
    whiten_images = A.standardize(xtr[:10000])

    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out, exist_ok=True)
    print(json.dumps({k: str(v) for k, v in cfg.items()}, indent=None))
    A.warmup_gpu(cfg)  # steady clocks + cudnn algo selection before the timed loop
    recs, net, wall = A.train(cfg, data, whiten_images, seed=args.seed,
                              ckpt_dir=args.out, eval_sets=eval_sets)
    print(f"wall_clock_sec {wall:.3f}   ({wall / cfg['epochs']:.3f} s/epoch)")

    if args.final:
        last = os.path.join(args.out, f"ckpt_epoch_{cfg['epochs']:04d}.pt")
        if os.path.exists(args.final):
            os.remove(args.final)
        shutil.copy(last, args.final)

    # total = everything this process did, including data load and GPU warm-up;
    # reported as wall_clock_sec because it is conservatively >= the ckpt span.
    total = time.perf_counter() - T_PROCESS_START
    with open("/workspace/run_log.json", "w") as f:  # kept out of the ladder dir
        json.dump(dict(cfg={k: str(v) for k, v in cfg.items()}, seed=args.seed,
                       wall_clock_sec=wall, total_wall_clock_sec=total,
                       records=recs), f, indent=2)
    print(f"total_wall_clock_sec {total:.3f}")


if __name__ == "__main__":
    main()
