"""Train on the full 50k train split and write a contiguous checkpoint ladder.

Writes ckpt_epoch_0001.pt .. ckpt_epoch_{E}.pt into --out (one genuine single pass over
the training split per epoch), then copies the last one to --final.
Timestamps come only from ScriptModule.save() and shutil.copy.
"""
import argparse, json, os, shutil, sys, time
import torch

sys.path.insert(0, "/workspace")
from train import DEFAULT_CFG, load_split, train_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/scratch_ckpt/ladder")
    ap.add_argument("--final", default=None)
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", default="{}", help="JSON dict of config overrides")
    ap.add_argument("--meta", default=None, help="where to write run metadata JSON")
    a = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    cfg.update(json.loads(a.cfg))
    cfg["epochs"] = a.epochs

    dev = "cuda"
    x, y = load_split("train", dev)
    assert len(x) == 50000, len(x)

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out, exist_ok=True)

    t_run_start = time.time()
    r = train_run(cfg, seed=a.seed, ckpt_dir=a.out, val_data=None,
                  x_tr=x, y_tr=y, log="ladder", quiet=True)
    t_run_end = time.time()

    files = sorted(os.listdir(a.out))
    assert len(files) == a.epochs, (len(files), a.epochs)
    mt = [os.path.getmtime(os.path.join(a.out, f)) for f in files]
    span = mt[-1] - mt[0]
    n = len(mt)
    per_epoch = span / (n - 1)
    meta = dict(cfg=cfg, seed=a.seed, epochs=a.epochs,
                run_wall_sec=t_run_end - t_run_start,
                train_loop_sec=r["wall"],
                ckpt_span_sec=span,
                mean_epoch_sec=per_epoch,
                nparam=r["nparam"],
                # what the verifier would reconstruct if epoch N were the crossing epoch
                t_star_if_cross={k: round(per_epoch * k, 3)
                                 for k in range(2, a.epochs + 1)})
    print(json.dumps({k: v for k, v in meta.items() if k != "t_star_if_cross"}, indent=1))

    if a.final:
        shutil.copy(os.path.join(a.out, files[-1]), a.final)
    if a.meta:
        with open(a.meta, "w") as f:
            json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
