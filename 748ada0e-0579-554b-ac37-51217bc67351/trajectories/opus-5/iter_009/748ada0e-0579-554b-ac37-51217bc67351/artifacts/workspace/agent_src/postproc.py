"""Offline prediction-time sweep on a saved checkpoint.

Everything here happens AFTER training, on a model produced by a single within-budget
run, so it costs no optimiser steps and no training-sample presentations.  Comparisons
are paired on one fixed checkpoint, which removes run-to-run training noise and makes
sub-0.5% differences measurable (only val sampling noise remains, and it is shared).

Val labels are used to MEASURE, never to fit anything that is then applied to graded:
the post-processing rules below are unsupervised (they read only the model's own
predicted marginal on whatever split they are applied to).
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/environment")
sys.path.insert(0, "/workspace/environment/starter")
import aello_data as data
import train as T


def load(ckpt, device="cuda"):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    m = T.make_model(cfg["seed"], data.N_CLASSES, arch=cfg["arch"], width=cfg["width"],
                     dropout=cfg["dropout"], se=cfg["se"], dsample=cfg["dsample"],
                     droppath=cfg["droppath"], act=cfg["act"], deepstem=cfg["deepstem"],
                     aux=cfg.get("aux", 0.0)).to(device)
    m.load_state_dict(ck["sd"])
    m = m.to(memory_format=torch.channels_last).eval()
    return m, cfg


@torch.no_grad()
def views_logprob(model, xn, pad, mode):
    """Mean log-prob over a view set.  `mode` selects the view set."""
    h = xn.shape[-1]
    if mode == "none":
        offs, do_flip = [(pad, pad)], False
    elif mode == "flip":
        offs, do_flip = [(pad, pad)], True
    elif mode == "flip5":
        offs, do_flip = T.crops5(pad), True
    elif mode == "flip9":
        offs, do_flip = T.crops9(pad), True
    elif mode == "flip13":
        a = max(1, pad // 2)
        offs = T.crops5(pad) + [(pad, pad - a), (pad, pad + a),
                                (pad - a, pad), (pad + a, pad)]
        do_flip = True
    else:
        raise ValueError(mode)
    out = []
    for s in range(0, xn.shape[0], 1000):
        xb = xn[s:s + 1000]
        xp = F.pad(xb, (pad,) * 4, mode="reflect")
        acc, nv = None, 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for (oy, ox) in offs:
                v = xp[:, :, oy:oy + h, ox:ox + h].contiguous(
                    memory_format=torch.channels_last)
                lg = F.log_softmax(model(v).float(), 1); nv += 1
                if do_flip:
                    lg = lg + F.log_softmax(model(v.flip(3)).float(), 1); nv += 1
                acc = lg if acc is None else acc + lg
        out.append((acc / nv).float())
    return torch.cat(out)


def bacc(lp, y):
    return T.balanced_acc(lp.argmax(1).cpu().numpy(), y)


def adjust(lp, tau):
    """Logit adjustment by the model's own predicted marginal on this split.

    The Bayes rule for balanced accuracy is argmax p(c|x)/p(c); q(c) = mean_x p(c|x)
    estimates the split's class prior without ever touching labels.  tau=0 is a no-op.
    """
    if tau == 0:
        return lp
    q = lp.exp().mean(0).clamp_min(1e-12)
    return lp - tau * q.log()[None, :]


def sinkhorn(lp, iters=20, target=None, tau=1.0):
    """Iterative proportional fitting: rescale columns so the mean predicted
    distribution matches `target` (uniform by default).  Unsupervised."""
    n, c = lp.shape
    t = torch.full((c,), 1.0 / c, device=lp.device) if target is None else target
    b = torch.zeros(c, device=lp.device)
    for _ in range(iters):
        p = F.softmax(lp + b[None, :], 1)
        q = p.mean(0).clamp_min(1e-12)
        b = b + tau * (t.log() - q.log())
    return lp + b[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ck_b1.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tta", default="flip5")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--sink", type=int, default=0)
    ap.add_argument("--sweep", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True

    model, cfg = load(a.ckpt, dev)
    xtr, ytr = data.load_train(cfg["instance"])
    xva, yva = data.load_val()
    tr = T.GPUData(xtr, ytr, dev, pad=cfg["pad"], res=cfg["res"])
    pad = tr.pad
    print("res %d pad %d" % (tr.res, pad))

    cnt = np.bincount(yva, minlength=100)
    print("val class counts: min %d max %d mean %.1f" % (cnt.min(), cnt.max(), cnt.mean()))

    cache = {}
    for mode in (["none", "flip", "flip5", "flip9", "flip13"] if a.sweep else [a.tta]):
        xvn = tr.norm(xva)
        lp = views_logprob(model, xvn, pad, mode)
        cache[mode] = lp
        print("[tta %-7s] val bacc %.4f  acc %.4f" %
              (mode, bacc(lp, yva), float((lp.argmax(1).cpu().numpy() == yva).mean())))

    if a.sweep:
        base = cache["flip5"]
        print("\n-- logit adjustment by predicted marginal (on flip5) --")
        for tau in [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0]:
            print("  tau %.3f  val bacc %.4f" % (tau, bacc(adjust(base, tau), yva)))
        print("\n-- sinkhorn to uniform marginal (on flip5) --")
        for it in [1, 3, 10, 30]:
            print("  iters %2d  val bacc %.4f" % (it, bacc(sinkhorn(base, it), yva)))
        print("\n-- probability-space vs log-prob-space view averaging --")
        # (log-prob space is what `cache` holds; compare with prob-space below)
        for mode in ["flip5", "flip9"]:
            lp = cache[mode]
            print("  %s logprob %.4f" % (mode, bacc(lp, yva)))

    if a.out:
        lp = cache[a.tta] if a.tta in cache else views_logprob(
            tr.norm(xva), pad, a.tta)
        xg = data.load_graded_images()
        lpg = views_logprob(model, tr.norm(xg), pad, a.tta)
        if a.sink:
            lpg = sinkhorn(lpg, a.sink)
        lpg = adjust(lpg, a.tau)
        pred = lpg.argmax(1).cpu().numpy().astype(np.int64)
        np.save(a.out, pred)
        print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
