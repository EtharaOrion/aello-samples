"""CIFAR-10 speedrun trainer.

Two modes:
  --split holdout : train on 45k, score on a fixed 5k slice of the *train*
                    split.  Every hyper-parameter decision is made here.
  --split full    : train on all 50k and write the graded checkpoint ladder.
                    The test split is never read by this script.

Checkpoints are written once per epoch as scripted modules.  The module is
scripted once, before training; `torch.jit.script` shares parameter storage
with the eager module, so re-saving it each epoch serialises current weights
without paying the scripting cost 25 times (asserted in tests/test_export.py).
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from model import Net, init_whitening, CIFAR_MEAN, CIFAR_STD

CACHE = "/workspace/cache"


# --------------------------------------------------------------------- data --
_SPLIT_CACHE = {}


def load_split(name, device):
    """uint8 NCHW tensor + int64 labels, kept resident so a sweep driver that
    calls `run()` many times in one process pays the disk cost only once."""
    key = (name, str(device))
    if key not in _SPLIT_CACHE:
        x = torch.from_numpy(np.load(f"{CACHE}/{name}_x.npy")).to(device)
        y = torch.from_numpy(np.load(f"{CACHE}/{name}_y.npy")).to(device)
        _SPLIT_CACHE[key] = (x, y)
    return _SPLIT_CACHE[key]


def standardize(x_u8, dtype=torch.float32):
    mean = torch.tensor(CIFAR_MEAN, device=x_u8.device).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(CIFAR_STD, device=x_u8.device).view(1, 3, 1, 1) * 255.0
    return ((x_u8.float() - mean) / std).to(dtype)


def stratified_holdout(y, n_val, seed=0):
    """Class-balanced validation slice, drawn with a dedicated RNG."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    y_cpu = y.cpu()
    val, tr = [], []
    per = n_val // 10
    for c in range(10):
        idx = (y_cpu == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        val.append(idx[:per])
        tr.append(idx[per:])
    return torch.cat(tr).to(y.device), torch.cat(val).to(y.device)


# ------------------------------------------------------------ augmentation --
class Augmenter:
    """GPU-resident random crop (from a pre-padded tensor) + mirror + cutout."""

    def __init__(self, x_std, pad, device, flip=True):
        self.pad = pad
        self.flip = flip
        self.xp = F.pad(x_std, (pad,) * 4, mode="reflect") if pad else x_std
        self.side = self.xp.shape[-1]
        self.ar = torch.arange(32, device=device).view(1, 1, 32, 1)

    def __call__(self, idx, g, cutout):
        xb = self.xp[idx]
        b, c = xb.shape[0], xb.shape[1]
        if self.pad:
            hi = 2 * self.pad + 1
            oy = torch.randint(0, hi, (b,), device=xb.device, generator=g)
            ox = torch.randint(0, hi, (b,), device=xb.device, generator=g)
            iy = (oy.view(b, 1, 1, 1) + self.ar).expand(b, c, 32, self.side)
            xb = torch.gather(xb, 2, iy)
            ix = (ox.view(b, 1, 1, 1) + self.ar.view(1, 1, 1, 32)).expand(b, c, 32, 32)
            xb = torch.gather(xb, 3, ix)
        if self.flip:
            flip = torch.rand(b, 1, 1, 1, device=xb.device, generator=g) < 0.5
            xb = torch.where(flip, torch.flip(xb, [3]), xb)
        if cutout:
            cy = torch.randint(0, 32, (b, 1, 1), device=xb.device, generator=g)
            cx = torch.randint(0, 32, (b, 1, 1), device=xb.device, generator=g)
            gy = torch.arange(32, device=xb.device).view(1, 32, 1)
            gx = torch.arange(32, device=xb.device).view(1, 1, 32)
            r = cutout // 2
            mask = ((gy - cy).abs() < r) & ((gx - cx).abs() < r)
            xb = xb.masked_fill(mask.unsqueeze(1), 0.0)
        return xb.contiguous(memory_format=torch.channels_last)


# ----------------------------------------------------------------- training --
def build_optimizer(model, cfg):
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay if p.ndim <= 1 else decay).append(p)
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": cfg.wd, "lr": cfg.lr},
         {"params": nodecay, "weight_decay": 0.0, "lr": cfg.lr * cfg.bias_scaler}],
        lr=cfg.lr, momentum=cfg.momentum, nesterov=True)


def lr_factor(step, total, warm_frac):
    warm = max(1, int(total * warm_frac))
    if step < warm:
        return (step + 1) / warm
    return max(0.0, (total - step) / max(1, total - warm))


@torch.no_grad()
def evaluate(model, x_std, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, x_std.shape[0], bs):
        xb = x_std[i:i + bs].float().contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", torch.bfloat16):
            out = model(xb)
        correct += (out.float().argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / x_std.shape[0]


def run(cfg):
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    g = torch.Generator(device=dev).manual_seed(cfg.seed + 1234)

    xtr_u8, ytr = load_split("train", dev)
    if cfg.split == "holdout":
        tr_idx, va_idx = stratified_holdout(ytr, cfg.n_val, seed=cfg.holdout_seed)
        xva, yva = standardize(xtr_u8[va_idx]), ytr[va_idx]
        xtr_u8, ytr = xtr_u8[tr_idx], ytr[tr_idx]
    else:
        xva = yva = None
    xtr = standardize(xtr_u8)
    del xtr_u8
    n = xtr.shape[0]

    model = Net(cfg.w1, cfg.w2, cfg.w3, cfg.scale, cfg.bn_momentum, tta=bool(cfg.tta)).to(dev)
    if cfg.zca:
        init_whitening(model, xtr)
    model = model.to(memory_format=torch.channels_last)
    model.train()

    aug = Augmenter(xtr, cfg.pad, dev, flip=bool(cfg.flip))
    opt = build_optimizer(model, cfg)
    steps_per_epoch = n // cfg.bs
    total = steps_per_epoch * cfg.epochs

    # --- warm-up: trigger cuDNN autotuning / allocator growth before the timed
    # loop.  This is ordinary practice and runs only on random data.
    if cfg.warmup:
        snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
        wu = torch.randn(cfg.bs, 3, 32, 32, device=dev).contiguous(memory_format=torch.channels_last)
        wl = torch.randint(0, 10, (cfg.bs,), device=dev)
        for _ in range(3):
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(model(wu), wl)
            loss.backward()
        del wu, wl, loss
        # Warm-up must leave no trace: restore weights *and* BN running stats,
        # drop the accumulated gradients and rebuild the optimizer state.
        with torch.no_grad():
            model.load_state_dict(snapshot)
        del snapshot
        for p in model.parameters():
            p.grad = None
        opt = build_optimizer(model, cfg)
        model.train()

    scripted = torch.jit.script(model) if cfg.save else None
    if cfg.save:
        os.makedirs(cfg.outdir, exist_ok=True)
        for f in os.listdir(cfg.outdir):
            if f.endswith(".pt"):
                os.remove(os.path.join(cfg.outdir, f))

    torch.cuda.synchronize()
    t0 = time.time()
    step = 0
    stamps = []
    for epoch in range(1, cfg.epochs + 1):
        perm = torch.randperm(n, device=dev, generator=g)
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.bs:(i + 1) * cfg.bs]
            xb = aug(idx, g, cfg.cutout)
            f = lr_factor(step, total, cfg.warm_frac)
            opt.param_groups[0]["lr"] = cfg.lr * f
            opt.param_groups[1]["lr"] = cfg.lr * cfg.bias_scaler * f
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(model(xb), ytr[idx], label_smoothing=cfg.ls)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        if cfg.save:
            model.eval()
            scripted.save(os.path.join(cfg.outdir, f"ckpt_epoch_{epoch:04d}.pt"))
            model.train()
        stamps.append(time.time() - t0)
    torch.cuda.synchronize()
    wall = time.time() - t0

    res = {"wall": wall, "epochs": cfg.epochs, "stamps": stamps,
           "params": sum(p.numel() for p in model.parameters())}
    if xva is not None:
        res["val_acc"] = evaluate(model, xva, yva)
    return model, res


def get_cfg(**over):
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.85)
    p.add_argument("--wd", type=float, default=0.002)
    p.add_argument("--bias_scaler", type=float, default=8.0)
    p.add_argument("--ls", type=float, default=0.2)
    p.add_argument("--warm_frac", type=float, default=0.15)
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--cutout", type=int, default=0)
    p.add_argument("--w1", type=int, default=128)
    p.add_argument("--w2", type=int, default=384)
    p.add_argument("--w3", type=int, default=512)
    p.add_argument("--scale", type=float, default=0.125)
    p.add_argument("--bn_momentum", type=float, default=0.6)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--flip", type=int, default=1)
    p.add_argument("--zca", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout_seed", type=int, default=0)
    p.add_argument("--n_val", type=int, default=5000)
    p.add_argument("--split", type=str, default="holdout")
    p.add_argument("--save", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--outdir", type=str, default="/workspace/scratch_ckpt")
    cfg = p.parse_args([] if over else None)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    cfg = get_cfg()
    model, res = run(cfg)
    print(json.dumps({**res, "cfg": vars(cfg)}))
