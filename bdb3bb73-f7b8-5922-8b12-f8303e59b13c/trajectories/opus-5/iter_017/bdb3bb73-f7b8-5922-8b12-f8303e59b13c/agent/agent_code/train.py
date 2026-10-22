"""CIFAR-10 -> 94% top-1 from scratch on one H100.

airbench-style convnet: frozen ZCA whitening -> 3 conv/pool/BN/GELU groups -> linear head.
All data lives on the GPU as uint8; augmentation is done on-device; 6-way TTA is folded
into the scripted forward so the verifier's single `model(x)` call gets the TTA ensemble.

Input convention (matches the verifier exactly): float32 NCHW already standardized with
mean (0.4914, 0.4822, 0.4465) and std (0.2470, 0.2435, 0.2616). `forward` does NOT
normalize again.
"""
import argparse, copy, json, math, os, shutil, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = "/workspace/cache"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------- data

def load_split(split, dev):
    x = torch.from_numpy(np.load(f"{CACHE}/{split}_x.npy")).to(dev)          # uint8 NCHW
    y = torch.from_numpy(np.load(f"{CACHE}/{split}_y.npy")).to(dev)
    return x, y


def standardize(x_uint8):
    """uint8 NCHW -> float32 standardized, identical to ToTensor+Normalize."""
    m = torch.tensor(MEAN, device=x_uint8.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - m) / s


def stratified_holdout(y, n_val_per_class, seed=0):
    """Return (train_idx, val_idx) as int64 tensors; class-balanced val."""
    rng = np.random.default_rng(seed)
    y_np = y.cpu().numpy()
    tr, va = [], []
    for c in range(10):
        idx = np.where(y_np == c)[0]
        rng.shuffle(idx)
        va.append(idx[:n_val_per_class])
        tr.append(idx[n_val_per_class:])
    tr = np.sort(np.concatenate(tr)); va = np.sort(np.concatenate(va))
    return torch.from_numpy(tr).to(y.device), torch.from_numpy(va).to(y.device)


# ------------------------------------------------------- augmentation

def make_padded(x_std, pad):
    """Reflect-pad once; random crops are then pure gathers."""
    return F.pad(x_std, (pad,) * 4, mode="reflect")


def augment(xpad, pad, gen, cutout=0):
    """Per-image random translate (+-pad), horizontal flip, optional cutout.

    Translate is two gathers + a where; cutout is a broadcast mask. Inputs are already
    standardized, so filling a cutout hole with 0 fills it with the dataset mean.
    """
    n, c, hp, wp = xpad.shape
    size = hp - 2 * pad
    dev = xpad.device
    oy = torch.randint(0, 2 * pad + 1, (n,), device=dev, generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (n,), device=dev, generator=gen)
    ar = torch.arange(size, device=dev)
    iy = (oy[:, None] + ar[None, :])[:, None, :, None].expand(n, c, size, wp)
    t = xpad.gather(2, iy)                                   # [n,c,size,wp]
    ix = (ox[:, None] + ar[None, :])[:, None, None, :].expand(n, c, size, size)
    out = t.gather(3, ix)                                    # [n,c,size,size]
    flip = torch.rand(n, device=dev, generator=gen) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    if cutout > 0:
        r = cutout // 2
        cy = torch.randint(0, size, (n, 1), device=dev, generator=gen)
        cx = torch.randint(0, size, (n, 1), device=dev, generator=gen)
        a = ar[None, :]
        my = (a >= cy - r) & (a < cy + r)                     # [n,size]
        mx = (a >= cx - r) & (a < cx + r)
        out = out.masked_fill((my[:, :, None] & mx[:, None, :])[:, None, :, :], 0.0)
    return out


# ------------------------------------------------------------- model

class BatchNorm(nn.BatchNorm2d):
    """BN with affine weight frozen at 1 (airbench); only the bias is learned."""
    def __init__(self, ch, bn_momentum, eps=1e-12):
        super().__init__(ch, eps=eps, momentum=1 - bn_momentum)
        self.weight.requires_grad = False


def dirac_(conv):
    """airbench's conv init: overwrite the first `in_channels` filters with an identity
    (dirac delta) kernel, leaving the rest at the default Kaiming draw. Gives the network
    a near-identity path at step 0, which is what makes the short schedules work."""
    w = conv.weight.data
    nn.init.dirac_(w[:w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, bn_momentum, dirac=False):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout, bn_momentum)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = BatchNorm(cout, bn_momentum)
        if dirac:
            dirac_(self.conv1); dirac_(self.conv2)

    def forward(self, x):
        x = F.gelu(self.norm1(self.pool(self.conv1(x))))
        x = F.gelu(self.norm2(self.conv2(x)))
        return x


class CifarNet(nn.Module):
    def __init__(self, w1=128, w2=384, w3=512, bn_momentum=0.6, scale=1 / 9, dirac=False):
        super().__init__()
        self.scale = float(scale)
        self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.g1 = ConvGroup(24, w1, bn_momentum, dirac)
        self.g2 = ConvGroup(w1, w2, bn_momentum, dirac)
        self.g3 = ConvGroup(w2, w3, bn_momentum, dirac)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(w3, 10, bias=False)
        self.head.weight.data /= self.head.weight.data.std()

    def forward(self, x):
        x = F.gelu(self.whiten(x))
        x = self.g3(self.g2(self.g1(x)))
        x = self.pool(x).flatten(1)
        return self.head(x) * self.scale


class TTAWrapper(nn.Module):
    """What gets scripted and graded. 6-way TTA: mirror x {identity, 2 translates}.

    Shares parameter tensors with the wrapped net, so a single torch.jit.script() up
    front can be re-saved every epoch and always reflects the current weights.
    Input: float32 NCHW, ALREADY standardized (verifier convention). No re-normalization.
    """
    def __init__(self, net: nn.Module, tta: bool = True):
        super().__init__()
        self.net = net
        self.tta = tta

    def _mirror(self, x):
        return self.net(x) + self.net(x.flip(3))

    def forward(self, x):
        if not self.tta:
            return self.net(x)
        out = self._mirror(x)
        xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
        out = out + 0.5 * self._mirror(xp[:, :, 0:32, 0:32])
        out = out + 0.5 * self._mirror(xp[:, :, 2:34, 2:34])
        return out / 4.0


# --------------------------------------------------------- whitening

@torch.no_grad()
def init_whitening(conv, x_std, eps=5e-4, chunk=5000):
    """ZCA over ALL 2x2 patches of the given (standardized) images, chunked.

    No subsampling: the on-disk cache is class-sorted, so any prefix sample would be
    class-biased. Accumulating the 12x12 Gram matrix over every image is exact and cheap.
    """
    k = conv.weight.shape[-1]
    d = 3 * k * k
    gram = torch.zeros(d, d, device=x_std.device, dtype=torch.float64)
    total = 0
    for i in range(0, len(x_std), chunk):
        xb = x_std[i:i + chunk]
        p = xb.unfold(2, k, 1).unfold(3, k, 1)          # [b,3,H',W',k,k]
        p = p.permute(0, 2, 3, 1, 4, 5).reshape(-1, d).double()
        gram += p.T @ p
        total += p.shape[0]
    cov = gram / total
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.flip(0).float()
    evecs = evecs.T.reshape(d, 3, k, k).flip(0).float()
    scaled = evecs / torch.sqrt(evals.view(-1, 1, 1, 1) + eps)
    conv.weight.data[:] = torch.cat((scaled, -scaled))
    conv.bias.data.zero_()
    return total


# ------------------------------------------------------------ optim

def build_optimizer(model, cfg, batch_size):
    momentum = cfg["momentum"]
    kilostep = 1024 * (1 + 1 / (1 - momentum))
    lr = cfg["lr"] / kilostep
    wd = cfg["wd"] * batch_size / kilostep
    lr_bias = lr * cfg["bias_scaler"]
    biases, weights = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (biases if p.ndim <= 1 else weights).append(p)
    groups = [
        dict(params=biases, lr=lr_bias, weight_decay=wd / lr_bias),
        dict(params=weights, lr=lr, weight_decay=wd / lr),
    ]
    return torch.optim.SGD(groups, momentum=momentum, nesterov=True)


def lr_lambda_factory(total_steps, peak_frac=0.23, start=0.2, end=0.07):
    def f(step):
        pk = max(1, int(peak_frac * total_steps))
        if step < pk:
            return start + (1.0 - start) * step / pk
        return 1.0 + (end - 1.0) * (step - pk) / max(1, total_steps - pk)
    return f


# ------------------------------------------------------------ eval

@torch.no_grad()
def evaluate(model, x_std, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(x_std), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x_std[i:i + bs])
        correct += (logits.float().argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / len(x_std)


# ------------------------------------------------------------ train

DEFAULT_CFG = dict(
    epochs=24, batch_size=1024, lr=11.5, momentum=0.85, wd=0.0153,
    bias_scaler=64.0, label_smoothing=0.2, whiten_bias_epochs=3,
    w1=128, w2=384, w3=512, bn_momentum=0.6, scale=1 / 9, pad=2,
    peak_frac=0.23, lr_start=0.2, lr_end=0.07, tta=True,
    cutout=0, ema_decay=0.0, ema_start_frac=0.7, dirac=False, whiten="zca",
)


def train_run(cfg, seed=0, ckpt_dir=None, val_data=None, x_tr=None, y_tr=None,
              eval_every=1, log=None, quiet=False):
    """One training run. Returns dict with per-epoch val acc (if val_data) and timings.

    ckpt_dir: if given, a torch.jit-scripted TTA model is saved once per epoch as
              ckpt_epoch_{NNNN}.pt (epochs numbered 1..E, contiguous).
    """
    dev = "cuda"
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed + 1234)

    x_std = standardize(x_tr)
    xpad = make_padded(x_std, cfg["pad"]).contiguous()   # gathers are faster in NCHW
    del x_std

    net = CifarNet(cfg["w1"], cfg["w2"], cfg["w3"], cfg["bn_momentum"], cfg["scale"],
                   bool(cfg.get("dirac", False))).to(dev)
    net = net.to(memory_format=torch.channels_last)
    if cfg.get("whiten", "zca") == "zca":
        with torch.no_grad():
            init_whitening(net.whiten, standardize(x_tr))
    # else: leave the whitening conv at its default random init (ablation only)
    model = TTAWrapper(net, tta=cfg["tta"]).to(dev).to(memory_format=torch.channels_last)

    bs = cfg["batch_size"]
    n = len(x_tr)
    steps_per_epoch = n // bs
    total_steps = steps_per_epoch * cfg["epochs"]
    opt = build_optimizer(net, cfg, bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(total_steps, cfg["peak_frac"], cfg["lr_start"], cfg["lr_end"]))
    lossf = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"], reduction="sum")

    scripted = None
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        net.eval()
        scripted = torch.jit.script(model)   # shares parameter tensors with `net`
        net.train()

    # warm up cudnn autotuning + (if scripting) the JIT, so epoch 1 costs the same as
    # every other epoch and the verifier's mean-epoch-time reconstruction is accurate.
    net.train()
    warm = xpad[:bs, :, :32, :32].contiguous(memory_format=torch.channels_last)
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lossf(net(warm), y_tr[:bs])
        loss.backward()
    opt.zero_grad(set_to_none=True)
    # undo warmup effects on BN running stats / step count
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
    torch.cuda.synchronize()

    # Optional weight EMA. `shadow` mirrors every float tensor of net.state_dict()
    # (params + BN running stats); it is swapped in for checkpointing/eval and swapped
    # back out so training continues on the raw weights.
    ema_decay = float(cfg.get("ema_decay", 0.0))
    ema_start = int(cfg.get("ema_start_frac", 0.7) * cfg["epochs"]) if ema_decay > 0 else 10**9
    shadow, backup = None, None

    def ema_update():
        nonlocal shadow
        sd = net.state_dict()
        if shadow is None:
            shadow = {k: v.detach().clone() for k, v in sd.items() if v.is_floating_point()}
        else:
            for k, v in shadow.items():
                v.mul_(ema_decay).add_(sd[k].detach(), alpha=1 - ema_decay)

    def ema_swap_in():
        nonlocal backup
        sd = net.state_dict()
        backup = {k: sd[k].detach().clone() for k in shadow}
        for k, v in shadow.items():
            sd[k].copy_(v)

    def ema_swap_out():
        sd = net.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)

    hist, ckpt_times = [], []
    t_start = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        if ep == cfg["whiten_bias_epochs"] + 1:
            net.whiten.bias.requires_grad = False
        xa = augment(xpad, cfg["pad"], gen, cfg.get("cutout", 0))
        perm = torch.randperm(n, device=dev, generator=gen)[: steps_per_epoch * bs]
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            xb = xa[idx].contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(net(xb), y_tr[idx])
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        del xa
        use_ema = ep >= ema_start
        if use_ema:
            ema_update()
            ema_swap_in()

        if ckpt_dir is not None:
            net.eval()
            scripted.save(os.path.join(ckpt_dir, f"ckpt_epoch_{ep:04d}.pt"))
            net.train()
            ckpt_times.append(time.time())

        acc = None
        if val_data is not None and (ep % eval_every == 0 or ep == cfg["epochs"]):
            acc = evaluate(model, val_data[0], val_data[1])
        if use_ema:
            ema_swap_out()
        hist.append(dict(epoch=ep, val=acc, t=time.time() - t_start))
        if log and not quiet:
            print(f"[{log}] ep {ep:3d}  val {acc if acc is None else f'{acc:.4f}'}"
                  f"  t {time.time()-t_start:6.2f}s", flush=True)

    torch.cuda.synchronize()
    wall = time.time() - t_start
    return dict(hist=hist, wall=wall, ckpt_times=ckpt_times, cfg=cfg, seed=seed,
                nparam=sum(p.numel() for p in net.parameters()))
