"""CIFAR-10 -> 94% speedrun on one H100.

Architecture (an airbench-style convnet, ~6.1M params):
    frozen 2x2 ZCA whitening conv (3 -> 24) + GELU
    3 x [conv3x3 -> maxpool2 -> BN -> GELU -> conv3x3 -> BN -> GELU]  widths 128/384/512
    maxpool3 -> flatten -> Linear(512, 10, bias=False) -> * 1/9

Conventions that matter for the verifier:
  * forward() expects inputs already standardized with CIFAR mean/std (the
    verifier's transform); no normalization happens inside the model.
  * forward() casts the input to the parameter dtype and returns float32, so a
    half-precision checkpoint still accepts the verifier's float32 batches.
  * 6-way TTA (mirror x {identity, -1px, +1px} reflect-translate) is baked into
    forward(), so torch.jit.load(...)(x).argmax(1) gets the TTA prediction.

All development decisions are made on a holdout carved out of the train split;
the test split is never seen during training.
"""
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CACHE = "/workspace/data_cache"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class ConvGroup(nn.Module):
    def __init__(self, cin: int, cout: int, bn_mom: float):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(cout, eps=1e-12, momentum=bn_mom)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(cout, eps=1e-12, momentum=bn_mom)
        self.act = nn.GELU()
        for norm in (self.norm1, self.norm2):
            norm.weight.data.fill_(1.0)
            norm.weight.requires_grad = False  # BN scale is redundant before a conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.pool(self.conv1(x))))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    def __init__(self, widths=(128, 384, 512), bn_mom: float = 0.4, scale: float = 1 / 9):
        super().__init__()
        ww = 24  # 2 * 3 * 2*2 : the +/- pairs of the 12 whitening eigenvectors
        self.whiten = nn.Conv2d(3, ww, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.whiten.bias.data.zero_()
        self.act = nn.GELU()
        self.g1 = ConvGroup(ww, widths[0], bn_mom)
        self.g2 = ConvGroup(widths[0], widths[1], bn_mom)
        self.g3 = ConvGroup(widths[1], widths[2], bn_mom)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(widths[2], 10, bias=False)
        self.head.weight.data.zero_()
        self.scale = scale
        self.tta = False

    def core(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.whiten(x))
        x = self.g3(self.g2(self.g1(x)))
        x = self.pool(x).flatten(1)
        return self.head(x) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inputs arrive already standardized (verifier convention) and float32;
        # match the parameter dtype, and always hand back float32 logits.
        x = x.to(self.whiten.weight.dtype)
        if not self.tta:
            return self.core(x).float()
        direct = self.core(x) + self.core(x.flip(-1))
        h = x.size(2)
        w = x.size(3)
        p = F.pad(x, (1, 1, 1, 1), "reflect")
        a = p[:, :, 0:h, 0:w]
        b = p[:, :, 2:2 + h, 2:2 + w]
        shifted = self.core(a) + self.core(a.flip(-1)) + self.core(b) + self.core(b.flip(-1))
        return (0.5 * direct + 0.25 * shifted).float()


def init_whitening(layer: nn.Conv2d, images: torch.Tensor, eps: float = 5e-4):
    """Frozen ZCA over 2x2x3 patches: eigenvectors scaled by 1/sqrt(eigenvalue)."""
    patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
    flat = patches.reshape(len(patches), -1)
    cov = (flat.T @ flat) / len(flat)
    evals, evecs = torch.linalg.eigh(cov, UPLO="U")
    scaled = evecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(evals.view(-1, 1, 1, 1) + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled)).to(layer.weight.dtype)


def make_net(cfg, whiten_images):
    net = Net(widths=cfg["widths"], bn_mom=cfg["bn_mom"], scale=cfg["scale"]).cuda()
    init_whitening(net.whiten, whiten_images)
    net = net.to(memory_format=torch.channels_last).half()
    for mod in net.modules():  # BN in fp32: cheap, and keeps running stats accurate
        if isinstance(mod, nn.BatchNorm2d):
            mod.float()
    return net


# --------------------------------------------------------------------------- #
# data (fully GPU resident; no DataLoader in the training loop)
# --------------------------------------------------------------------------- #
def load_split(split: str):
    d = torch.load(os.path.join(CACHE, f"{split}.pt"))
    return d["x"], d["y"]


def standardize(x_uint8: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, device="cuda").view(1, 3, 1, 1) * 255.0
    std = torch.tensor(CIFAR_STD, device="cuda").view(1, 3, 1, 1) * 255.0
    return (x_uint8.cuda().float() - mean) / std


class GpuTrainData:
    """Holds the reflect-padded standardized train set and emits augmented batches.

    Augmentation per image: random +/-2px translate (one gather over a table of
    the 25 crop offsets) and random horizontal mirror (one `where`).
    """

    PAD = 2

    def __init__(self, x_uint8: torch.Tensor, y: torch.Tensor, size: int = 32):
        x = standardize(x_uint8)
        self.padded = F.pad(x, (self.PAD,) * 4, "reflect").half().contiguous()
        self.y = y.cuda()
        self.n = len(y)
        self.size = size
        side = size + 2 * self.PAD
        base = (torch.arange(size, device="cuda").view(-1, 1) * side
                + torch.arange(size, device="cuda").view(1, -1))
        offs = [dy * side + dx for dy in range(2 * self.PAD + 1) for dx in range(2 * self.PAD + 1)]
        self.crop_table = torch.stack([base.reshape(-1) + o for o in offs])  # (25, size*size)
        self.flat = self.padded.view(self.n, 3, side * side)

    def epoch(self, batch_size: int, gen: torch.Generator):
        perm = torch.randperm(self.n, device="cuda", generator=gen)
        steps = self.n // batch_size
        s2 = self.size * self.size
        for i in range(steps):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            b = idx.numel()
            choice = torch.randint(0, self.crop_table.size(0), (b,), device="cuda", generator=gen)
            crop = self.crop_table[choice].unsqueeze(1).expand(b, 3, s2)
            xb = self.flat[idx].gather(2, crop).view(b, 3, self.size, self.size)
            flip = torch.rand(b, 1, 1, 1, device="cuda", generator=gen) < 0.5
            xb = torch.where(flip, xb.flip(-1), xb)
            yield xb.contiguous(memory_format=torch.channels_last), self.y[idx]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
DEFAULT_CFG = dict(
    widths=(128, 384, 512),
    bn_mom=0.4,
    scale=1 / 9,
    epochs=20,
    batch_size=1000,
    lr=1.5,             # peak lr in "mean-loss" units
    wd=2.0e-3,          # decoupled decay per step at peak lr
    momentum=0.85,
    bias_scaler=64.0,
    label_smoothing=0.2,
    warm_frac=0.23,     # fraction of steps spent ramping up
    lr_start=0.2,       # * lr at step 0
    lr_end=0.0,         # * lr at the final step
    whiten_bias_epochs=3,
)


def build_optimizer(net, cfg, batch_size):
    lr = cfg["lr"] / batch_size          # loss uses reduction='sum'
    lr_bias = lr * cfg["bias_scaler"]
    # SGD applies lr*wd*p, so dividing by lr makes the decay lr-schedule-coupled
    # but bias-group-consistent (same decay for both groups at any point).
    norm_biases, others = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (norm_biases if "norm" in name else others).append(p)
    groups = [
        dict(params=others, lr=lr, weight_decay=cfg["wd"] / lr),
        dict(params=norm_biases, lr=lr_bias, weight_decay=cfg["wd"] / lr_bias),
    ]
    return torch.optim.SGD(groups, momentum=cfg["momentum"], nesterov=True)


def lr_factor(step: int, total: int, cfg) -> float:
    warm = max(1, int(cfg["warm_frac"] * total))
    if step < warm:
        return cfg["lr_start"] + (1.0 - cfg["lr_start"]) * step / warm
    t = (step - warm) / max(1, total - warm)
    return 1.0 + (cfg["lr_end"] - 1.0) * t


def warmup_gpu(cfg, steps: int = 60):
    """Bring SM clocks to their steady state and let cudnn.benchmark pick algos.

    The H100 idles at ~345 MHz against a 1980 MHz max, so a cold first epoch runs
    ~2x slow and the ramp bleeds into later ones.  This runs the exact training
    step on *random noise* with a throwaway network, which is then discarded --
    no weights, statistics or data from it reach the real run.  It only ensures
    that every timed epoch is measured under the same steady-state conditions.
    """
    net = Net(widths=cfg["widths"], bn_mom=cfg["bn_mom"], scale=cfg["scale"]).cuda()
    net = net.to(memory_format=torch.channels_last).half()
    for mod in net.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.float()
    bs = cfg["batch_size"]
    opt = build_optimizer(net, cfg, bs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"], reduction="sum")
    x = torch.randn(bs, 3, 32, 32, device="cuda", dtype=torch.half).contiguous(
        memory_format=torch.channels_last)
    y = torch.randint(0, 10, (bs,), device="cuda")
    for _ in range(steps):
        loss = loss_fn(net.core(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.tta = True
    net.eval()
    torch.jit.script(net)  # pay the TorchScript compile cost once, up front
    torch.cuda.synchronize()
    del net, opt, x, y


@torch.no_grad()
def evaluate(net, x_uint8, y, batch_size=1000, tta=True):
    """Accuracy on a held-out set, through the same path the verifier uses."""
    was_tta, was_train = net.tta, net.training
    net.tta = tta
    net.eval()
    x = standardize(x_uint8).contiguous(memory_format=torch.channels_last)
    y = y.cuda()
    correct = 0
    for i in range(0, len(y), batch_size):
        out = net(x[i:i + batch_size])
        correct += (out.argmax(1) == y[i:i + batch_size]).sum().item()
    net.tta = was_tta
    net.train(was_train)
    return correct / len(y)


def train(cfg, data: "GpuTrainData", whiten_images, seed=0, ckpt_dir=None,
          eval_sets=None, log=print, save_every=1):
    """One full training run. Returns (per-epoch records, net, wall_clock_sec)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed + 1234)

    net = make_net(cfg, whiten_images)
    bs = cfg["batch_size"]
    steps_per_epoch = data.n // bs
    total_steps = steps_per_epoch * cfg["epochs"]
    opt = build_optimizer(net, cfg, bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps, cfg))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"], reduction="sum")

    scripted = None
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        net.tta = True
        net.eval()
        scripted = torch.jit.script(net)  # shares parameter storage with `net`
        net.train()
        net.tta = False

    records = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        net.train()
        if epoch > cfg["whiten_bias_epochs"]:
            net.whiten.bias.requires_grad_(False)
        for xb, yb in data.epoch(bs, gen):
            out = net.core(xb)
            loss = loss_fn(out.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if ckpt_dir is not None and (epoch % save_every == 0 or epoch == cfg["epochs"]):
            net.eval()
            scripted.save(os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        rec = dict(epoch=epoch, t=elapsed)
        if eval_sets:
            for name, (ex, ey) in eval_sets.items():
                rec[name] = evaluate(net, ex, ey)
        records.append(rec)
        log(f"  epoch {epoch:3d}  t={elapsed:7.2f}s  "
            + "  ".join(f"{k}={v:.4f}" for k, v in rec.items() if k not in ("epoch", "t")))
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return records, net, wall
