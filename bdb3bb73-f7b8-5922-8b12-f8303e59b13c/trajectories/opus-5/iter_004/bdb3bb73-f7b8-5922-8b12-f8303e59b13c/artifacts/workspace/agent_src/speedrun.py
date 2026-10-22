"""CIFAR-10 -> 94% speedrun core library (single H100, from scratch).

Design notes
------------
* Architecture is in the "airbench" lineage: a frozen 2x2 whitening convolution
  (initialised from the training-set patch covariance) followed by three
  conv/pool/BN/GELU groups and a bias-free linear head with a fixed logit scale.
* The whole training split lives on the GPU as uint8; standardisation and
  augmentation (random translate / horizontal flip / optional cutout) run on
  device, so there is no host->device traffic in the training loop.
* The training loop performs *no* evaluation of any kind. Every accuracy number
  in this project is produced afterwards, offline, from the saved checkpoints:
  model-selection numbers come from a held-out slice of the *train* split, and
  test numbers come from `verify_ladder.py` replaying the verifier's exact
  pipeline. The test split is never seen by a training process.
* Checkpoints are exported through `DeployNet`, a thin wrapper that harmonises
  the input range, runs mirror+translate TTA, and returns float32 logits. It is
  `torch.jit.script`-able, which is what the verifier loads.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_NPY = "/workspace/data_npy"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_uint8(split, device="cuda"):
    """Return (N,3,32,32) uint8 images and (N,) int64 labels, resident on `device`."""
    x = np.load(f"{DATA_NPY}/{split}_x.npy")  # N,32,32,3
    y = np.load(f"{DATA_NPY}/{split}_y.npy")
    x = torch.from_numpy(x).to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return x, y


def standardize(x_uint8, dtype=torch.float16):
    """uint8 [0,255] -> per-channel standardized tensor, matching the verifier transform."""
    mean = torch.tensor(CIFAR_MEAN, device=x_uint8.device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x_uint8.device, dtype=dtype).view(1, 3, 1, 1)
    return x_uint8.to(dtype).div_(255.0).sub_(mean).div_(std)


class GPUTrainData:
    """Augmenting batch source. Holds one reflect-padded fp16 copy of the split."""

    def __init__(self, x_uint8, y, pad=2, dtype=torch.float16):
        self.y = y
        self.n = x_uint8.size(0)
        self.pad = pad
        img = standardize(x_uint8, dtype)
        self.padded = F.pad(img, (pad,) * 4, mode="reflect") if pad else img
        self.side = 32
        self._ar = torch.arange(32, device=img.device)

    def epoch(self, batch_size, generator, flip=True, cutout=0):
        """Yield (x, y) batches for one genuine single pass over the split."""
        order = torch.randperm(self.n, device=self.y.device, generator=generator)
        for i in range(0, self.n - batch_size + 1, batch_size):  # drop last partial batch
            idx = order[i : i + batch_size]
            yield self._make_batch(idx, generator, flip, cutout), self.y[idx]

    def _make_batch(self, idx, generator, flip, cutout):
        b = idx.numel()
        x = self.padded[idx]
        if self.pad:
            p, s = self.pad, self.side
            off = torch.randint(0, 2 * p + 1, (2, b), device=x.device, generator=generator)
            rows = off[0].view(b, 1) + self._ar.view(1, s)
            cols = off[1].view(b, 1) + self._ar.view(1, s)
            x = x.gather(2, rows.view(b, 1, s, 1).expand(b, 3, s, x.size(3)))
            x = x.gather(3, cols.view(b, 1, 1, s).expand(b, 3, s, s))
        if flip:
            m = torch.rand(b, 1, 1, 1, device=x.device, generator=generator) < 0.5
            x = torch.where(m, x.flip(-1), x)
        if cutout:
            x = apply_cutout(x, cutout, generator)
        return x


def apply_cutout(x, size, generator):
    b, s = x.size(0), x.size(-1)
    cy = torch.randint(0, s, (b, 1, 1), device=x.device, generator=generator)
    cx = torch.randint(0, s, (b, 1, 1), device=x.device, generator=generator)
    ar = torch.arange(s, device=x.device)
    r = size // 2
    my = (ar.view(1, s, 1) - cy).abs() < r
    mx = (ar.view(1, 1, s) - cx).abs() < r
    return x.masked_fill((my & mx).unsqueeze(1), 0.0)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class BatchNorm(nn.BatchNorm2d):
    """BN with a frozen unit scale (GELU+conv can absorb it) and a trainable shift."""

    def __init__(self, c, momentum=0.6, eps=1e-12):
        super().__init__(c, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout, k=3, padding="same", bias=False):
        super().__init__(cin, cout, kernel_size=k, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])  # partial identity init speeds early progress


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, bn_momentum=0.6):
        super().__init__()
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout, bn_momentum)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout, bn_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x = self.activ(self.norm2(self.conv2(x)))
        return x


class Net(nn.Module):
    """Core classifier. Consumes *already standardized* NCHW input, returns logits."""

    def __init__(self, widths=(64, 256, 256), logit_scale=1 / 9, whiten_k=2, bn_momentum=0.6):
        super().__init__()
        ww = 2 * 3 * whiten_k * whiten_k
        self.whiten = Conv(3, ww, k=whiten_k, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.activ = nn.GELU()
        self.block1 = ConvGroup(ww, widths[0], bn_momentum)
        self.block2 = ConvGroup(widths[0], widths[1], bn_momentum)
        self.block3 = ConvGroup(widths[1], widths[2], bn_momentum)
        self.pool = nn.MaxPool2d(3)
        self.head = nn.Linear(widths[2], 10, bias=False)
        self.head.weight.data.zero_()
        self.logit_scale = logit_scale

    def forward(self, x):
        x = self.activ(self.whiten(x))
        x = self.block3(self.block2(self.block1(x)))
        x = self.pool(x).flatten(1)
        return self.head(x) * self.logit_scale


@torch.no_grad()
def init_whitening(conv, images, eps=5e-4):
    """Set the frozen first conv to the ZCA filters of the training-set 2x2 patches."""
    c, h, w = conv.weight.shape[1:]
    patches = images.unfold(2, h, 1).unfold(3, w, 1)
    patches = patches.transpose(1, 3).reshape(-1, c, h, w).float()
    cov = torch.cov(patches.reshape(patches.size(0), -1).T)
    eigval, eigvec = torch.linalg.eigh(cov, UPLO="U")
    filt = eigvec.T.reshape(-1, c, h, w) / torch.sqrt(eigval.view(-1, 1, 1, 1) + eps)
    conv.weight.data[:] = torch.cat((filt, -filt)).to(conv.weight.dtype)
    conv.bias.data.zero_()


# --------------------------------------------------------------------------- #
# deployment wrapper (this is what gets scripted and saved)
# --------------------------------------------------------------------------- #
class DeployNet(nn.Module):
    """Inference wrapper: input-range harmonisation + mirror/translate TTA.

    The verifier feeds float32 NCHW batches that are *already* standardized with
    (0.4914,0.4822,0.4465)/(0.2470,0.2435,0.2616); that is the pass-through path.
    For robustness we also detect raw [0,1] and [0,255] batches and standardize
    them, so the checkpoint scores the same under any of the three conventions.
    """

    def __init__(self, core: nn.Module, tta: int = 2):
        super().__init__()
        self.core = core
        self.tta = tta
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    def _to_standard(self, x: torch.Tensor) -> torch.Tensor:
        hi = float(x.max())
        lo = float(x.min())
        if hi > 3.0:  # [0,255]
            return (x / 255.0 - self.mean) / self.std
        if lo >= -0.02 and hi <= 1.02:  # [0,1]
            return (x - self.mean) / self.std
        return x  # already standardized -- do not touch

    def _mirror(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x) + self.core(x.flip(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_standard(x.float())
        if self.tta == 0:
            return self.core(x)
        logits = self._mirror(x)
        if self.tta == 1:
            return logits
        p = F.pad(x, (1, 1, 1, 1), mode="reflect")
        shifted = 0.5 * (self._mirror(p[:, :, 0:32, 0:32]) + self._mirror(p[:, :, 2:34, 2:34]))
        return 0.5 * logits + 0.5 * shifted


def export_scripted(core: nn.Module, tta: int, device="cuda"):
    """Return a scripted fp32 DeployNet holding a snapshot of `core`'s weights."""
    import copy

    dep = DeployNet(copy.deepcopy(core).float().eval(), tta=tta).to(device).eval()
    return torch.jit.script(dep)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    epochs=24,
    batch_size=1024,
    lr=11.5,
    momentum=0.85,
    wd=0.0153,
    bias_scaler=64.0,
    label_smoothing=0.2,
    widths=(64, 256, 256),
    logit_scale=1 / 9,
    pad=2,
    flip=True,
    cutout=0,
    tta=2,
    peak_frac=0.23,
    lr_start=0.2,
    lr_end=0.07,
    whiten_k=2,
    bn_momentum=0.6,  # fraction of the running BN stat replaced per step
    whiten_init="zca",  # 'zca' | 'random' (frozen random) | 'learned'
    seed=0,
)


def make_schedule(total_steps, peak_frac, lr_start, lr_end):
    xs = np.arange(total_steps + 1)
    return np.interp(xs, [0, int(peak_frac * total_steps), total_steps], [lr_start, 1.0, lr_end])


def train(cfg, train_x, train_y, ckpt_dir=None, log=print, t0=None, eval_hook=None):
    """Run one full training job. Returns (core_net, stats dict).

    If `ckpt_dir` is given, a scripted DeployNet is written after *every* epoch as
    ckpt_epoch_{NNNN}.pt (contiguous 1..epochs), which is the ladder the verifier
    reads. Save cost is deliberately kept inside the timed region.
    """
    c = dict(DEFAULTS)
    c.update(cfg)
    dev = train_x.device
    torch.manual_seed(c["seed"])
    torch.backends.cudnn.benchmark = True
    gen = torch.Generator(device=dev)
    gen.manual_seed(c["seed"] + 1234)

    data = GPUTrainData(train_x, train_y, pad=c["pad"])
    net = Net(tuple(c["widths"]), c["logit_scale"], c["whiten_k"], c["bn_momentum"]).to(dev).to(torch.float16)
    net.head.weight.data.zero_()
    if c["whiten_init"] == "zca":
        with torch.no_grad():
            sample = standardize(train_x[:5000].clone(), torch.float16)
            init_whitening(net.whiten, sample)
    elif c["whiten_init"] == "learned":  # random init, but let SGD train the filters
        net.whiten.weight.requires_grad = True

    # two param groups: decayed weights, and 0-wd biases/BN-shifts on a boosted LR
    biases = [p for n, p in net.named_parameters() if p.requires_grad and p.ndim <= 1]
    weights = [p for n, p in net.named_parameters() if p.requires_grad and p.ndim > 1]
    steps_per_epoch = data.n // c["batch_size"]
    total_steps = steps_per_epoch * c["epochs"]

    # LR/WD normalisation borrowed from the airbench recipe so that the tuned
    # constants stay meaningful when batch size or momentum change.
    # `lr` is the peak; the LambdaLR below supplies the 0.2 -> 1.0 -> lr_end shape.
    kilostep = c["batch_size"] * (1 + 1 / (1 - c["momentum"]))
    lr = c["lr"] / kilostep
    wd = c["wd"] * c["batch_size"] / kilostep
    opt = torch.optim.SGD(
        [
            dict(params=weights, lr=lr, weight_decay=wd / lr),
            dict(params=biases, lr=lr * c["bias_scaler"], weight_decay=wd / (lr * c["bias_scaler"])),
        ],
        momentum=c["momentum"],
        nesterov=True,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s, t=make_schedule(total_steps, c["peak_frac"], c["lr_start"], c["lr_end"]): t[min(s, len(t) - 1)]
    )
    lossfn = nn.CrossEntropyLoss(label_smoothing=c["label_smoothing"], reduction="sum")

    scripted = None
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        scripted = export_scripted(net, c["tta"], dev)  # script once; reuse each epoch

    if t0 is None:
        t0 = time.time()
    save_times = []
    for ep in range(1, c["epochs"] + 1):
        net.train()
        for xb, yb in data.epoch(c["batch_size"], gen, c["flip"], c["cutout"]):
            out = net(xb)
            loss = lossfn(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if ckpt_dir is not None:
            net.eval()
            sd = {k: v.float() for k, v in net.state_dict().items()}
            scripted.core.load_state_dict(sd)
            scripted.save(f"{ckpt_dir}/ckpt_epoch_{ep:04d}.pt")
            save_times.append(time.time())
        if eval_hook is not None:  # offline experiments only; never set for the graded run
            eval_hook(ep, net)
    torch.cuda.synchronize()
    total = time.time() - t0
    log(f"[train] {c['epochs']} epochs in {total:.2f}s")
    return net, dict(cfg=c, total_sec=total, save_times=save_times, t0=t0)


# --------------------------------------------------------------------------- #
# offline evaluation helpers (never used inside the training loop)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_module(module, x_uint8, y, batch=512, tta=None):
    """Accuracy of an nn.Module / ScriptModule on standardized float32 batches."""
    module.eval()
    correct = 0
    for i in range(0, x_uint8.size(0), batch):
        xb = standardize(x_uint8[i : i + batch].clone(), torch.float32)
        out = module(xb) if tta is None else tta(module, xb)
        correct += (out.argmax(1) == y[i : i + batch]).sum().item()
    return correct / x_uint8.size(0)
