import os
import shutil
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)


@torch.compile
def zeropower(g: torch.Tensor, steps: int = 3):
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.24, momentum=0.6):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                update = zeropower(g.reshape(len(g), -1)).view_as(g)
                p.add_(update, alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, n):
        super().__init__(n, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 3, padding=1, bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.dirac_(self.weight.data[: self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        return self.activ(self.norm2(self.conv2(x)))


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(), ConvGroup(24, 64), ConvGroup(64, 256),
            ConvGroup(256, 256), nn.MaxPool2d(3)
        )
        self.head = nn.Linear(256, 10, bias=False)
        for m in self.modules():
            m.float() if isinstance(m, BatchNorm) else m.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    @torch.no_grad()
    def init_whiten(self, images, eps=5e-4):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.view(len(patches), -1)
        vals, vecs = torch.linalg.eigh((flat.T @ flat) / len(flat), UPLO="U")
        scaled = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data[:] = torch.cat((scaled, -scaled)).half()

    def forward(self, x, whiten_bias_grad: bool = True):
        b = self.whiten.bias
        if b is None:
            raise RuntimeError("whitening convolution requires bias")
        x = F.conv2d(x.half(), self.whiten.weight,
                     b if whiten_bias_grad else b.detach())
        x = self.layers(x).flatten(1)
        return self.head(x) / x.size(1)


class VerifierModel(nn.Module):
    """Six-view TTA; input is already CIFAR-standardized float32 NCHW."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def mirror(self, x):
        return 0.5 * self.net(x, False) + 0.5 * self.net(x.flip(-1), False)

    def forward(self, x):
        logits = self.mirror(x)
        p = F.pad(x, (1, 1, 1, 1), "reflect")
        translated = 0.5 * (self.mirror(p[:, :, :32, :32]) + self.mirror(p[:, :, 2:34, 2:34]))
        return 0.5 * (logits + translated)


def load_train():
    ds = ImageFolder("/workspace/data/cifar10/train", transform=PILToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=2000, num_workers=0,
                                         pin_memory=True)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.cuda(non_blocking=True))
        ys.append(y.cuda(non_blocking=True))
    images = torch.cat(xs).to(memory_format=torch.channels_last).half().div_(255)
    labels = torch.cat(ys)
    return (images - MEAN.half()) / STD.half(), labels


def crop(images, pad=2):
    padded = F.pad(images, (pad,) * 4, "reflect")
    shifts = torch.randint(0, 2 * pad + 1, (len(images), 2), device="cuda")
    out = torch.empty_like(images)
    for sy in range(2 * pad + 1):
        for sx in range(2 * pad + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = padded[mask, :, sy:sy + 32, sx:sx + 32]
    return out


def save_scripted(model, epoch, outdir):
    model.eval()
    wrapper = VerifierModel(model)
    scripted = torch.jit.script(wrapper)
    scripted.save(os.path.join(outdir, f"ckpt_epoch_{epoch:04d}.pt"))


def main():
    seed = int(os.environ.get("AIRBENCH_SEED", "1337"))
    torch.manual_seed(seed)
    images, labels = load_train()
    outdir = os.environ.get("AIRBENCH_OUTDIR", "/workspace/periodic_checkpoints")
    os.makedirs(outdir, exist_ok=True)
    for name in os.listdir(outdir):
        if name.startswith("ckpt_epoch_") and name.endswith(".pt"):
            os.remove(os.path.join(outdir, name))

    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")
    # Compile the inference-time (detached whitening bias) specialization before
    # checkpoint 1, so compilation cannot create a pause midway through training.
    synthetic = torch.randn(2000, 3, 32, 32, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    model(synthetic, False).sum().backward()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    del synthetic
    model.reset()
    model.init_whiten(images[:5000])

    batch_size = 2000
    batches = len(images) // batch_size
    total_steps = 8 * batches
    bias_steps = 3 * batches
    wd = 2e-6 * batch_size
    filters = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    configs = [
        dict(params=[model.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
        dict(params=norm_biases, lr=0.053, weight_decay=wd / 0.053),
        dict(params=[model.head.weight], lr=0.67, weight_decay=wd / 0.67),
    ]
    sgd = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(filters)
    for opt in (sgd, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    flip_mask = (torch.rand(len(images), device="cuda") < 0.5).view(-1, 1, 1, 1)
    base_images = torch.where(flip_mask, images.flip(-1), images)
    step = 0
    for epoch in range(1, 9):
        model.train()
        augmented = crop(base_images)
        if epoch % 2 == 0:
            augmented = augmented.flip(-1)
        order = torch.randperm(len(images), device="cuda")
        for i in range(batches):
            idx = order[i * batch_size:(i + 1) * batch_size]
            x, y = augmented[idx], labels[idx]
            logits = model(x, step < bias_steps)
            F.cross_entropy(logits, y, label_smoothing=0.2, reduction="sum").backward()
            sgd.param_groups[0]["lr"] = sgd.param_groups[0]["initial_lr"] * max(0.0, 1 - step / bias_steps)
            for group in sgd.param_groups[1:] + muon.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            sgd.step(); muon.step(); model.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        save_scripted(model, epoch, outdir)
        print(f"saved epoch {epoch}", flush=True)

    final = os.path.join(outdir, "ckpt_epoch_0008.pt")
    destination = os.environ.get("AIRBENCH_FINAL", "/workspace/checkpoint.pt")
    if os.path.exists(destination):
        os.remove(destination)
    os.link(final, destination)


if __name__ == "__main__":
    main()
