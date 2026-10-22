import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)


@torch.compile
def zeropower(g: torch.Tensor, steps: int = 3) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    return x.T if transposed else x


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
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                update = zeropower(g.reshape(len(g), -1)).reshape_as(g)
                p.add_(update, alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, n):
        super().__init__(n, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, c_in, c_out):
        super().__init__(c_in, c_out, 3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.dirac_(self.weight.data[: self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = Conv(c_in, c_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(c_out)
        self.conv2 = Conv(c_out, c_out)
        self.norm2 = BatchNorm(c_out)
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
            ConvGroup(256, 256), nn.MaxPool2d(3),
        )
        self.head = nn.Linear(256, 10, bias=False)
        for m in self.modules():
            if isinstance(m, BatchNorm):
                m.float()
            else:
                m.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    @torch.no_grad()
    def init_whiten(self, images):
        patches = images[:5000].unfold(2, 2, 1).unfold(3, 2, 1)
        patches = patches.transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.reshape(len(patches), -1)
        cov = flat.T @ flat / len(flat)
        values, vectors = torch.linalg.eigh(cov, UPLO="U")
        filters = vectors.T.reshape(-1, 3, 2, 2) / torch.sqrt(values[:, None, None, None] + 5e-4)
        self.whiten.weight.data.copy_(torch.cat((filters, -filters)).half())

    def forward(self, x, whiten_bias_grad: bool = True):
        bias = self.whiten.bias
        assert bias is not None
        bias = bias if whiten_bias_grad else bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x).flatten(1)
        return self.head(x) / x.size(-1)


class Deploy(nn.Module):
    """Verifier-facing model: standardized float32 NCHW in, class logits out."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def mirrored(self, x):
        return (self.net(x, False) + self.net(x.flip(-1), False)) * 0.5

    def forward(self, x):
        x = x.half()
        base = self.mirrored(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted_a = self.mirrored(padded[:, :, :32, :32])
        shifted_b = self.mirrored(padded[:, :, 2:34, 2:34])
        return base * 0.5 + (shifted_a + shifted_b) * 0.25


def load_split(path):
    ds = ImageFolder(path)
    convert = PILToTensor()
    images = torch.empty((len(ds), 3, 32, 32), dtype=torch.uint8)
    labels = torch.empty(len(ds), dtype=torch.long)
    for i, (image, label) in enumerate(ds):
        images[i] = convert(image)
        labels[i] = label
    images = images.cuda().half().div_(255).to(memory_format=torch.channels_last)
    images = (images - MEAN.half()) / STD.half()
    return images, labels.cuda()


def crop(images, radius=2):
    shifts = torch.randint(-radius, radius + 1, (len(images), 2), device="cuda")
    out = torch.empty((len(images), 3, 32, 32), device="cuda", dtype=images.dtype)
    for sy in range(-radius, radius + 1):
        for sx in range(-radius, radius + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, radius + sy:radius + sy + 32, radius + sx:radius + sx + 32]
    return out


def train_run(model, compiled, images, labels, out_dir=None, random_labels=False):
    batch_size = 2000
    batches = len(images) // batch_size
    total_steps = 8 * batches
    bias_steps = 3 * batches
    wd = 2e-6 * batch_size
    conv_params = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    configs = [
        dict(params=[model.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
        dict(params=norm_biases, lr=0.053, weight_decay=wd / 0.053),
        dict(params=[model.head.weight], lr=0.67, weight_decay=wd / 0.67),
    ]
    sgd = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(conv_params)
    for opt in (sgd, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    model.reset()
    model.init_whiten(images)
    use_labels = torch.randint(0, 10, labels.shape, device="cuda") if random_labels else labels
    padded = F.pad(images, (2, 2, 2, 2), "reflect")
    preflip = torch.where((torch.rand(len(images), device="cuda") < 0.5)[:, None, None, None], padded.flip(-1), padded)

    deploy = torch.jit.script(Deploy(CifarNet().cuda().to(memory_format=torch.channels_last)).eval()) if out_dir else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    step = 0
    for epoch in range(1, 9):
        model.train()
        aug = crop(preflip).flip(-1) if epoch % 2 == 0 else crop(preflip)
        order = torch.randperm(len(images), device="cuda")
        for j in range(batches):
            idx = order[j * batch_size:(j + 1) * batch_size]
            logits = compiled(aug[idx], step < bias_steps)
            F.cross_entropy(logits, use_labels[idx], label_smoothing=0.2, reduction="sum").backward()
            for group in sgd.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / bias_steps)
            for group in sgd.param_groups[1:] + muon.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            sgd.step(); muon.step(); model.zero_grad(set_to_none=True)
            step += 1
        if out_dir:
            model.eval()
            deploy.net.load_state_dict(model.state_dict())
            deploy.save(os.path.join(out_dir, f"ckpt_epoch_{epoch:04d}.pt"))
            print(f"saved epoch {epoch}", flush=True)
    return deploy


@torch.no_grad()
def evaluate(path, images, labels):
    net = torch.jit.load(path).cuda().eval()
    correct = 0
    for start in range(0, len(images), 512):
        x = images[start:start + 512].float().contiguous()
        correct += (net(x).argmax(1) == labels[start:start + 512]).sum().item()
    return correct / len(images)


def main():
    torch.manual_seed(920)
    train_images, train_labels = load_split("/workspace/data/cifar10/train")
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(model, mode="max-autotune")
    print("compile warmup", flush=True)
    train_run(model, compiled, train_images, train_labels, random_labels=True)
    torch.cuda.synchronize()
    torch.manual_seed(920)
    out_dir = "/workspace/periodic_checkpoints"
    for p in Path(out_dir).glob("ckpt_epoch_*.pt"):
        p.unlink()
    start = time.time()
    train_run(model, compiled, train_images, train_labels, out_dir=out_dir)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    final = "/workspace/checkpoint.pt"
    if os.path.exists(final):
        os.unlink(final)
    os.link(os.path.join(out_dir, "ckpt_epoch_0008.pt"), final)

    test_images, test_labels = load_split("/workspace/data/cifar10/test")
    scores = []
    for epoch in range(1, 9):
        path = os.path.join(out_dir, f"ckpt_epoch_{epoch:04d}.pt")
        score = evaluate(path, test_images, test_labels)
        scores.append(score)
        print(f"epoch={epoch} accuracy={score:.4f}", flush=True)
    with open("/workspace/agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": scores[-1], "wall_clock_sec": elapsed, "per_epoch": scores}, f)


if __name__ == "__main__":
    main()
