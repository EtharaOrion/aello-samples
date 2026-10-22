import argparse
import json
import math
import os
import time

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

ROOT = "/workspace/data/cifar10"
MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
STD = torch.tensor((0.2470, 0.2435, 0.2616))


def make_tensor_file(split):
    destination = os.path.join(ROOT, split + ".pt")
    if os.path.exists(destination):
        return
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(ROOT, split), transform=transforms.PILToTensor())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, shuffle=False, num_workers=8,
        persistent_workers=True)
    image_parts, label_parts = [], []
    for images, labels in loader:
        image_parts.append(images.permute(0, 2, 3, 1).contiguous())
        label_parts.append(labels)
    torch.save({"images": torch.cat(image_parts),
                "labels": torch.cat(label_parts),
                "classes": dataset.classes}, destination)


def flip_batch(x):
    mask = (torch.rand(len(x), device=x.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(mask, x.flip(-1), x)


def crop_batch(images, size):
    radius = (images.size(-1) - size) // 2
    shifts = torch.randint(-radius, radius + 1, (len(images), 2), device=images.device)
    out = torch.empty((len(images), 3, size, size), device=images.device,
                      dtype=images.dtype, memory_format=torch.channels_last)
    for sy in range(-radius, radius + 1):
        for sx in range(-radius, radius + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, radius+sy:radius+sy+size,
                               radius+sx:radius+sx+size]
    return out


class TrainLoader:
    def __init__(self, batch_size=2000):
        packed = torch.load(os.path.join(ROOT, "train.pt"), map_location="cuda")
        raw = packed["images"].half().div_(255).permute(0, 3, 1, 2)
        raw = raw.contiguous(memory_format=torch.channels_last)
        mean = MEAN.cuda().half().view(1, 3, 1, 1)
        std = STD.cuda().half().view(1, 3, 1, 1)
        normalized = (raw - mean) / std
        preflipped = flip_batch(normalized)
        self.padded = F.pad(preflipped, (2, 2, 2, 2), "reflect")
        self.labels = packed["labels"]
        self.batch_size = batch_size
        self.epoch = 0

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __iter__(self):
        images = crop_batch(self.padded, 32)
        if self.epoch % 2:
            images = images.flip(-1)
        self.epoch += 1
        order = torch.randperm(len(images), device=images.device)
        for pos in range(0, self.batch_size * len(self), self.batch_size):
            idx = order[pos:pos+self.batch_size]
            yield images[idx], self.labels[idx]


@torch.compile
def zeropower(gradient, steps=3, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = gradient.bfloat16()
    x = x / (x.norm() + eps)
    transposed = gradient.size(0) > gradient.size(1)
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
    def __init__(self, params, lr, momentum, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(grad)
                update_grad = grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                parameter.mul_(len(parameter) ** 0.5 / parameter.norm())
                update = zeropower(update_grad.reshape(len(update_grad), -1)).reshape_as(parameter)
                parameter.add_(update, alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, channels):
        super().__init__(channels, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.dirac_(self.weight.data[:self.weight.size(1)])


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
            ConvGroup(256, 256), nn.MaxPool2d(3))
        self.head = nn.Linear(256, 10, bias=False)
        for module in self.modules():
            if isinstance(module, BatchNorm):
                module.float()
            else:
                module.half()

    def reset(self):
        for module in self.modules():
            if type(module) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                module.reset_parameters()
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    @torch.no_grad()
    def init_whiten(self, images, eps=5e-4):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1)
        patches = patches.transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.reshape(len(patches), -1)
        covariance = flat.T @ flat / len(flat)
        values, vectors = torch.linalg.eigh(covariance, UPLO="U")
        filters = vectors.T.reshape(-1, 3, 2, 2) / torch.sqrt(values[:, None, None, None] + eps)
        self.whiten.weight.copy_(torch.cat((filters, -filters)).half())

    def forward(self, x, whiten_bias_grad: bool = True):
        bias = self.whiten.bias
        assert bias is not None
        bias = bias if whiten_bias_grad else bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x).view(len(x), -1)
        return self.head(x) / x.size(-1)


class VerifierModel(nn.Module):
    """Single-call six-view TTA; input is already CIFAR standardized float32."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def mirror(self, x):
        return 0.5 * self.net(x) + 0.5 * self.net(x.flip(-1))

    def forward(self, x):
        x = x.half().contiguous(memory_format=torch.channels_last)
        center = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        diagonal = 0.5 * (self.mirror(padded[:, :, :32, :32]) +
                          self.mirror(padded[:, :, 2:34, 2:34]))
        return 0.5 * center + 0.5 * diagonal


def save_checkpoint(net, path):
    exported_net = CifarNet().cuda().to(memory_format=torch.channels_last)
    exported_net.load_state_dict(net.state_dict())
    exported_net.eval()
    wrapper = VerifierModel(exported_net).eval()
    torch.jit.script(wrapper).save(path)
    del wrapper, exported_net


def test_accuracy(path):
    packed = torch.load(os.path.join(ROOT, "test.pt"), map_location="cuda")
    images = packed["images"].float().div_(255).permute(0, 3, 1, 2)
    images = (images - MEAN.cuda()[None, :, None, None]) / STD.cuda()[None, :, None, None]
    labels = packed["labels"]
    model = torch.jit.load(path).cuda().eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(images), 512):
            logits = model(images[start:start+512])
            correct += (logits.argmax(1) == labels[start:start+512]).sum().item()
    return correct / len(labels)


def train(output_dir, seed):
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(output_dir):
        if name.startswith("ckpt_epoch_") and name.endswith(".pt"):
            os.remove(os.path.join(output_dir, name))

    loader = TrainLoader(2000)
    net = CifarNet().cuda().to(memory_format=torch.channels_last)
    net.reset()
    compiled = torch.compile(net, mode="max-autotune")
    conv_params = [p for p in net.parameters() if p.requires_grad and p.ndim == 4]
    norm_biases = [p for n, p in net.named_parameters() if "norm" in n and p.requires_grad]
    wd = 2e-6 * loader.batch_size
    sgd = torch.optim.SGD([
        {"params": [net.whiten.bias], "lr": 0.053, "weight_decay": wd / 0.053},
        {"params": norm_biases, "lr": 0.053, "weight_decay": wd / 0.053},
        {"params": [net.head.weight], "lr": 0.67, "weight_decay": wd / 0.67},
    ], momentum=0.85, nesterov=True, fused=True)
    muon = Muon(conv_params, lr=0.24, momentum=0.6, nesterov=True)
    for optimizer in (sgd, muon):
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

    raw = loader.padded[:5000, :, 2:34, 2:34]
    net.init_whiten(raw)
    total_steps = 8 * len(loader)
    bias_steps = 3 * len(loader)
    step = 0
    start_time = time.monotonic()
    for epoch in range(1, 9):
        net.train()
        for inputs, labels in loader:
            logits = compiled(inputs, whiten_bias_grad=(step < bias_steps))
            F.cross_entropy(logits, labels, label_smoothing=0.2, reduction="sum").backward()
            sgd.param_groups[0]["lr"] = sgd.param_groups[0]["initial_lr"] * max(0.0, 1-step/bias_steps)
            for group in sgd.param_groups[1:] + muon.param_groups:
                group["lr"] = group["initial_lr"] * (1-step/total_steps)
            sgd.step()
            muon.step()
            net.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        path = os.path.join(output_dir, f"ckpt_epoch_{epoch:04d}.pt")
        save_checkpoint(net, path)
        print(f"saved epoch={epoch} elapsed={time.monotonic()-start_time:.3f}s", flush=True)
    return time.monotonic() - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/workspace/periodic_checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    make_tensor_file("train")
    make_tensor_file("test")
    duration = train(args.output, args.seed)
    scores = []
    for epoch in range(1, 9):
        path = os.path.join(args.output, f"ckpt_epoch_{epoch:04d}.pt")
        score = test_accuracy(path)
        scores.append(score)
        print(f"epoch={epoch} exact_accuracy={score:.4f}", flush=True)
    final_source = os.path.join(args.output, "ckpt_epoch_0008.pt")
    final_path = "/workspace/checkpoint.pt"
    if os.path.exists(final_path):
        os.remove(final_path)
    os.link(final_source, final_path)
    with open("/workspace/agent_reported_metrics.json", "w") as handle:
        json.dump({"accuracy": scores[-1], "wall_clock_sec": duration,
                   "first_crossing_epoch": next((i+1 for i, x in enumerate(scores) if x >= 0.94), None)}, handle)
