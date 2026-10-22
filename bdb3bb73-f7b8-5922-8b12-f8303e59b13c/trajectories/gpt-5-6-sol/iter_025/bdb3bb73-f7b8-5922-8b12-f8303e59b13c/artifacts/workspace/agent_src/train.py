import math
import os
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

ROOT = "/workspace"
DATA = os.path.join(ROOT, "data", "cifar10")
OUT = os.environ.get("CHECKPOINT_DIR", os.path.join(ROOT, "periodic_checkpoints"))
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "9"))
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda", dtype=torch.float16).view(1, 3, 1, 1)


@torch.compile
def zeropower(g: torch.Tensor, steps: int = 3) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = g.size(0) > g.size(1)
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
                update = zeropower(g.reshape(len(g), -1)).reshape_as(g)
                p.add_(update, alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, n):
        super().__init__(n, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, ni, no):
        super().__init__(ni, no, 3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.dirac_(self.weight.data[:self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.conv1 = Conv(ni, no)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(no)
        self.conv2 = Conv(no, no)
        self.norm2 = BatchNorm(no)
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
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1)
        patches = patches.transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.reshape(len(patches), -1)
        cov = flat.T @ flat / len(flat)
        vals, vecs = torch.linalg.eigh(cov, UPLO="U")
        filters = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.view(-1, 1, 1, 1) + 5e-4)
        self.whiten.weight.copy_(torch.cat((filters, -filters)).half())

    def forward(self, x: torch.Tensor, whiten_bias_grad: bool = False) -> torch.Tensor:
        bias = self.whiten.bias
        assert bias is not None
        b = bias if whiten_bias_grad else bias.detach()
        x = F.conv2d(x, self.whiten.weight, b)
        x = self.layers(x)
        x = x.reshape(len(x), -1)
        return self.head(x) / x.size(-1)


class InferenceModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def basic(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x, False)

    def mirror(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.basic(x) + 0.5 * self.basic(x.flip(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half().contiguous(memory_format=torch.channels_last)
        logits = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted = 0.5 * self.mirror(padded[:, :, :32, :32])
        shifted = shifted + 0.5 * self.mirror(padded[:, :, 2:34, 2:34])
        return 0.5 * logits + 0.5 * shifted


def load_split(split):
    images, labels = [], []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATA, split, cls)
        for name in sorted(os.listdir(folder)):
            with Image.open(os.path.join(folder, name)) as im:
                images.append(np.asarray(im, dtype=np.uint8).copy())
            labels.append(label)
    arr = np.stack(images)
    x = torch.from_numpy(arr).cuda().permute(0, 3, 1, 2)
    x = (x.half() / 255).contiguous(memory_format=torch.channels_last)
    y = torch.tensor(labels, device="cuda", dtype=torch.long)
    return x, y


def normalize(x):
    return (x - MEAN) / STD


def crop(images, pad=2):
    n = len(images)
    shifts = torch.randint(-pad, pad + 1, (n, 2), device="cuda")
    out = torch.empty((n, 3, 32, 32), device="cuda", dtype=images.dtype,
                      memory_format=torch.channels_last)
    for sy in range(-pad, pad + 1):
        for sx in range(-pad, pad + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, pad + sy:pad + sy + 32, pad + sx:pad + sx + 32]
    return out


def make_optimizers(model, batch_size):
    wd = 2e-6 * batch_size
    filters = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    configs = [dict(params=[model.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
               dict(params=norm_biases, lr=0.053, weight_decay=wd / 0.053),
               dict(params=[model.head.weight], lr=0.67, weight_decay=wd / 0.67)]
    opt1 = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    opt2 = Muon(filters)
    for opt in (opt1, opt2):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return opt1, opt2


def train_run(model, compiled, train_x, train_y, save=False):
    batch_size = 2000
    batches = len(train_x) // batch_size
    total_steps = EPOCHS * batches
    bias_steps = 3 * batches
    model.reset()
    model.init_whiten(normalize(train_x[:5000]))
    opt1, opt2 = make_optimizers(model, batch_size)
    padded = F.pad(normalize(train_x), (2, 2, 2, 2), "reflect")
    preflip = torch.where((torch.rand(len(train_x), device="cuda") < 0.5).view(-1, 1, 1, 1),
                          padded.flip(-1), padded)
    step = 0
    if save:
        os.makedirs(OUT, exist_ok=True)
        for name in os.listdir(OUT):
            os.remove(os.path.join(OUT, name))
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_images = crop(preflip.flip(-1) if epoch % 2 == 0 else preflip)
        order = torch.randperm(len(train_x), device="cuda")
        for i in range(batches):
            idx = order[i * batch_size:(i + 1) * batch_size]
            output = compiled(epoch_images[idx], step < bias_steps)
            F.cross_entropy(output, train_y[idx], label_smoothing=0.2, reduction="sum").backward()
            opt1.param_groups[0]["lr"] = opt1.param_groups[0]["initial_lr"] * max(0.0, 1 - step / bias_steps)
            for group in opt1.param_groups[1:] + opt2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            opt1.step()
            opt2.step()
            model.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        if save:
            model.eval()
            scripted = torch.jit.script(InferenceModel(model))
            path = os.path.join(OUT, f"ckpt_epoch_{epoch:04d}.pt")
            torch.jit.save(scripted, path)
            print(f"saved epoch {epoch} at {time.time():.6f}", flush=True)
    return model


@torch.no_grad()
def evaluate_file(path, test_x, test_y):
    model = torch.jit.load(path).cuda().eval()
    x = normalize(test_x).float()
    correct = 0
    for i in range(0, len(x), 512):
        correct += (model(x[i:i + 512]).argmax(1) == test_y[i:i + 512]).sum().item()
    return correct / len(x)


def main():
    train_x, train_y = load_split("train")
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(model, mode="max-autotune")
    print("warming compiled training graph", flush=True)
    dummy_labels = torch.randint(0, 10, train_y.shape, device="cuda")
    train_run(model, compiled, train_x, dummy_labels, save=False)
    torch.cuda.synchronize()
    print("starting graded run", flush=True)
    start = time.time()
    train_run(model, compiled, train_x, train_y, save=True)
    end = time.time()
    final_path = os.path.join(OUT, f"ckpt_epoch_{EPOCHS:04d}.pt")
    if os.path.abspath(OUT) == os.path.join(ROOT, "periodic_checkpoints"):
        shutil.copy(final_path, os.path.join(ROOT, "checkpoint.pt"))
    test_x, test_y = load_split("test")
    for epoch in range(1, EPOCHS + 1):
        path = os.path.join(OUT, f"ckpt_epoch_{epoch:04d}.pt")
        print(f"epoch {epoch}: accuracy={evaluate_file(path, test_x, test_y):.4f}", flush=True)
    print(f"training_wall={end - start:.3f}", flush=True)


if __name__ == "__main__":
    main()
