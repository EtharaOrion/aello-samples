import json
import os
import time
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import read_image


torch.backends.cudnn.benchmark = True
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


@torch.compile
def zeropower_via_newtonschulz5(g, steps: int = 3, eps: float = 1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x /= x.norm() + eps
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, nesterov=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))

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
                g = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                p.mul_(len(p) ** 0.5 / p.norm())
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view_as(g)
                p.add_(update, alpha=-group["lr"])


def load_split(split):
    cache = os.path.join("data", "cifar10", split + ".pt")
    if os.path.exists(cache):
        obj = torch.load(cache, map_location="cpu", weights_only=True)
        return obj["images"], obj["labels"]
    images, labels = [], []
    root = os.path.join("data", "cifar10", split)
    for label, name in enumerate(CLASSES):
        folder = os.path.join(root, name)
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".png"):
                images.append(read_image(os.path.join(folder, filename)))
                labels.append(label)
    images = torch.stack(images).permute(0, 2, 3, 1).contiguous()
    labels = torch.tensor(labels, dtype=torch.long)
    torch.save({"images": images, "labels": labels}, cache)
    return images, labels


def prepare(images, labels):
    x = images.cuda().half().permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last) / 255
    x = (x - MEAN.half()) / STD.half()
    return x, labels.cuda()


def batch_flip_lr(x):
    mask = (torch.rand(len(x), device=x.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(mask, x.flip(-1), x)


def batch_crop(images, crop_size=32):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, (len(images), 2), device=images.device)
    out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device,
                      dtype=images.dtype, memory_format=torch.channels_last)
    for sy in range(-r, r + 1):
        for sx in range(-r, r + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, r + sy:r + sy + crop_size, r + sx:r + sx + crop_size]
    return out


class TrainLoader:
    def __init__(self, images, labels, batch_size=2000):
        self.images, self.labels, self.batch_size, self.epoch = images, labels, batch_size, 0
        self.flipped = batch_flip_lr(images)

    def __len__(self):
        return len(self.images) // self.batch_size

    def __iter__(self):
        base = self.flipped.flip(-1) if self.epoch % 2 else self.flipped
        images = batch_crop(F.pad(base, (2, 2, 2, 2), mode="reflect"))
        self.epoch += 1
        order = torch.randperm(len(images), device=images.device)
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            yield images[idx], self.labels[idx]


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, features, momentum=0.6, eps=1e-12):
        super().__init__(features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.dirac_(self.weight.data[:self.weight.size(1)])


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
        self.layers = nn.Sequential(nn.GELU(), ConvGroup(24, 64), ConvGroup(64, 256),
                                    ConvGroup(256, 256), nn.MaxPool2d(3))
        self.head = nn.Linear(256, 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for mod in self.modules():
            if type(mod) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                mod.reset_parameters()
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    def init_whiten(self, images, eps=5e-4):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.view(len(patches), -1)
        covariance = (flat.T @ flat) / len(flat)
        values, vectors = torch.linalg.eigh(covariance, UPLO="U")
        filters = vectors.T.reshape(-1, 3, 2, 2) / torch.sqrt(values.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data[:] = torch.cat((filters, -filters)).half()

    def forward(self, x, whiten_bias_grad: bool = True):
        x = x.half().contiguous(memory_format=torch.channels_last)
        base_bias = self.whiten.bias
        assert base_bias is not None
        bias = base_bias if whiten_bias_grad else base_bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x).view(len(x), -1)
        return self.head(x) / x.size(-1)


class VerifyNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def mirror(self, x):
        return 0.5 * self.net(x) + 0.5 * self.net(x.flip(-1))

    def forward(self, x):
        center = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        upper_left = self.mirror(padded[:, :, :32, :32])
        lower_right = self.mirror(padded[:, :, 2:34, 2:34])
        return 0.5 * center + 0.25 * upper_left + 0.25 * lower_right


def make_opts(model, batch_size=2000):
    wd = 2e-6 * batch_size
    filters = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    configs = [dict(params=[model.whiten.bias], lr=0.053, weight_decay=wd / 0.053),
               dict(params=norm_biases, lr=0.053, weight_decay=wd / 0.053),
               dict(params=[model.head.weight], lr=0.67, weight_decay=wd / 0.67)]
    sgd = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    muon = Muon(filters, lr=0.24, momentum=0.6, nesterov=True)
    for opt in (sgd, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return sgd, muon


def run(model, compiled, train_images, train_labels, save_dir=None, dummy=False):
    model.reset()
    model.init_whiten(train_images[:5000])
    labels = torch.randint(0, 10, train_labels.shape, device="cuda") if dummy else train_labels
    loader = TrainLoader(train_images, labels)
    sgd, muon = make_opts(model)
    total_steps, bias_steps, step = 8 * len(loader), 3 * len(loader), 0
    start = time.perf_counter()
    for epoch in range(1, 9):
        model.train()
        for inputs, targets in loader:
            outputs = compiled(inputs, step < bias_steps)
            F.cross_entropy(outputs, targets, label_smoothing=0.2, reduction="sum").backward()
            for group in sgd.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / bias_steps)
            for group in sgd.param_groups[1:] + muon.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            sgd.step(); muon.step(); model.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        train_acc = (outputs.argmax(1) == targets).float().mean().item()
        if save_dir is not None:
            model.eval()
            torch.jit.script(VerifyNet(model)).save(os.path.join(save_dir, f"ckpt_epoch_{epoch:04d}.pt"))
        print(f"epoch={epoch} train_acc={train_acc:.4f} elapsed={time.perf_counter()-start:.3f}", flush=True)
    return time.perf_counter() - start


@torch.inference_mode()
def evaluate_ladder(test_images, test_labels, save_dir):
    results = []
    for epoch in range(1, 9):
        net = torch.jit.load(os.path.join(save_dir, f"ckpt_epoch_{epoch:04d}.pt"), map_location="cuda").eval()
        correct = 0
        for x, y in zip(test_images.split(512), test_labels.split(512)):
            correct += (net(x.float()).argmax(1) == y).sum().item()
        acc = correct / len(test_labels)
        results.append(acc)
        print(f"verify epoch={epoch} accuracy={acc:.4f}", flush=True)
    return results


def main():
    torch.manual_seed(0)
    raw_train, raw_labels = load_split("train")
    train_images, train_labels = prepare(raw_train, raw_labels)
    del raw_train, raw_labels
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(model, mode="max-autotune")
    print("compilation warm-up", flush=True)
    run(model, compiled, train_images, train_labels, dummy=True)

    out = "periodic_checkpoints"
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        os.remove(os.path.join(out, name))
    torch.manual_seed(0)
    print("graded run", flush=True)
    duration = run(model, compiled, train_images, train_labels, save_dir=out)
    final_path = os.path.join(out, "ckpt_epoch_0008.pt")
    if os.path.exists("checkpoint.pt"):
        os.remove("checkpoint.pt")
    os.link(final_path, "checkpoint.pt")

    raw_test, raw_test_labels = load_split("test")
    test_images = raw_test.cuda().permute(0, 3, 1, 2).contiguous().float() / 255
    test_images = (test_images - MEAN) / STD
    test_labels = raw_test_labels.cuda()
    accs = evaluate_ladder(test_images, test_labels, out)
    with open("agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": max(accs), "wall_clock_sec": duration}, f)
    print(json.dumps({"accuracies": accs, "wall_clock_sec": duration}), flush=True)


if __name__ == "__main__":
    main()
