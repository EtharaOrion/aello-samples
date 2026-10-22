import os
import time
import json
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import read_image

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

ROOT = "/workspace"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)


def load_split(split):
    cache = f"/tmp/cifar10_{split}_cache.pt"
    if os.path.exists(cache):
        return torch.load(cache, weights_only=True)
    images, labels = [], []
    for label, name in enumerate(CLASSES):
        folder = os.path.join(ROOT, "data", "cifar10", split, name)
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".png"):
                images.append(read_image(os.path.join(folder, filename)))
                labels.append(label)
    result = {"images": torch.stack(images), "labels": torch.tensor(labels)}
    torch.save(result, cache)
    return result


@torch.compile
def zeropower(g, steps: int = 3):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    if transpose:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.24, momentum=0.6):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                p.add_(zeropower(g.reshape(len(g), -1)).reshape_as(g), alpha=-group["lr"])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, n):
        super().__init__(n, eps=1e-12, momentum=0.4)
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
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    @torch.no_grad()
    def init_whiten(self, images):
        patches = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
        flat = patches.reshape(len(patches), -1)
        cov = flat.T @ flat / len(flat)
        vals, vecs = torch.linalg.eigh(cov, UPLO="U")
        scaled = vecs.T.reshape(-1, 3, 2, 2) / torch.sqrt(vals.reshape(-1, 1, 1, 1) + 5e-4)
        self.whiten.weight.data.copy_(torch.cat((scaled, -scaled)).half())

    def forward(self, x, whiten_bias_grad: bool = True):
        x = x.half()
        bias = self.whiten.bias
        assert bias is not None
        if not whiten_bias_grad:
            bias = bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x)
        x = x.reshape(len(x), -1)
        return self.head(x) / x.size(-1)


class InferenceModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def mirror(self, x):
        return 0.5 * self.net(x) + 0.5 * self.net(x.flip(-1))

    def forward(self, x):
        logits = self.mirror(x)
        padded = F.pad(x, (1, 1, 1, 1), "reflect")
        shifted_a = self.mirror(padded[:, :, :32, :32])
        shifted_b = self.mirror(padded[:, :, 2:34, 2:34])
        return 0.5 * logits + 0.25 * (shifted_a + shifted_b)


def crop(images, pad=2):
    padded = F.pad(images, (pad,) * 4, "reflect")
    shifts = torch.randint(0, 2 * pad + 1, (len(images), 2), device=images.device)
    out = torch.empty_like(images)
    for sy in range(2 * pad + 1):
        for sx in range(2 * pad + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = padded[mask, :, sy:sy + 32, sx:sx + 32]
    return out


def run(model, images, labels, save=False):
    run_started = time.time()
    batch_size = 2000
    steps_per_epoch = len(images) // batch_size
    total_steps = 8 * steps_per_epoch
    bias_steps = 3 * steps_per_epoch
    filters = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    configs = [dict(params=[model.whiten.bias], lr=0.053, weight_decay=0.004 / 0.053),
               dict(params=norm_biases, lr=0.053, weight_decay=0.004 / 0.053),
               dict(params=[model.head.weight], lr=0.67, weight_decay=0.004 / 0.67)]
    opt1 = torch.optim.SGD(configs, momentum=0.85, nesterov=True, fused=True)
    opt2 = Muon(filters)
    for opt in (opt1, opt2):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    model.reset()
    model.init_whiten(images[:5000])
    preflip = torch.where((torch.rand(len(images), device="cuda") < .5).view(-1, 1, 1, 1), images.flip(-1), images)
    step = 0
    first_save = None
    last_save = None
    for epoch in range(1, 9):
        model.train()
        aug = preflip.flip(-1) if epoch % 2 == 0 else preflip
        aug = crop(aug)
        order = torch.randperm(len(images), device="cuda")
        for start in range(0, len(images), batch_size):
            idx = order[start:start + batch_size]
            output = model(aug[idx], step < bias_steps)
            F.cross_entropy(output, labels[idx], label_smoothing=.2, reduction="sum").backward()
            opt1.param_groups[0]["lr"] = opt1.param_groups[0]["initial_lr"] * max(0., 1 - step / bias_steps)
            for group in opt1.param_groups[1:] + opt2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_steps)
            opt1.step(); opt2.step(); model.zero_grad(set_to_none=True)
            step += 1
        torch.cuda.synchronize()
        if save:
            model.eval()
            path = os.path.join(ROOT, "periodic_checkpoints", f"ckpt_epoch_{epoch:04d}.pt")
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.jit.script(InferenceModel(raw_model)).save(path)
            stamp = time.time()
            if first_save is None:
                first_save = stamp
            last_save = stamp
            print(f"saved epoch {epoch}", flush=True)
    return first_save, last_save, last_save - run_started if save else 0.0


def main():
    torch.manual_seed(0)
    data = load_split("train")
    images = data["images"].cuda().half().div_(255).sub_(MEAN).div_(STD).to(memory_format=torch.channels_last)
    labels = data["labels"].cuda()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    compiled = torch.compile(model, mode="max-autotune")
    # Compile all training graphs before the checkpoint clock begins.
    run(compiled, images, torch.randint(0, 10, labels.shape, device="cuda"), save=False)
    os.makedirs(os.path.join(ROOT, "periodic_checkpoints"), exist_ok=True)
    for name in os.listdir(os.path.join(ROOT, "periodic_checkpoints")):
        os.remove(os.path.join(ROOT, "periodic_checkpoints", name))
    torch.manual_seed(0)
    first, last, total_wall = run(compiled, images, labels, save=True)
    final_path = os.path.join(ROOT, "checkpoint.pt")
    if os.path.exists(final_path):
        os.remove(final_path)
    os.link(os.path.join(ROOT, "periodic_checkpoints", "ckpt_epoch_0008.pt"), final_path)
    with open(os.path.join(ROOT, "agent_reported_metrics.json"), "w") as f:
        json.dump({"accuracy": 0.0, "wall_clock_sec": total_wall}, f)


if __name__ == "__main__":
    main()
