import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path('/workspace')
DATA = ROOT / 'data' / 'cifar10'
OUT = ROOT / 'periodic_checkpoints'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device='cuda').view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device='cuda').view(1, 3, 1, 1)


def load_split(name):
    images, labels = [], []
    for label, cls in enumerate(CLASSES):
        for path in sorted((DATA / name / cls).glob('*.png')):
            with Image.open(path) as im:
                images.append(np.asarray(im.convert('RGB'), dtype=np.uint8).copy())
            labels.append(label)
    x = torch.from_numpy(np.stack(images)).cuda().permute(0, 3, 1, 2)
    y = torch.tensor(labels, device='cuda')
    return x, y


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mul(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, channels):
        super().__init__(channels, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, ci, co, kernel_size=3, padding=1, bias=False):
        super().__init__(ci, co, kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        nn.init.dirac_(self.weight.data[:self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv1 = Conv(ci, co)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(co)
        self.conv2 = Conv(co, co)
        self.norm2 = BatchNorm(co)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        return self.activ(self.norm2(self.conv2(x)))


class AirBench94(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Conv(3, 24, 2, padding=0, bias=True), nn.GELU(),
            ConvGroup(24, 64), ConvGroup(64, 256), ConvGroup(256, 256),
            nn.MaxPool2d(3), Flatten(), nn.Linear(256, 10, bias=False), Mul(1 / 9),
        )
        self.layers[0].weight.requires_grad = False

    def forward(self, x):
        return self.layers(x)


class Deploy(nn.Module):
    """Verifier-facing model: standardized float32 NCHW in, class logits out."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def mirrored(self, x):
        return (self.net(x) + self.net(x.flip(-1))) * 0.5

    def forward(self, x):
        x = x.to(dtype=torch.float16, memory_format=torch.channels_last)
        base = self.mirrored(x)
        p = F.pad(x, (1, 1, 1, 1), 'reflect')
        shifted = (self.mirrored(p[:, :, :32, :32]) + self.mirrored(p[:, :, 2:34, 2:34])) * 0.5
        return (base + shifted) * 0.5


def whitening_init(layer, train_x):
    x = train_x[:5000].float() / 255
    x = (x - MEAN) / STD
    p = x.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2).float()
    flat = p.view(len(p), -1)
    vals, vecs = torch.linalg.eigh(flat.T @ flat / len(flat), UPLO='U')
    vals = vals.flip(0).view(-1, 1, 1, 1)
    vecs = vecs.T.reshape(12, 3, 2, 2).flip(0)
    scaled = vecs / torch.sqrt(vals + 5e-4)
    layer.weight.data[:] = torch.cat((scaled, -scaled)).half()


def crop_batch(padded):
    n = len(padded)
    shifts = torch.randint(-2, 3, (n, 2), device='cuda')
    out = torch.empty((n, 3, 32, 32), device='cuda', dtype=padded.dtype,
                      memory_format=torch.channels_last)
    for sy in range(-2, 3):
        for sx in range(-2, 3):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = padded[mask, :, 2 + sy:34 + sy, 2 + sx:34 + sx]
    return out


class Lookahead:
    def __init__(self, net):
        self.ema = {k: v.clone() for k, v in net.state_dict().items()}

    @torch.no_grad()
    def update(self, net, decay):
        for a, b in zip(self.ema.values(), net.state_dict().values()):
            if b.dtype in (torch.float16, torch.float32):
                a.lerp_(b, 1 - decay)
                b.copy_(a)


def save_scripted(net, epoch):
    was_training = net.training
    net.eval()
    scripted = torch.jit.script(Deploy(net))
    scripted.save(str(OUT / f'ckpt_epoch_{epoch:04d}.pt'))
    if was_training:
        net.train()


def main():
    torch.manual_seed(747)
    torch.cuda.manual_seed_all(747)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    train_u8, train_y = load_split('train')
    print('loaded', train_u8.shape, flush=True)
    train_x = ((train_u8.float() / 255 - MEAN) / STD).half()
    flip_mask = torch.rand(len(train_x), device='cuda') < 0.5
    train_x[flip_mask] = train_x[flip_mask].flip(-1)
    padded = F.pad(train_x, (2, 2, 2, 2), 'reflect')

    net = AirBench94().cuda().half().to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    whitening_init(net.layers[0], train_u8)

    batch_size = 1024
    epochs = 12
    steps_per_epoch = len(train_x) // batch_size
    total_steps = epochs * steps_per_epoch
    momentum = 0.85
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = 11.5 / kilostep_scale
    wd = 0.0153 * batch_size / kilostep_scale
    norm_biases = [p for k, p in net.named_parameters() if 'norm' in k and p.requires_grad]
    other = [p for k, p in net.named_parameters() if 'norm' not in k and p.requires_grad]
    opt = torch.optim.SGD([
        {'params': norm_biases, 'lr': lr * 64, 'weight_decay': wd / (lr * 64)},
        {'params': other, 'lr': lr, 'weight_decay': wd / lr},
    ], momentum=momentum, nesterov=True)

    def lr_factor(step):
        peak = int(0.23 * total_steps)
        if step <= peak:
            return 0.2 + 0.8 * step / peak
        return 1.0 + (0.07 - 1.0) * (step - peak) / (total_steps - peak)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction='sum')
    lookahead = Lookahead(net)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob('ckpt_epoch_*.pt'):
        old.unlink()
    step = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        net.layers[0].bias.requires_grad = epoch <= 3
        net.train()
        images = crop_batch(padded)
        if epoch % 2 == 0:
            images = images.flip(-1)
        order = torch.randperm(len(images), device='cuda')
        loss_sum = 0.0
        for i in range(steps_per_epoch):
            idx = order[i * batch_size:(i + 1) * batch_size]
            logits = net(images[idx])
            loss = loss_fn(logits, train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 5 == 0:
                alpha = 0.95 ** 5 * (step / total_steps) ** 3
                lookahead.update(net, alpha)
            loss_sum += loss.detach().item()
        save_scripted(net, epoch)
        torch.cuda.synchronize()
        print(f'epoch {epoch:02d} loss={loss_sum/(steps_per_epoch*batch_size):.4f} elapsed={time.perf_counter()-start:.3f}', flush=True)

    final = ROOT / 'checkpoint.pt'
    if final.exists() or final.is_symlink():
        final.unlink()
    os.link(OUT / f'ckpt_epoch_{epochs:04d}.pt', final)

    # Independent post-run check using exactly standardized float32 batches.
    test_u8, test_y = load_split('test')
    test_x = ((test_u8.float() / 255 - MEAN) / STD).contiguous()
    loaded = torch.jit.load(str(final), map_location='cuda').eval()
    correct = 0
    with torch.no_grad():
        for x, y in zip(test_x.split(512), test_y.split(512)):
            correct += (loaded(x).argmax(1) == y).sum().item()
    acc = correct / len(test_y)
    elapsed = time.perf_counter() - start
    print(f'exact_accuracy={acc:.4%} total_elapsed={elapsed:.3f}', flush=True)
    with open(ROOT / 'agent_reported_metrics.json', 'w') as f:
        json.dump({'accuracy': acc, 'wall_clock_sec': elapsed}, f)


if __name__ == '__main__':
    main()
