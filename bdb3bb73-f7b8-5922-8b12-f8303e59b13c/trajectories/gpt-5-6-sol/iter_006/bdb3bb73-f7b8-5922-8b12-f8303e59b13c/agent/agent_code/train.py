import json
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
TRAIN = ROOT / 'data/cifar10/train'
OUT = ROOT / 'periodic_checkpoints'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)


class ConvBN(nn.Module):
    def __init__(self, ci: int, co: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(co, momentum=0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Residual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.a = ConvBN(channels, channels)
        self.b = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.b(self.a(x)), inplace=True)


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = ConvBN(3, 64)
        self.stage1 = nn.Sequential(ConvBN(64, 128), nn.MaxPool2d(2), Residual(128))
        self.stage2 = nn.Sequential(ConvBN(128, 256), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(ConvBN(256, 512), nn.MaxPool2d(2), Residual(512))
        self.head = nn.Linear(512, 10, bias=False)

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.head(x) * 0.125

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is already normalized by the caller, exactly as required.
        logits = self._forward_once(x)
        if self.training:
            return logits
        return (logits + self._forward_once(x.flip(3))) * 0.5


def load_train() -> tuple[torch.Tensor, torch.Tensor]:
    cache = ROOT / 'train_uint8.pt'
    if cache.exists():
        obj = torch.load(cache, map_location='cpu')
        return obj['images'], obj['labels']
    images, labels = [], []
    for label, name in enumerate(CLASSES):
        for path in sorted((TRAIN / name).glob('*.png')):
            images.append(torch.from_numpy(np.array(Image.open(path).convert('RGB'), copy=True)).permute(2, 0, 1))
            labels.append(label)
    x = torch.stack(images).to(torch.uint8)
    y = torch.tensor(labels, dtype=torch.long)
    torch.save({'images': x, 'labels': y}, cache)
    return x, y


def augment(xpad: torch.Tensor, indices: torch.Tensor, mean: torch.Tensor,
            std: torch.Tensor) -> torch.Tensor:
    x = xpad[indices]
    b = x.shape[0]
    oy = torch.randint(0, 9, (b,), device=x.device)
    ox = torch.randint(0, 9, (b,), device=x.device)
    q = torch.arange(32, device=x.device)
    rows = oy[:, None] + q[None, :]
    cols = ox[:, None] + q[None, :]
    z = x.permute(0, 2, 3, 1)
    bi = torch.arange(b, device=x.device)[:, None, None]
    z = z[bi, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
    flip = torch.rand(b, device=x.device) < 0.5
    z[flip] = z[flip].flip(3)
    z = z.float().div_(255.0)
    return z.sub_(mean).div_(std).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def save_scripted(model: nn.Module, epoch: int) -> Path:
    model.eval()
    path = OUT / f'ckpt_epoch_{epoch:04d}.pt'
    scripted = torch.jit.script(model)
    scripted.save(str(path))
    model.train()
    return path


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    raw, labels = load_train()
    device = torch.device('cuda')
    raw = raw.to(device)
    labels = labels.to(device)
    # Reflection padding is materialized once; every epoch samples a new crop.
    xpad = F.pad(raw.float(), (4, 4, 4, 4), mode='reflect').to(torch.uint8)
    mean, std = MEAN.to(device), STD.to(device)

    model = CifarNet().to(device, memory_format=torch.channels_last)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    compiled = torch.compile(model, mode='max-autotune')
    epochs, bs = 30, 1000
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9,
                                nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.8, epochs=epochs, steps_per_epoch=50000 // bs,
        pct_start=0.22, div_factor=10.0, final_div_factor=1000.0,
    )
    scaler = torch.cuda.amp.GradScaler()

    # Compile before starting the timestamped ladder.
    warm = augment(xpad, torch.arange(bs, device=device), mean, std)
    with torch.autocast('cuda', dtype=torch.float16):
        loss = F.cross_entropy(compiled(warm), labels[:bs], label_smoothing=0.1)
    scaler.scale(loss).backward()
    optimizer.zero_grad(set_to_none=True)
    # Reinitialize so compilation has no training effect.
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.reset_parameters()
        elif isinstance(m, nn.Linear):
            m.reset_parameters()
    optimizer.state.clear()

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob('ckpt_epoch_*.pt'):
        old.unlink()
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(50000, device=device)
        running = 0.0
        for j in range(0, 50000, bs):
            idx = perm[j:j + bs]
            xb = augment(xpad, idx, mean, std)
            yb = labels[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = compiled(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss)
        torch.cuda.synchronize()
        path = save_scripted(model, epoch)
        elapsed = time.perf_counter() - start
        print(json.dumps({'epoch': epoch, 'loss': running / 50,
                          'elapsed': elapsed, 'path': str(path)}), flush=True)

    final = ROOT / 'checkpoint.pt'
    if final.exists():
        final.unlink()
    os.link(OUT / f'ckpt_epoch_{epochs:04d}.pt', final)
    with open(ROOT / 'agent_reported_metrics.json', 'w') as f:
        json.dump({'accuracy': 0.0, 'wall_clock_sec': elapsed}, f)


if __name__ == '__main__':
    main()
