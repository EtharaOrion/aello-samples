import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


ROOT = Path('/workspace')
DATA = ROOT / 'data/cifar10'
OUT = ROOT / 'periodic_checkpoints'
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device='cuda').view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device='cuda').view(1, 3, 1, 1)


class ConvGroup(nn.Module):
    def __init__(self, cin: int, cout: int, pool: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = ConvGroup(3, 64)
        self.g1 = ConvGroup(64, 128, True)
        self.r1a = ConvGroup(128, 128)
        self.r1b = ConvGroup(128, 128)
        self.g2 = ConvGroup(128, 256, True)
        self.g3 = ConvGroup(256, 512, True)
        self.r3a = ConvGroup(512, 512)
        self.r3b = ConvGroup(512, 512)
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, 10, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _forward_once(self, x):
        x = self.prep(x)
        x = self.g1(x)
        x = x + self.r1b(self.r1a(x))
        x = self.g2(x)
        x = self.g3(x)
        x = x + self.r3b(self.r3a(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def forward(self, x):
        logits = self._forward_once(x)
        if not self.training:
            logits = (logits + self._forward_once(x.flip(-1))) * 0.5
        return logits


def load_split(name, workers=4):
    ds = ImageFolder(DATA / name, transform=ToTensor())
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=workers,
                    pin_memory=True, persistent_workers=False, prefetch_factor=1)
    xs, ys = [], []
    for x, y in dl:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).cuda(), torch.cat(ys).cuda()


@torch.no_grad()
def init_whitening(model, x):
    # A representative sample of normalized 2x2 RGB patches.
    z = ((x[:5000] - MEAN) / STD)
    patches = F.unfold(z, (2, 2)).transpose(1, 2).reshape(-1, 12)
    patches = patches[::8]
    patches = patches - patches.mean(0)
    cov = patches.T @ patches / patches.shape[0]
    vals, vecs = torch.linalg.eigh(cov)
    w = vecs.T.reshape(12, 3, 2, 2) / torch.sqrt(vals.view(12, 1, 1, 1) + 1e-2)
    w = w.flip(0)
    model.whiten.weight.copy_(torch.cat((w, -w), 0))


def augment(x):
    # Reflection translation and independent flips.
    x = F.pad(x, (2, 2, 2, 2), mode='reflect')
    crops = x.unfold(2, 32, 1).unfold(3, 32, 1)
    ids = torch.arange(x.shape[0], device=x.device)
    oy = torch.randint(0, 5, (x.shape[0],), device=x.device)
    ox = torch.randint(0, 5, (x.shape[0],), device=x.device)
    x = crops[ids, :, oy, ox]
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x[flip] = x[flip].flip(-1)
    cy = torch.randint(4, 29, (x.shape[0], 1, 1), device=x.device)
    cx = torch.randint(4, 29, (x.shape[0], 1, 1), device=x.device)
    yy = torch.arange(32, device=x.device).view(1, 32, 1)
    xx = torch.arange(32, device=x.device).view(1, 1, 32)
    keep = ((yy - cy).abs() >= 4) | ((xx - cx).abs() >= 4)
    x = x * keep.unsqueeze(1)
    return x


def schedule(t):
    # Short warmup and linear decay, expressed as a multiplier.
    if t < 0.2:
        return t / 0.2
    return max(0.0, (1.0 - t) / 0.8)


@torch.no_grad()
def save_script(model, epoch):
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(str(OUT / f'ckpt_epoch_{epoch:04d}.pt'))
    model.train()


def main():
    torch.manual_seed(15)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    train_x, train_y = load_split('train')
    model = Net().cuda().to(memory_format=torch.channels_last)
    train_x = ((train_x - MEAN) / STD).to(memory_format=torch.channels_last)

    # Large batches and a high learning rate are the core of the fast recipe.
    epochs, bs = 40, 512
    params = [p for p in model.parameters() if p.requires_grad]
    base_lr = 0.4
    opt = torch.optim.SGD(params, lr=base_lr, momentum=0.85, nesterov=True,
                          weight_decay=5e-4)
    steps = epochs * (len(train_x) // bs)
    compiled = torch.compile(model, mode='max-autotune-no-cudagraphs')

    OUT.mkdir(exist_ok=True)
    for p in OUT.glob('*.pt'):
        p.unlink()
    start = time.time()
    step = 0
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_x), device='cuda')
        for j in range(0, len(train_x), bs):
            idx = order[j:j + bs]
            x, y = augment(train_x[idx]), train_y[idx]
            t = step / steps
            for g in opt.param_groups:
                g['lr'] = base_lr * schedule(t)
            opt.zero_grad(set_to_none=True)
            logits = compiled(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            loss.backward()
            opt.step()
            step += 1
        save_script(model, epoch)
        print(f'epoch {epoch} saved {time.time()-start:.3f}s', flush=True)

    final = OUT / f'ckpt_epoch_{epochs:04d}.pt'
    target = ROOT / 'checkpoint.pt'
    if target.exists():
        target.unlink()
    os.link(final, target)

    # Evaluation is strictly after training and checkpoint creation.
    test_x, test_y = load_split('test', workers=4)
    test_x = ((test_x - MEAN) / STD).to(memory_format=torch.channels_last)
    loaded = torch.jit.load(str(target)).cuda().eval()
    correct = 0
    with torch.no_grad():
        for j in range(0, len(test_x), 512):
            correct += (loaded(test_x[j:j+512]).argmax(1) == test_y[j:j+512]).sum().item()
    elapsed = time.time() - start
    acc = correct / len(test_y)
    print(f'accuracy {acc:.4%}, total {elapsed:.3f}s')
    with open(ROOT / 'agent_reported_metrics.json', 'w') as f:
        json.dump({'accuracy': acc, 'wall_clock_sec': elapsed}, f)


if __name__ == '__main__':
    main()
