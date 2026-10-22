import os, time, shutil, random, glob, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = "/workspace/data/cifar10"
CKPT_DIR = "/workspace/scratch_ckpt"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 24
BATCH = 1024
WORKERS = 8
LR = 1.2
DEVICE = "cuda"

os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

torch.backends.cudnn.benchmark = True

class Block(nn.Module):
    def __init__(self, inp, out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != out:
            self.shortcut = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), nn.BatchNorm2d(out))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 128, 4, 1)
        self.layer2 = self._make(128, 256, 4, 2)
        self.layer3 = self._make(256, 512, 4, 2)
        self.bn_final = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    def _make(self, inp, out, n, stride):
        layers = [Block(inp, out, stride)]
        for _ in range(1, n):
            layers.append(Block(out, out, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn_final(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Faster aug: RandomCrop + Flip + Cutout via RandomErasing (no AutoAugment)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = (1 - lam) ** 0.5
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    bbx1 = max(cx - cut_w // 2, 0); bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W); bby2 = min(cy + cut_h // 2, H)
    return bbx1, bby1, bbx2, bby2

train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
test_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=test_tf)
print(f"Train {len(train_ds)} Test {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

model = Net().to(DEVICE).to(memory_format=torch.channels_last)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)
scaler = torch.amp.GradScaler('cuda')
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

start_wall = time.time()
for epoch in range(1, EPOCHS+1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)
        # CutMix 50%
        use_cutmix = random.random() < 0.5
        if use_cutmix:
            lam = random.betavariate(1.0, 1.0)
            idx = torch.randperm(x.size(0), device=x.device)
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2-bbx1)*(bby2-bby1)/(32*32))
            y_a, y_b = y, y[idx]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            if use_cutmix:
                loss = lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)
            else:
                loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update(); scheduler.step()

    # eval
    model.eval(); correct=total=0
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for x,y in test_loader:
                x,y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                x = x.to(memory_format=torch.channels_last)
                correct += (model(x).argmax(1)==y).sum().item(); total+=y.size(0)
    acc = correct/total
    elapsed = time.time()-start_wall
    print(f"Epoch {epoch}/{EPOCHS} acc={acc:.4f} elapsed={elapsed:.1f}s lr={optimizer.param_groups[0]['lr']:.4f}")
    # save
    m = model.to(memory_format=torch.contiguous_format)
    torch.jit.script(m).save(os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt"))
    m.to(memory_format=torch.channels_last)

# copy best to periodic if better - for now just report
# find last acc
print(f"Done wall {time.time()-start_wall:.1f}s final acc {acc:.4f}")
