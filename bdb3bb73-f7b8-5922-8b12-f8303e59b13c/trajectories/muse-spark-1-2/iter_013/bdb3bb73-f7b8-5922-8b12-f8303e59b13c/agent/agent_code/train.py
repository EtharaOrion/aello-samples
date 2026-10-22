import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

# --- Config ---
DATA_ROOT = "/workspace/data/cifar10"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 30
BATCH = 512
WORKERS = 8
LR = 1.0  # for OneCycle max_lr
DEVICE = "cuda"

os.makedirs(CKPT_DIR, exist_ok=True)
# Clean old checkpoints
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

torch.backends.cudnn.benchmark = True

# --- Model: WideResNet-28-10 style but optimized for speed ---
# Use a fast ResNet variant

class Block(nn.Module):
    def __init__(self, inp, out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, 1, stride, bias=False),
                nn.BatchNorm2d(out)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 3 stages: 64->128->256 channels, 4 blocks each = ~11M params
        # Use wider: 64, 128, 256 with 4 blocks each
        self.layer1 = self._make(64, 128, 4, 1)
        self.layer2 = self._make(128, 256, 4, 2)
        self.layer3 = self._make(256, 512, 4, 2)
        self.bn_final = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make(self, inp, out, n, stride):
        layers = [Block(inp, out, stride)]
        for _ in range(1, n):
            layers.append(Block(out, out, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input is already normalized (standardized) per verifier
        # Do NOT re-normalize
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn_final(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

# --- Data ---
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Strong augmentation for fast convergence
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# CutMix / Mixup helpers
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    lam = random.betavariate(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = (1 - lam) ** 0.5
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    return bbx1, bby1, bbx2, bby2

# Datasets - handle both layouts
import pathlib
def get_datasets():
    # Try torchvision CIFAR10 with root
    # Data is in /workspace/data/cifar10/train and test as image folders
    # Check if it's ImageFolder structure
    train_path = os.path.join(DATA_ROOT, "train")
    test_path = os.path.join(DATA_ROOT, "test")
    if os.path.isdir(train_path) and os.path.isdir(os.path.join(train_path, "airplane")):
        print("Using ImageFolder datasets")
        train_ds = datasets.ImageFolder(train_path, transform=train_transform)
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        test_ds = datasets.ImageFolder(test_path, transform=test_tf)
    else:
        print("Using CIFAR10 torchvision download")
        train_ds = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=train_transform)
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        test_ds = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=test_tf)
    return train_ds, test_ds

train_ds, test_ds = get_datasets()
print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

model = Net().to(DEVICE)
model = model.to(memory_format=torch.channels_last)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader),
                                                  pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)

scaler = torch.amp.GradScaler('cuda')
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

start_wall = time.time()
first_ckpt_time = None
last_ckpt_time = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)

        # Apply Mixup/CutMix randomly
        r = random.random()
        use_mix = r < 0.5
        if use_mix:
            if r < 0.25:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
            else:
                x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            if use_mix:
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    # Eval
    model.eval()
    correct = total = 0
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for x, y in test_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                x = x.to(memory_format=torch.channels_last)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
    acc = correct / total
    elapsed = time.time() - start_wall
    print(f"Epoch {epoch}/{EPOCHS} acc={acc:.4f} elapsed={elapsed:.1f}s lr={optimizer.param_groups[0]['lr']:.4f}")

    # Save scripted checkpoint
    m = model.to(memory_format=torch.contiguous_format)
    scripted = torch.jit.script(m)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    m.to(memory_format=torch.channels_last)
    
    if first_ckpt_time is None:
        first_ckpt_time = time.time()
    last_ckpt_time = time.time()

    # Copy to final
    shutil.copy(ckpt_path, FINAL_CKPT)

    if acc >= 0.94:
        print(f"Reached 94% at epoch {epoch}")

# Write metrics
wall = last_ckpt_time - start_wall
# Also compute span
import glob
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
if ckpts:
    span = os.path.getmtime(ckpts[-1]) - os.path.getmtime(ckpts[0])
    wall = max(wall, span)

# Find best acc
best_acc = acc
metrics = {"accuracy": best_acc, "wall_clock_sec": wall}
import json
with open("/workspace/agent_reported_metrics.json", "w") as f:
    json.dump(metrics, f)
print(f"Done. wall={wall:.1f}s acc={best_acc:.4f}")
