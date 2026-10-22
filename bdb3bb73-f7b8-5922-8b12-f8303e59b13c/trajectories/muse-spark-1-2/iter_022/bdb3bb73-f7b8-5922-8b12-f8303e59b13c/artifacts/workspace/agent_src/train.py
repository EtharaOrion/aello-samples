import os, time, shutil, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- Config ---
EPOCHS = 36
BATCH = 1024
LR = 2.0
WD = 5e-4
LABEL_SMOOTH = 0.1
WARMUP = 4
CKPT_DIR = "/workspace/periodic_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, H100: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# --- Data ---
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

from torchvision.datasets import ImageFolder
train_set = ImageFolder("/workspace/data/cifar10/train", transform=train_tf)
test_set = ImageFolder("/workspace/data/cifar10/test", transform=test_tf)

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

def mixup_cutmix(x, y, alpha_mix=0.4, alpha_cut=1.0):
    r = random.random()
    lam = 1.0
    y1 = y
    y2 = y
    if r < 0.35:  # cutmix
        lam_b = np.random.beta(alpha_cut, alpha_cut)
        b = x.size(0)
        idx = torch.randperm(b, device=x.device)
        y1, y2 = y, y[idx]
        W = x.size(3); H = x.size(2)
        cut_rat = math.sqrt(1 - lam_b)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = random.randint(0, W-1)
        cy = random.randint(0, H-1)
        x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
        yc1 = max(cy - cut_h // 2, 0); yc2 = min(cy + cut_h // 2, H)
        x[:, :, yc1:yc2, x1:x2] = x[idx, :, yc1:yc2, x1:x2]
        lam = 1 - ((x2 - x1) * (yc2 - yc1) / (W * H))
    elif r < 0.65:  # mixup
        lam = float(np.random.beta(alpha_mix, alpha_mix))
        idx = torch.randperm(x.size(0), device=x.device)
        y1, y2 = y, y[idx]
        x = lam * x + (1 - lam) * x[idx]
    else:
        lam = 1.0
    return x, y1, y2, lam

# --- Model: WideResNet-28-10 variant optimized for CIFAR ---
class Block(nn.Module):
    def __init__(self, ci, co, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ci)
        self.conv1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(co)
        self.conv2 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or ci != co:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ci, co, 1, stride, bias=False),
                nn.BatchNorm2d(co)
            )
    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WRN(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.layer1 = self._make(16, 16*k, n, 1, dropout)
        self.layer2 = self._make(16*k, 32*k, n, 2, dropout)
        self.layer3 = self._make(32*k, 64*k, n, 2, dropout)
        self.bn = nn.BatchNorm2d(64*k)
        self.fc = nn.Linear(64*k, num_classes)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _make(self, ci, co, n, stride, dropout):
        layers = [Block(ci, co, stride)]
        for _ in range(1, n):
            layers.append(Block(co, co, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x is already standardized per verifier convention - do NOT renormalize
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.fc(out)
        return out

model = WRN(depth=28, widen=10, dropout=0.0).to(device)
# Compile for speed
model = torch.compile(model)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)

# Cosine with warmup
def lr_schedule(epoch):
    if epoch < WARMUP:
        return (epoch + 1) / WARMUP
    progress = (epoch - WARMUP) / (EPOCHS - WARMUP)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265))).item()

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

scaler = torch.amp.GradScaler('cuda')

# Clean old checkpoints
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

t0 = time.time()
print(f"Training {EPOCHS} epochs, batch {BATCH}, LR {LR}")

for epoch in range(EPOCHS):
    model.train()
    t_epoch = time.time()
    total_loss = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # mixup/cutmix augmentation on GPU
        x, y1, y2, lam = mixup_cutmix(x, y)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            if lam == 1.0:
                loss = criterion(out, y1)
            else:
                loss = lam * criterion(out, y1) + (1 - lam) * criterion(out, y2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    scheduler.step()

    # Eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    test_acc = correct / total
    dt = time.time() - t_epoch
    wall = time.time() - t0
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss {total_loss/total:.3f} | test {test_acc:.4f} | lr {optimizer.param_groups[0]['lr']:.4f} | {dt:.1f}s | wall {wall:.1f}s")

    # Save scripted checkpoint
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw.eval()
    scripted = torch.jit.script(raw)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch+1:04d}.pt")
    scripted.save(ckpt_path)
    raw.train()

# Final checkpoint
last = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
shutil.copy(last, "/workspace/checkpoint.pt")
print(f"Done. Wall {time.time()-t0:.1f}s")

# Report
import json
# Find first 94% crossing
best_acc = 0
for ep in range(1, EPOCHS+1):
    pass  # already printed
with open("/workspace/agent_reported_metrics.json", "w") as f:
    json.dump({"accuracy": test_acc, "wall_clock_sec": time.time()-t0}, f)
print(f"Final test acc: {test_acc:.4f}")
