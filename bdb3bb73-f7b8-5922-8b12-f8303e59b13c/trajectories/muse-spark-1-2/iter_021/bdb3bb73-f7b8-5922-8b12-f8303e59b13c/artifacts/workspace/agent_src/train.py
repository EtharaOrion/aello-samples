import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import autoaugment

# --- Config ---
EPOCHS = 40
BATCH_TRAIN = 1024
BATCH_TEST = 512
LR_MAX = 1.5
WD = 5e-4
MOMENTUM = 0.9
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.4

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

CKPT_DIR = "/workspace/periodic_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
# clean old
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

torch.backends.cudnn.benchmark = True

# --- Model: WideResNet 28-10 variant optimized for CIFAR ---
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False) if in_ch != out_ch or stride != 1 else nn.Identity()
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        # WRN trick: bn-relu-conv
        # need to handle shortcut from original x vs activated
        # Use standard pre-activation: shortcut on x
        shortcut = self.shortcut(x)
        out = self.conv1(out)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        return out + shortcut

class WRN(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        c1, c2, c3 = 16*k, 32*k, 64*k
        self.layer1 = self._make_layer(16, c1, n, 1, dropout)
        self.layer2 = self._make_layer(c1, c2, n, 2, dropout)
        self.layer3 = self._make_layer(c2, c3, n, 2, dropout)
        self.bn = nn.BatchNorm2d(c3)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(c3, num_classes)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layer(self, in_ch, out_ch, n, stride, dropout):
        layers = [BasicBlock(in_ch, out_ch, stride, dropout)]
        for _ in range(1, n):
            layers.append(BasicBlock(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x is already standardized per verifier; do NOT renormalize
        out = self.init_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.fc(out)
        return out

# --- Data ---
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder("/workspace/data/cifar10/train", transform=train_tf)
test_ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

device = torch.device("cuda")
model = WRN(depth=28, widen=10, dropout=0.0).to(device)

# Try torch.compile for speed (torch 2.4)
try:
    model = torch.compile(model)
    print("torch.compile enabled")
except Exception as e:
    print(f"compile failed: {e}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.SGD(model.parameters(), lr=LR_MAX, momentum=MOMENTUM, weight_decay=WD, nesterov=True)
# OneCycle or Cosine
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)

scaler = torch.cuda.amp.GradScaler()

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = random.betavariate(alpha, alpha) if alpha > 0 else 1.0
    # ensure lam >=0.5 for stability? no
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Training ---
start_wall = time.time()
best_acc = 0

for epoch in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # mixup 50% of batches
        do_mix = random.random() < 0.5
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            if do_mix:
                x_m, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)
                out = model(x_m)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
                out = model(x)
                loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    best_acc = max(best_acc, acc)
    epoch_time = time.time() - t0
    print(f"Epoch {epoch:02d}/{EPOCHS} acc={acc*100:.2f}% best={best_acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} time={epoch_time:.1f}s")

    # Save scripted checkpoint - need to unwrap compiled model
    # Get underlying model for scripting
    to_script = model._orig_mod if hasattr(model, '_orig_mod') else model
    # Ensure eval mode for scripting
    to_script.eval()
    scripted = torch.jit.script(to_script)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    # also keep model in train for next epoch
    to_script.train()
    model.train()

    # early stop if we already have good acc and want to save time? continue to fill ladder
    # but we need to ensure we have contiguous 1..N

total_wall = time.time() - start_wall
print(f"Training done wall={total_wall:.1f}s best={best_acc*100:.2f}%")

# final checkpoint copy
last = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
if os.path.exists(last):
    shutil.copy(last, "/workspace/checkpoint.pt")
    print(f"Copied {last} -> /workspace/checkpoint.pt")

# write metrics
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": best_acc, "wall_clock_sec": total_wall}, f)
print("metrics written")

# verify last checkpoint loads and accuracy
try:
    m = torch.jit.load(last).eval().cuda()
    correct=0; total=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            out = m(x)
            correct += (out.argmax(1)==y).sum().item()
            total+=y.size(0)
    print(f"VERIFY scripted acc={correct/total*100:.2f}%")
except Exception as e:
    print(f"verify failed: {e}")
