import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- Config ---
EPOCHS = 45
BATCH = 1024
LR = 1.0  # for SGD with cosine
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1
CKPT_DIR = "/workspace/periodic_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# --- Data: preload to GPU-friendly pipeline ---
# Use standard CIFAR-10 mean/std
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# Training transforms - strong but fast augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])

# For Cutout we do it manually via transform, but AutoAugment already strong
# Add Cutout via RandomErasing as alternative
# We'll use CutMix/Mixup in training loop

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

from torchvision.datasets import ImageFolder
train_set = ImageFolder("/workspace/data/cifar10/train", transform=train_transform)
test_set = ImageFolder("/workspace/data/cifar10/test", transform=test_transform)

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

# --- Model: WideResNet-28-10 variant optimized for CIFAR ---
# Using a fast ResNet with width scaling

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i==0 and in_planes or out_planes, out_planes, i==0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropRate=0.0, num_classes=10):
        super().__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

model = WideResNet(28, 10, dropRate=0.0).to(device)
# Use channels_last for speed on H100
model = model.to(memory_format=torch.channels_last)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Compile for speed (PyTorch 2.0+)
try:
    model = torch.compile(model)
    print("torch.compile enabled")
except Exception as e:
    print(f"compile failed: {e}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler = torch.amp.GradScaler('cuda')

# Mixup/CutMix helpers
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

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Clean old checkpoints
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

start_wall = time.time()

for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_start = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last), targets.to(device, non_blocking=True)

        # Apply Mixup/CutMix with 50% prob each, 50% no mix
        r = random.random()
        use_mix = r < 0.5
        is_cutmix = r < 0.25

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            if use_mix:
                if is_cutmix:
                    inputs, y_a, y_b, lam = cutmix_data(inputs, targets, alpha=1.0)
                else:
                    inputs, y_a, y_b, lam = mixup_data(inputs, targets, alpha=0.2)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

    # Eval
    model.eval()
    correct = total = 0
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                _, pred = outputs.max(1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
    acc = correct / total
    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_wall
    print(f"Epoch {epoch}/{EPOCHS} | acc {acc*100:.2f}% | lr {optimizer.param_groups[0]['lr']:.4f} | epoch_time {epoch_time:.1f}s | total {total_time:.1f}s")

    # Save scripted checkpoint - need to unwrap compiled model
    # Get underlying model if compiled
    save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    # Ensure it's eval and scriptable - remove channels_last for scripting? keep it
    save_model.eval()
    # Script - need to handle channels_last model scripting
    # Create a wrapper that is scriptable
    try:
        scripted = torch.jit.script(save_model)
    except Exception as e:
        print(f"jit.script failed: {e}, trying trace fallback via script of eval copy")
        # Try without compile wrapper
        import copy
        m2 = copy.deepcopy(save_model)
        scripted = torch.jit.script(m2)

    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    # Also verify it loads
    try:
        loaded = torch.jit.load(ckpt_path, map_location="cpu")
        loaded.eval()
        # quick sanity: run dummy
        with torch.no_grad():
            dummy = torch.randn(2, 3, 32, 32)
            # standardize dummy like verifier does: already standardized range, but dummy is N(0,1) ~ similar
            _ = loaded(dummy)
        print(f"  -> saved & verified {ckpt_path}")
    except Exception as e:
        print(f"  -> save verify failed: {e}")

    # Re-set to train
    save_model.train()
    model.train()

    if acc >= 0.94:
        print(f"*** Hit 94% at epoch {epoch} ***")

# Final checkpoint copy
last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
if os.path.exists(last_ckpt):
    shutil.copy(last_ckpt, "/workspace/checkpoint.pt")
    print(f"Copied {last_ckpt} -> /workspace/checkpoint.pt")

# Write metrics
wall = time.time() - start_wall
# Find first epoch hitting 94% by re-eval (we already have acc, but use last)
# For metrics, report wall clock
import json
# Find best acc
# We printed acc, but just report final
with open("/workspace/agent_reported_metrics.json", "w") as f:
    json.dump({"accuracy": acc, "wall_clock_sec": wall}, f)
print(f"Done. wall={wall:.1f}s acc={acc:.4f}")
