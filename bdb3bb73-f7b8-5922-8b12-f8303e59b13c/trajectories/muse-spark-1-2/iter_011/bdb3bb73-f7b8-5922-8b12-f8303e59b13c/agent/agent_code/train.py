import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import autoaugment

# --- Config ---
DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 50
BATCH_SIZE = 1024
NUM_WORKERS = 8
LR = 1.2  # max lr for OneCycle
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.1
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

os.makedirs(CKPT_DIR, exist_ok=True)

# --- Model: ResNet for CIFAR (scriptable) ---
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CifarResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 3 stages: 64, 128, 256, 512 with [2,2,2,2] blocks -> ResNet18 variant for CIFAR
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(in_planes, planes, s))
            in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# --- Transforms ---
# Use strong augmentation: RandomCrop, Flip, AutoAugment, Cutout
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Cutout via RandomErasing as approximation (fast)
# Add RandomErasing after ToTensor+Normalize
# We'll use a custom Cutout transform
class Cutout:
    def __init__(self, size=16):
        self.size = size
    def __call__(self, img):
        # img is Tensor C H W already normalized
        h, w = img.shape[1], img.shape[2]
        y = random.randint(0, h-1)
        x = random.randint(0, w-1)
        y1 = max(0, y - self.size//2)
        y2 = min(h, y + self.size//2)
        x1 = max(0, x - self.size//2)
        x2 = min(w, x + self.size//2)
        img[:, y1:y2, x1:x2] = 0
        return img

# Wrap train transform to include Cutout
# We'll do: RandomCrop, Flip, AutoAugment, ToTensor, Normalize, Cutout
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    Cutout(16),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_transform)
test_ds = datasets.ImageFolder(DATA_TEST, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CifarResNet().to(device)

# Use channels_last for speed on H100
model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
# OneCycleLR: fast convergence
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)

scaler = torch.cuda.amp.GradScaler()

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = x.to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast():
                out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

start_wall = time.time()
first_ckpt_time = None
last_ckpt_time = None

print(f"Training {EPOCHS} epochs, {len(train_loader)} steps/epoch, batch {BATCH_SIZE}, device {device}")
for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_start = time.time()
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    # save checkpoint (scripted)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    # Need to script model: ensure eval mode for scripting? Keep train but script works either.
    # Use copy of model in eval for scripting
    m_to_save = model.eval()
    # Script - need to handle channels_last? scripting preserves.
    scripted = torch.jit.script(m_to_save)
    scripted.save(ckpt_path)
    # back to train
    model.train()
    last_ckpt_time = time.time()
    if first_ckpt_time is None:
        first_ckpt_time = last_ckpt_time
    # quick eval
    acc = evaluate(test_loader)
    lr_cur = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:03d}/{EPOCHS}  acc={acc*100:.2f}%  lr={lr_cur:.4f}  time={time.time()-epoch_start:.1f}s  wall={time.time()-start_wall:.1f}s")
    # early stop if we already hit 94 and have some margin? continue to ensure best
    # but we keep training all epochs

# copy last checkpoint to /workspace/checkpoint.pt
last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
shutil.copy(last_ckpt, FINAL_CKPT)
print(f"Done. Copied {last_ckpt} -> {FINAL_CKPT}")

# write metrics
total_wall = last_ckpt_time - start_wall if last_ckpt_time else 0
# also compute best acc
best_acc = evaluate(test_loader)
print(f"Final acc {best_acc*100:.2f}% wall {total_wall:.1f}s")
# For agent_reported_metrics, wall_clock_sec must be >= span between first and last ckpt
# Use total_wall as wall_clock_sec (from start to last ckpt)
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": float(best_acc), "wall_clock_sec": float(total_wall)}, f)
print("Wrote agent_reported_metrics.json")
