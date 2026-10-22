import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 40
BATCH_SIZE = 1024
NUM_WORKERS = 4
LR = 1.5
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.makedirs(CKPT_DIR, exist_ok=True)

# --- Fast ResNet9 (DAWNBench style) ---
class ConvBN(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = ConvBN(3, 64, 3, 1, 1)
        self.layer1 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128, 1))
        self.layer2 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256, 1))
        self.layer3 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- Transforms: light but effective ---
# Use RandomCrop + Flip + Cutout (via RandomErasing) - all CPU but light
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_transform)
test_ds = datasets.ImageFolder(DATA_TEST, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device("cuda")
model = ResNet9().to(device)
model = model.to(memory_format=torch.channels_last)

# Try torch.compile for speed (torch 2.4)
try:
    model = torch.compile(model, mode="max-autotune")
    print("torch.compile enabled")
except Exception as e:
    print(f"compile failed: {e}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)

scaler = torch.amp.GradScaler('cuda')

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda'):
                out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

start_wall = time.time()
first_ckpt_time = None
last_ckpt_time = None

print(f"Training {EPOCHS} epochs, {len(train_loader)} steps/epoch, batch {BATCH_SIZE}")
for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_start = time.time()
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    # save checkpoint - need to handle compiled model: unwrap
    # For scripting, we need the underlying module
    to_script = model._orig_mod if hasattr(model, '_orig_mod') else model
    # Ensure eval
    to_script.eval()
    scripted = torch.jit.script(to_script)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    # back to train
    to_script.train()
    # if compiled, need to keep compiled wrapper in train mode too
    if hasattr(model, '_orig_mod'):
        model.train()

    last_ckpt_time = time.time()
    if first_ckpt_time is None:
        first_ckpt_time = last_ckpt_time
    acc = evaluate(test_loader)
    lr_cur = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:03d}/{EPOCHS}  acc={acc*100:.2f}%  lr={lr_cur:.4f}  epoch_time={time.time()-epoch_start:.1f}s  wall={time.time()-start_wall:.1f}s")

last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
shutil.copy(last_ckpt, FINAL_CKPT)
print(f"Done. Copied {last_ckpt} -> {FINAL_CKPT}")
total_wall = last_ckpt_time - start_wall
best_acc = evaluate(test_loader)
print(f"Final acc {best_acc*100:.2f}% wall {total_wall:.1f}s")
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": float(best_acc), "wall_clock_sec": float(total_wall)}, f)
