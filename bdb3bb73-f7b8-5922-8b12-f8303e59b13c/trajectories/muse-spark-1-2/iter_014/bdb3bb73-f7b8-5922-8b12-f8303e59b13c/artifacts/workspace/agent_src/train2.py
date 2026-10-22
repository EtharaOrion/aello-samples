import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

DATA_ROOT = "/workspace/data/cifar10"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
BATCH = 512
EPOCHS = 40
DEVICE = "cuda"

os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_set = ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
test_set = ImageFolder(os.path.join(DATA_ROOT, "test"), transform=test_transform)
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

class Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, 1, stride, bias=False)
    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10):
        super().__init__()
        n = (depth - 4) // 6
        k = widen
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.layer1 = self._make(16, 16*k, n, 1)
        self.layer2 = self._make(16*k, 32*k, n, 2)
        self.layer3 = self._make(32*k, 64*k, n, 2)
        self.bn = nn.BatchNorm2d(64*k)
        self.fc = nn.Linear(64*k, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    def _make(self, in_c, out_c, n, stride):
        layers = [Block(in_c, out_c, stride)]
        for _ in range(1, n):
            layers.append(Block(out_c, out_c, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)

model = WideResNet(28, 10).to(DEVICE)
model = model.to(memory_format=torch.channels_last)
torch.backends.cudnn.benchmark = True

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.5, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=50)

scaler = torch.amp.GradScaler('cuda')

start = time.time()
for epoch in range(1, EPOCHS+1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    # eval
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda'):
                out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct/total
    elapsed = time.time() - start
    print(f"Epoch {epoch} acc={acc:.4f} time={elapsed:.1f}s lr={optimizer.param_groups[0]['lr']:.5f}")

    # save scripted
    to_script = model
    # need to handle channels_last - scripting should work
    m_eval = WideResNet(28,10)
    m_eval.load_state_dict(model.state_dict())
    m_eval = m_eval.to(DEVICE).eval()
    scripted = torch.jit.script(m_eval)
    path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(path)
    print(f"  saved {path}")

    if acc >= 0.94:
        print(f"Reached 94% at epoch {epoch}")

# copy final
import glob
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_epoch_*.pt")))
if ckpts:
    shutil.copy(ckpts[-1], FINAL_CKPT)
    print(f"Copied {ckpts[-1]} to {FINAL_CKPT}")

# verify
for p in sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt"))):
    m = torch.jit.load(p, map_location="cpu")
    m.eval()
    with torch.no_grad():
        o = m(torch.randn(2,3,32,32))
    assert o.shape == (2,10)
    print(f"verify {os.path.basename(p)} ok")

total_time = time.time() - start
# final acc
m = torch.jit.load(FINAL_CKPT, map_location=DEVICE).eval()
correct=0; total=0
with torch.no_grad():
    for x,y in test_loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        correct+=(m(x).argmax(1)==y).sum().item()
        total+=y.size(0)
final_acc=correct/total
print(f"Final acc {final_acc:.4f} time {total_time:.1f}s")
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": final_acc, "wall_clock_sec": total_time}, f)
