import os, time, shutil, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

EPOCHS = 30
BATCH = 1024
LR = 1.0
WD = 5e-4
WARMUP = 2
CKPT_DIR = "/workspace/scratch_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)
device = torch.device("cuda")
print(f"H100: {torch.cuda.get_device_name(0)}")

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# Minimal aug for speed: just crop+flip, no AutoAugment
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

train_set = ImageFolder("/workspace/data/cifar10/train", transform=train_tf)
test_set = ImageFolder("/workspace/data/cifar10/test", transform=test_tf)
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

class Block(nn.Module):
    def __init__(self, ci, co, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ci)
        self.conv1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(co)
        self.conv2 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or ci != co:
            self.shortcut = nn.Sequential(nn.Conv2d(ci, co, 1, stride, bias=False), nn.BatchNorm2d(co))
    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WRN(nn.Module):
    def __init__(self, depth=28, widen=10):
        super().__init__()
        n = (depth-4)//6; k=widen
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.layer1 = self._make(16, 16*k, n, 1)
        self.layer2 = self._make(16*k, 32*k, n, 2)
        self.layer3 = self._make(32*k, 64*k, n, 2)
        self.bn = nn.BatchNorm2d(64*k)
        self.fc = nn.Linear(64*k, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'); nn.init.constant_(m.bias,0)
    def _make(self, ci, co, n, stride):
        layers=[Block(ci,co,stride)]
        for _ in range(1,n): layers.append(Block(co,co,1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out=self.conv1(x)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=F.relu(self.bn(out))
        out=F.adaptive_avg_pool2d(out,1).flatten(1)
        return self.fc(out)

model = WRN(28,10).to(device)
model = model.to(memory_format=torch.channels_last)
model = torch.compile(model)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
def lr_fn(epoch):
    if epoch < WARMUP: return (epoch+1)/WARMUP
    p=(epoch-WARMUP)/(EPOCHS-WARMUP)
    return 0.5*(1+math.cos(math.pi*p))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
scaler = torch.amp.GradScaler('cuda')

for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR,f))

t0=time.time()
print(f"Training {EPOCHS}ep B{BATCH} LR{LR}")

for epoch in range(EPOCHS):
    model.train()
    te=time.time()
    for x,y in train_loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        x = x.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out=model(x)
            loss=criterion(out,y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    model.eval()
    correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda'): out=model(x)
            correct+=(out.argmax(1)==y).sum().item()
            total+=y.size(0)
    acc=correct/total
    dt=time.time()-te; wall=time.time()-t0
    print(f"Ep {epoch+1:02d} acc {acc:.4f} lr {optimizer.param_groups[0]['lr']:.4f} {dt:.1f}s wall {wall:.1f}s")
    raw=model._orig_mod if hasattr(model,'_orig_mod') else model
    raw.eval()
    torch.jit.script(raw).save(os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch+1:04d}.pt"))
    raw.train()
    if acc >= 0.94:
        print(f"  *** 94% at epoch {epoch+1} wall {wall:.1f}s ***")

import json, glob
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
if ckpts:
    print(f"Done wall {time.time()-t0:.1f}s final acc {acc:.4f}")
