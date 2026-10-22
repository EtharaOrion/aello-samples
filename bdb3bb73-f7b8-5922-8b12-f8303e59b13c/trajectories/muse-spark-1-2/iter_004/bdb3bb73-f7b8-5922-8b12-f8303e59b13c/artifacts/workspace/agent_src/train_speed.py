import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CKPT_DIR = "/workspace/scratch_ckpt"
DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
BATCH = 2048
EPOCHS = 30
LR = 1.2
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1

torch.backends.cudnn.benchmark = True
device = torch.device("cuda")

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)
        else:
            self.convShortcut = nn.Identity()
    def forward(self, x):
        if self.equalInOut:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
            out = self.conv2(out)
            return x + out
        else:
            out = self.relu1(self.bn1(x))
            shortcut = self.convShortcut(out)
            out = self.conv1(out)
            out = self.relu2(self.bn2(out))
            out = self.conv2(out)
            return shortcut + out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride):
        super().__init__()
        layers=[]
        for i in range(nb_layers):
            layers.append(BasicBlock(i==0 and in_planes or out_planes, out_planes, i==0 and stride or 1))
        self.layer = nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10):
        super().__init__()
        nChannels = [16, 16*widen, 32*widen, 64*widen]
        assert (depth-4)%6==0
        n = (depth-4)//6
        self.conv1 = nn.Conv2d(3, nChannels[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], 2)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()
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

train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_transform)
test_ds = datasets.ImageFolder(DATA_TEST, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train {len(train_ds)} Test {len(test_ds)}")

model = WideResNet(depth=28, widen=10).to(device)
model = model.to(memory_format=torch.channels_last)
# no compile for speed test first
# try simple compile
try:
    model = torch.compile(model)
    print("compile enabled")
except Exception as e:
    print(e)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)
warmup_epochs = 2
scaler = torch.amp.GradScaler('cuda')

def rand_bbox(size, lam):
    W = size[-1]; H = size[-2]
    cut_rat = (1. - lam) ** 0.5
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = random.randint(0, W-1); cy = random.randint(0, H-1)
    bbx1 = max(0, cx - cut_w//2); bby1 = max(0, cy - cut_h//2)
    bbx2 = min(W, cx + cut_w//2); bby2 = min(H, cy + cut_h//2)
    return bbx1, bby1, bbx2, bby2

def evaluate(loader, mdl):
    mdl.eval()
    correct=0; total=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda'):
                out = mdl(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total

os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    fp=os.path.join(CKPT_DIR,f)
    if os.path.isfile(fp):
        os.remove(fp)

start_wall = time.time()
first_ckpt_time=None
last_ckpt_time=None

for epoch in range(1, EPOCHS+1):
    model.train()
    if epoch <= warmup_epochs:
        warmup_lr = LR * epoch / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr']=warmup_lr
    else:
        scheduler.step()
    epoch_loss=0
    for x,y in train_loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)
        r = random.random()
        if r < 0.3:
            lam = random.betavariate(0.2, 0.2)
            idx = torch.randperm(x.size(0), device=device)
            mixed_x = lam * x + (1-lam) * x[idx]
            y_a, y_b = y, y[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(mixed_x)
                loss = lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        elif r < 0.6:
            lam = random.betavariate(1.0, 1.0)
            idx = torch.randperm(x.size(0), device=device)
            bbx1,bby1,bbx2,bby2 = rand_bbox(x.size(), lam)
            mixed_x = x.clone()
            mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
            lam = 1 - ((bbx2-bbx1)*(bby2-bby1) / (32*32))
            y_a, y_b = y, y[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(mixed_x)
                loss = lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        epoch_loss += loss.item()

    acc = evaluate(test_loader, model)
    lr_cur = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:03d}/{EPOCHS} loss {epoch_loss/len(train_loader):.4f} acc {acc*100:.2f}% lr {lr_cur:.5f} time {time.time()-start_wall:.1f}s", flush=True)

    cpu_model = WideResNet(depth=28, widen=10)
    cpu_model.load_state_dict(model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict())
    cpu_model.eval()
    scripted = torch.jit.script(cpu_model)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    if first_ckpt_time is None:
        first_ckpt_time=time.time()
    last_ckpt_time=time.time()

ckpts=sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_epoch_")])
if ckpts:
    last_ckpt=os.path.join(CKPT_DIR, ckpts[-1])
    shutil.copy(last_ckpt, "/workspace/scratch_final.pt")
    print(f"Copied {last_ckpt}")

total_wall = last_ckpt_time - start_wall if last_ckpt_time else time.time()-start_wall
print(f"Total wall {total_wall:.1f}s")
import json
try:
    final_model=torch.jit.load("/workspace/scratch_final.pt").to(device).eval()
    acc_final=evaluate(test_loader, final_model)
    print(f"Final acc {acc_final*100:.2f}%")
    with open("/workspace/scratch_metrics.json","w") as f:
        json.dump({"accuracy": float(acc_final), "wall_clock_sec": float(total_wall)}, f)
except Exception as e:
    print(f"final eval failed {e}")
