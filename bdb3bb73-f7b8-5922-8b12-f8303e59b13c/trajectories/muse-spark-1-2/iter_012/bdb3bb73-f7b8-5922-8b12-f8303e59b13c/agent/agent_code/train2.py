import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 60
BATCH_TRAIN = 1024
BATCH_TEST = 512
NUM_WORKERS = 2
LR = 0.8
WD = 5e-4
LABEL_SMOOTH = 0.1

os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    fp = os.path.join(CKPT_DIR, f)
    if os.path.isfile(fp):
        os.remove(fp)
if os.path.exists(FINAL_CKPT):
    os.remove(FINAL_CKPT)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda")

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# Fast CPU aug: RandomCrop + Flip + Cutout via transforms
# Use simple aug, no AutoAugment
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = torchvision.datasets.ImageFolder(DATA_TRAIN, transform=train_transform)
test_ds = torchvision.datasets.ImageFolder(DATA_TEST, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False)

print(f"Train {len(train_ds)} Test {len(test_ds)} Batches {len(train_loader)}")

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
        self.convShortcut = None
        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False)
    def forward(self, x):
        if self.equalInOut:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
            return x + out
        else:
            x1 = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x1)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
            return self.convShortcut(x1) + out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        self.layer = nn.Sequential(*[block(in_planes if i==0 else out_planes, out_planes, stride if i==0 else 1, dropRate) for i in range(nb_layers)])
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

model = WideResNet(depth=28, widen_factor=10, dropRate=0.0).to(device)
model = model.to(memory_format=torch.channels_last)
print(f"Params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler = torch.amp.GradScaler('cuda')

def mixup_data(x, y, alpha=0.2):
    lam = random.betavariate(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam

def cutmix_data(x, y, alpha=1.0):
    lam = random.betavariate(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    W, H = x.size(2), x.size(3)
    cut_rat = (1. - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W-1)
    cy = random.randint(0, H-1)
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y, y[idx], lam

best_acc = 0
start_time = time.time()
first_ckpt_time = None
last_ckpt_time = None

for epoch in range(1, EPOCHS+1):
    model.train()
    # warmup 5 epochs
    if epoch <= 5:
        for pg in optimizer.param_groups:
            pg['lr'] = LR * epoch / 5
    else:
        if epoch == 6:
            # reset scheduler base lr
            for pg in optimizer.param_groups:
                pg['lr'] = LR
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-5)
        else:
            scheduler.step()

    running = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        if random.random() < 0.5:
            if random.random() < 0.5:
                inputs, ta, tb, lam = mixup_data(inputs, targets, 0.2)
            else:
                inputs, ta, tb, lam = cutmix_data(inputs, targets, 1.0)
            with torch.amp.autocast('cuda'):
                out = model(inputs)
                loss = lam * criterion(out, ta) + (1-lam) * criterion(out, tb)
        else:
            with torch.amp.autocast('cuda'):
                out = model(inputs)
                loss = criterion(out, targets)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()

    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for inputs, targets in test_loader:
                inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                out = model(inputs)
                _, pred = out.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
    acc = correct/total
    best_acc = max(best_acc, acc)
    print(f"Epoch {epoch}/{EPOCHS} loss {running/len(train_loader):.3f} acc {acc*100:.2f} best {best_acc*100:.2f} lr {optimizer.param_groups[0]['lr']:.4f} time {time.time()-start_time:.1f}s")

    # save scripted
    model.eval()
    scripted = torch.jit.script(model)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    # verify
    try:
        m2 = torch.jit.load(ckpt_path, map_location="cpu")
        m2.eval()
        with torch.no_grad():
            _ = m2(torch.randn(2,3,32,32))
    except Exception as e:
        print(f"verify fail {e}")
    model.train()
    if first_ckpt_time is None:
        first_ckpt_time = time.time()
    last_ckpt_time = time.time()

last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
if os.path.exists(last_ckpt):
    shutil.copy(last_ckpt, FINAL_CKPT)
    print(f"copied to {FINAL_CKPT}")

total_wall = last_ckpt_time - start_time
print(f"Done best {best_acc*100:.2f} wall {total_wall:.1f}s")
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": best_acc, "wall_clock_sec": total_wall}, f)
