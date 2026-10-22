import os, time, shutil, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 40
BATCH = 1024
LR = 0.8
WD = 5e-4

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

if os.path.exists(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
os.makedirs(CKPT_DIR, exist_ok=True)
device = torch.device("cuda")

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

class CutoutWrapper(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        if random.random() < 0.5:
            img = img.clone()
            y = random.randint(0, 31)
            x = random.randint(0, 31)
            y1 = max(0, y-8); y2 = min(32, y+8)
            x1 = max(0, x-8); x2 = min(32, x+8)
            img[:, y1:y2, x1:x2] = 0
        return img, label

train_base = ImageFolder(DATA_TRAIN, transform=train_tf)
train_ds = CutoutWrapper(train_base)
test_ds = ImageFolder(DATA_TEST, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False)

class BasicBlock(nn.Module):
    __constants__ = ['equalInOut']
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
            self.convShortcut = nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False)
        else:
            self.convShortcut = nn.Identity()
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        out1 = self.conv1(out)
        out1 = self.relu2(self.bn2(out1))
        out1 = self.conv2(out1)
        if self.equalInOut:
            return x + out1
        else:
            return self.convShortcut(out) + out1

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(i==0 and in_planes or out_planes, out_planes, i==0 and stride or 1))
        self.layer = nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10):
        super().__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
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

model = WideResNet(28, 10, 10).to(device)
model = model.to(memory_format=torch.channels_last)
model = torch.compile(model)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler('cuda')

def mixup_data(x, y, alpha=0.2):
    lam = random.betavariate(alpha, alpha) if alpha>0 else 1
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1-lam) * x[idx]
    return mixed, y, y[idx], lam

start = time.time()
first = None
last = None
best_acc = 0

for epoch in range(1, EPOCHS+1):
    model.train()
    for x, y in train_loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        if random.random() < 0.5:
            x, y_a, y_b, lam = mixup_data(x, y, 0.2)
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)
        else:
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    acc = correct/total
    print(f"Epoch {epoch} acc={acc:.4f} lr={optimizer.param_groups[0]['lr']:.4f}", flush=True)
    if acc > best_acc:
        best_acc = acc

    unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
    script_model = WideResNet(28, 10, 10).to(device)
    script_model.load_state_dict(unwrapped.state_dict())
    script_model.eval()
    scripted = torch.jit.script(script_model)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    if first is None:
        first = time.time()
    last = time.time()
    try:
        m = torch.jit.load(ckpt_path, map_location="cpu")
        m.eval()
        dummy = torch.randn(2,3,32,32)
        _ = m(dummy)
    except Exception as e:
        print(f"verify fail {e}")

last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
shutil.copy(last_ckpt, FINAL_CKPT)
wall = last - start
span = last - first if first else wall
print(f"Wall {wall:.2f} span {span:.2f} best {best_acc:.4f}")
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": best_acc, "wall_clock_sec": wall}, f)
