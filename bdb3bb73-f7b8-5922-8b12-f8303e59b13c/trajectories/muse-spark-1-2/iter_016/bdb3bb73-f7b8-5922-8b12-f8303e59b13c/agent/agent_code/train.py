import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# --- Config ---
DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 35
BATCH = 1024
NUM_WORKERS = 8
LR = 1.0
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Clean ckpt dir
if os.path.exists(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# Cutout
class Cutout:
    def __init__(self, size=16):
        self.size = size
    def __call__(self, img):
        # img is tensor CxHxW normalized
        h, w = img.shape[1], img.shape[2]
        y = random.randint(0, h-1)
        x = random.randint(0, w-1)
        y1 = max(0, y - self.size//2)
        y2 = min(h, y + self.size//2)
        x1 = max(0, x - self.size//2)
        x2 = min(w, x + self.size//2)
        img[:, y1:y2, x1:x2] = 0
        return img

cutout = Cutout(16)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# we apply cutout after normalize via wrapper dataset
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

class CutoutDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        # img already normalized tensor
        if random.random() < 0.5:
            img = img.clone()
            h, w = img.shape[1], img.shape[2]
            y = random.randint(0, h-1)
            x = random.randint(0, w-1)
            y1 = max(0, y - 8)
            y2 = min(h, y + 8)
            x1 = max(0, x - 8)
            x2 = min(w, x + 8)
            img[:, y1:y2, x1:x2] = 0
        return img, label

train_base = ImageFolder(DATA_TRAIN, transform=train_tf)
train_ds = CutoutDataset(train_base)
test_ds = ImageFolder(DATA_TEST, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=2)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False)

# --- Model: WideResNet 28-10 scriptable ---
class BasicBlock(nn.Module):
    __constants__ = ['equalInOut']
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
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
            shortcut = self.convShortcut(out)
            return shortcut + out1

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
        assert (depth-4)%6==0
        n = (depth-4)//6
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

model = WideResNet(28, 10, 0.3, 10).to(device)
model = model.to(memory_format=torch.channels_last)

# EMA
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k,v in model.state_dict().items() if v.dtype==torch.float32}
    def update(self, model):
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1-self.decay)
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

ema = EMA(model, 0.999)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
# Cosine with warmup
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# Warmup manually for first 5 epochs
def warmup_lr(epoch):
    if epoch < 5:
        return (epoch+1)/5
    return 1.0

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

scaler = torch.cuda.amp.GradScaler()

# Mixup
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

start_time = time.time()
first_ckpt_time = None
last_ckpt_time = None

for epoch in range(1, EPOCHS+1):
    model.train()
    # warmup lr scaling
    if epoch <= 5:
        for pg in optimizer.param_groups:
            pg['lr'] = LR * (epoch / 5)
    else:
        # cosine after warmup: we use scheduler step after epoch, but adjust
        pass

    for x, y in train_loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        # mixup 50% chance
        use_mixup = random.random() < 0.5
        if use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)
        else:
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

    if epoch > 5:
        scheduler.step()

    # eval with EMA model for checkpoint? Use EMA weights for better accuracy
    # Create eval model with EMA weights
    eval_model = WideResNet(28, 10, 0.3, 10).to(device)
    eval_model.load_state_dict(ema.shadow, strict=False)
    eval_model.eval()
    # quick eval
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                out = eval_model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    acc = correct/total
    print(f"Epoch {epoch}/{EPOCHS} acc={acc:.4f} lr={optimizer.param_groups[0]['lr']:.4f}")

    # save checkpoint as scripted EMA model
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    # script the eval_model
    scripted = torch.jit.script(eval_model)
    scripted.save(ckpt_path)
    if epoch == 1:
        first_ckpt_time = time.time()
    last_ckpt_time = time.time()
    # also verify load
    try:
        m = torch.jit.load(ckpt_path, map_location="cpu")
        m.eval()
        # quick sanity: run dummy
        dummy = torch.randn(2,3,32,32)
        # need to normalize dummy? just test forward
        _ = m(dummy)
    except Exception as e:
        print(f"Checkpoint verify failed: {e}")

# final copy
last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
if os.path.exists(last_ckpt):
    shutil.copy(last_ckpt, FINAL_CKPT)
    print(f"Copied {last_ckpt} to {FINAL_CKPT}")

wall = last_ckpt_time - start_time
span = last_ckpt_time - first_ckpt_time if first_ckpt_time else wall
print(f"Wall {wall:.2f}s span {span:.2f}s")

# write metrics
import json
# find best acc checkpoint
best_acc = 0
best_epoch = EPOCHS
# we already have last acc, but re-eval best? just use last
# For metrics, report last acc and wall
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": acc, "wall_clock_sec": wall}, f)
print("Done")
