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
EPOCHS = 50
BATCH_TRAIN = 1024
BATCH_TEST = 512
LR = 0.8
WD = 5e-4
MOMENTUM = 0.9
WARMUP_EPOCHS = 3
NUM_WORKERS = 0

os.makedirs(CKPT_DIR, exist_ok=True)
# clean old checkpoints
for f in os.listdir(CKPT_DIR):
    fp = os.path.join(CKPT_DIR, f)
    if os.path.isfile(fp):
        os.remove(fp)
if os.path.exists(FINAL_CKPT):
    os.remove(FINAL_CKPT)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, torch {torch.__version__}")

# --- Model: WideResNet-28-4 style for CIFAR ---
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

class WRN(nn.Module):
    def __init__(self, depth=28, widen=4, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen
        self.in_planes = 16
        nStages = [16, 16*k, 32*k, 64*k]
        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nStages[0])
        self.layer1 = self._make_layer(BasicBlock, nStages[1], n, stride=1)
        self.layer2 = self._make_layer(BasicBlock, nStages[2], n, stride=2)
        self.layer3 = self._make_layer(BasicBlock, nStages[3], n, stride=2)
        self.bn_final = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)
        self.dropout = dropout

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        if self.training and self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=True)
        out = self.linear(out)
        return out

model = WRN(depth=28, widen=4, dropout=0.0).to(device)
# channels_last for speed
model = model.to(memory_format=torch.channels_last)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# --- Data ---
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = ImageFolder(DATA_TRAIN, transform=train_transform)
test_ds = ImageFolder(DATA_TEST, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)

# --- Optim & Sched ---
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD, nesterov=True)
# Cosine with warmup
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    else:
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# MixUp / CutMix
def mixup_data(x, y, alpha=0.8):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = (1 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    return bbx1, bby1, bbx2, bby2

def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Training ---
start_wall = time.time()
first_ckpt_time = None
last_ckpt_time = None

for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_start = time.time()
    total_loss = 0
    correct = 0
    total = 0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        # apply mixup/cutmix with 50% prob
        use_mix = random.random() < 0.5
        if use_mix:
            if random.random() < 0.5:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=0.8)
            else:
                x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
            use_mix_flag = True
        else:
            use_mix_flag = False

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out = model(x)
            if use_mix_flag:
                loss = mix_criterion(criterion, out, y_a, y_b, lam)
            else:
                loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        # accuracy without mix for logging (approx)
        if not use_mix_flag:
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    scheduler.step()
    # eval
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                out = model(x)
            pred = out.argmax(1)
            test_correct += (pred == y).sum().item()
            test_total += y.size(0)
    acc = test_correct / test_total
    lr_cur = optimizer.param_groups[0]['lr']
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch:02d}/{EPOCHS} | loss {total_loss/len(train_ds):.4f} | test_acc {acc*100:.2f}% | lr {lr_cur:.4f} | {epoch_time:.1f}s")

    # save checkpoint as scripted model
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    # script a CPU copy in eval mode
    m_copy = WRN(depth=28, widen=4, dropout=0.0)
    m_copy.load_state_dict(model.state_dict())
    m_copy.eval()
    scripted = torch.jit.script(m_copy)
    scripted.save(ckpt_path)
    if epoch == 1:
        first_ckpt_time = time.time()
    last_ckpt_time = time.time()
    # also verify load
    try:
        _ = torch.jit.load(ckpt_path)
    except Exception as e:
        print(f"Checkpoint load failed: {e}")

    # early stop if 94% reached and we have some margin? continue to save all epochs but we can break after 94% + few epochs
    # keep training for at least 40 epochs to ensure stable 94%

# final checkpoint copy
last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
if os.path.exists(last_ckpt):
    shutil.copy(last_ckpt, FINAL_CKPT)
    print(f"Copied {last_ckpt} -> {FINAL_CKPT}")

wall_clock = last_ckpt_time - start_wall
span = last_ckpt_time - first_ckpt_time if first_ckpt_time else wall_clock
print(f"Wall clock total: {wall_clock:.1f}s, span first-last: {span:.1f}s")

# evaluate final accuracy for report
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(x)
        pred = out.argmax(1)
        test_correct += (pred == y).sum().item()
        test_total += y.size(0)
final_acc = test_correct / test_total
print(f"Final acc: {final_acc*100:.2f}%")

# write metrics
import json
with open("/workspace/agent_reported_metrics.json", "w") as f:
    json.dump({"accuracy": final_acc, "wall_clock_sec": wall_clock}, f)
print("Done")
