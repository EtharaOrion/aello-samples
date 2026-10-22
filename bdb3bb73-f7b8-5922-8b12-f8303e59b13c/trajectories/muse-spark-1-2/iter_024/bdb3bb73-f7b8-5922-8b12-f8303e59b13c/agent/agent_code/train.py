import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import autoaugment

# --- Config ---
EPOCHS = 35
BATCH = 512
LR = 0.6
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5

CKPT_DIR = "/workspace/periodic_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
# clean old
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# --- Data ---
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = datasets.ImageFolder("/workspace/data/cifar10/train", transform=train_tf)
test_ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# --- Model: WideResNet 28-10 ---
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.shortcut = nn.Identity() if in_c==out_c and stride==1 else nn.Conv2d(in_c, out_c, 1, stride, bias=False)
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        # need to handle shortcut after bn+relu for WRN style? Use preact
        # For simplicity: if shortcut is conv, apply to x after bn+relu? Actually WRN uses bn-relu-conv
        # We'll do: out = conv1(relu(bn(x))), then bn-relu-dropout-conv2
        # shortcut: if needed, conv on relu(bn(x))? Let's do standard preact
        shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else self.shortcut(x)
        out = self.conv1(out)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        return out + shortcut

class WRN(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth-4)%6==0
        n = (depth-4)//6
        k = widen
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        c1, c2, c3 = 16*k, 32*k, 64*k
        self.layer1 = self._make_layer(16, c1, n, 1, dropout)
        self.layer2 = self._make_layer(c1, c2, n, 2, dropout)
        self.layer3 = self._make_layer(c2, c3, n, 2, dropout)
        self.bn = nn.BatchNorm2d(c3)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(c3, num_classes)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, in_c, out_c, n, stride, dropout):
        layers=[]
        layers.append(BasicBlock(in_c, out_c, stride, dropout))
        for _ in range(1,n):
            layers.append(BasicBlock(out_c, out_c, 1, dropout))
        return nn.Sequential(*layers)
    def forward(self, x):
        # x is already normalized [-2,2], do NOT renormalize
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.fc(x)
        return x

model = WRN(depth=28, widen=10, dropout=0.0).to(device)
model = model.to(memory_format=torch.channels_last)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
# cosine with warmup
import math
def lr_lambda(epoch):
    # warmup 3 epochs
    warmup = 3
    if epoch < warmup:
        return (epoch+1)/warmup
    # cosine from warmup to EPOCHS
    progress = (epoch - warmup) / (EPOCHS - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

def mixup_cutmix(x, y, alpha_mix=0.2, alpha_cut=1.0, p_cut=0.5):
    # randomly choose mixup or cutmix or none
    r = random.random()
    if r < 0.5:
        # mixup
        lam = torch.distributions.Beta(alpha_mix, alpha_mix).sample().item() if alpha_mix>0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1-lam) * x[idx]
        return mixed_x, y, y[idx], lam
    elif r < 0.5 + p_cut:
        # cutmix
        lam = torch.distributions.Beta(alpha_cut, alpha_cut).sample().item() if alpha_cut>0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2-bbx1)*(bby2-bby1) / (x.size(-1)*x.size(-2)))
        return mixed_x, y, y[idx], lam
    else:
        return x, y, None, 1.0

def rand_bbox(size, lam):
    W = size[-1]
    H = size[-2]
    cut_rat = (1 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W-1)
    cy = random.randint(0, H-1)
    bbx1 = max(0, cx - cut_w//2)
    bby1 = max(0, cy - cut_h//2)
    bbx2 = min(W, cx + cut_w//2)
    bby2 = min(H, cy + cut_h//2)
    return bbx1, bby1, bbx2, bby2

def evaluate(model):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = x.to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast():
                out = model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total

start_wall = time.time()
first_ckpt_time = None
last_ckpt_time = None

for epoch in range(1, EPOCHS+1):
    model.train()
    for x,y in train_loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)
        # mixup/cutmix
        x_mix, y_a, y_b, lam = mixup_cutmix(x, y, MIXUP_ALPHA, CUTMIX_ALPHA, CUTMIX_PROB)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out = model(x_mix)
            if y_b is None:
                loss = criterion(out, y_a)
            else:
                loss = lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()

    # eval
    acc = evaluate(model)
    print(f"Epoch {epoch}/{EPOCHS} acc={acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f}")

    # save scripted checkpoint
    model.eval()
    # need to script - ensure model is on cpu? script on gpu model works but save scripted
    # Use torch.jit.script
    scripted = torch.jit.script(model)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    if first_ckpt_time is None:
        first_ckpt_time = time.time()
    last_ckpt_time = time.time()
    # also verify load
    try:
        m2 = torch.jit.load(ckpt_path, map_location="cpu")
        m2.eval()
    except Exception as e:
        print(f"verify load failed {e}")

    if acc >= 0.94:
        print(f"Reached 94% at epoch {epoch}")

# final checkpoint.pt
last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
shutil.copy(last_ckpt, "/workspace/checkpoint.pt")
total_wall = last_ckpt_time - start_wall
span = last_ckpt_time - first_ckpt_time if first_ckpt_time else total_wall
print(f"Done total_wall={total_wall:.1f}s span={span:.1f}s")
# write metrics
import json
# find first crossing epoch
# we already printed, but compute again by evaluating saved checkpoints? Use last acc
# For metrics, report total wall from start to last ckpt
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": acc, "wall_clock_sec": total_wall}, f)
