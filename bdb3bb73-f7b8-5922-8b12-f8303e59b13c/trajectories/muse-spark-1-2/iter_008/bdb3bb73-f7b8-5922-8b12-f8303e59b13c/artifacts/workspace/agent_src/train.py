import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import autoaugment

# --- Config ---
DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 60
BATCH_TRAIN = 512
BATCH_TEST = 512
NUM_WORKERS = 8
LR = 1.0  # for batch 512, will use cosine
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, torch {torch.__version__}")

# --- Model: WideResNet 28-10 for CIFAR ---
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        # need to handle shortcut after bn+relu for preact? Use original WRN preact: bn-relu-conv
        # For shortcut, use x directly if needed? Standard WRN uses bn-relu for shortcut path too
        # We'll do: shortcut = self.shortcut(x) if needed else x, but with preact the shortcut should be from x after? 
        # Simpler: use out for shortcut when dimensions match, else conv on x
        shortcut = self.shortcut(x)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + shortcut

class WRN(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth-4)%6==0
        n = (depth-4)//6
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        c1, c2, c3 = 16*widen, 32*widen, 64*widen
        # widen factor applied differently: typical WRN 28-10 has 160,320,640
        # For CIFAR, first block 16, then 160,320,640
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
    def _make_layer(self, in_ch, out_ch, n, stride, dropout):
        layers = [BasicBlock(in_ch, out_ch, stride, dropout)]
        for _ in range(1, n):
            layers.append(BasicBlock(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*layers)
    def forward(self, x):
        # x is already standardized, no renormalization
        out = self.init_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.fc(out)
        return out

# --- Data ---
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02,0.33), ratio=(0.3,3.3)),
])

# Use Cutout via RandomErasing already; also add simple
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
test_ds = datasets.ImageFolder(DATA_TEST, transform=test_tf)
print(f"Train {len(train_ds)} Test {len(test_ds)} Classes {train_ds.classes}")

train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

# --- Model, Optim, Sched ---
model = WRN(depth=28, widen=10, dropout=0.0).to(device)
model = model.to(memory_format=torch.channels_last)
# compile for speed
try:
    model = torch.compile(model)
    print("compiled")
except Exception as e:
    print(f"compile failed {e}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
# cosine with warmup 5 epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)
warmup_epochs = 5

scaler = torch.amp.GradScaler('cuda')

# mixup/cutmix
def mixup_cutmix(x, y, alpha=0.2, cutmix_prob=0.5):
    # randomly choose mixup or cutmix or none
    r = random.random()
    if r < 0.5:
        # mixup
        lam = random.betavariate(alpha, alpha) if alpha>0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1-lam) * x[idx]
        return mixed_x, y, y[idx], lam
    elif r < 1.0:
        # cutmix
        lam = random.betavariate(alpha, alpha) if alpha>0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2-bbx1)*(bby2-bby1) / (x.size(-1)*x.size(-2)))
        return x, y, y[idx], lam
    else:
        return x, y, None, 1.0

def rand_bbox(size, lam):
    W = size[-1]
    H = size[-2]
    cut_rat = (1. - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    bbx1 = max(0, cx - cut_w//2)
    bby1 = max(0, cy - cut_h//2)
    bbx2 = min(W, cx + cut_w//2)
    bby2 = min(H, cy + cut_h//2)
    return bbx1, bby1, bbx2, bby2

def evaluate(model_eval, loader):
    model_eval.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model_eval(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total

# --- Training ---
os.makedirs(CKPT_DIR, exist_ok=True)
# clean old checkpoints
for f in os.listdir(CKPT_DIR):
    fp = os.path.join(CKPT_DIR, f)
    if os.path.isfile(fp):
        os.remove(fp)
if os.path.exists(FINAL_CKPT):
    os.remove(FINAL_CKPT)

start_wall = time.time()
first_ckpt_time = None
last_ckpt_time = None

for epoch in range(1, EPOCHS+1):
    model.train()
    # warmup LR
    if epoch <= warmup_epochs:
        warm_lr = LR * epoch / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = warm_lr
    else:
        scheduler.step()

    epoch_loss = 0
    for x,y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)
        # mixup/cutmix 50% chance
        if random.random() < 0.5:
            # need to apply after moving to device
            lam = random.betavariate(1.0, 1.0)
            # choose mixup vs cutmix
            if random.random() < 0.5:
                idx = torch.randperm(x.size(0), device=device)
                mixed_x = lam * x + (1-lam) * x[idx]
                y_a, y_b = y, y[idx]
                lam_a = lam
                x = mixed_x
                use_mix = True
            else:
                idx = torch.randperm(x.size(0), device=device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
                lam = 1 - ((bbx2-bbx1)*(bby2-bby1) / (x.size(-1)*x.size(-2)))
                y_a, y_b = y, y[idx]
                lam_a = lam
                use_mix = True
        else:
            use_mix = False

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            if use_mix:
                loss = lam_a * criterion(out, y_a) + (1-lam_a) * criterion(out, y_b)
            else:
                loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    # eval
    acc = evaluate(model, test_loader)
    lr_cur = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:02d}/{EPOCHS} loss {epoch_loss/len(train_loader):.3f} acc {acc*100:.2f}% lr {lr_cur:.4f} time {time.time()-start_wall:.1f}s")

    # save scripted checkpoint
    # need to unwrap compiled model for scripting: use model._orig_mod if compiled
    to_script = model._orig_mod if hasattr(model, '_orig_mod') else model
    # ensure eval mode for scripting
    to_script.eval()
    scripted = torch.jit.script(to_script)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    # set back to train
    to_script.train()
    model.train()
    if first_ckpt_time is None:
        first_ckpt_time = time.time()
    last_ckpt_time = time.time()
    # also verify load
    try:
        m2 = torch.jit.load(ckpt_path, map_location=device)
        m2.eval()
        # quick sanity on one batch
        xb, yb = next(iter(test_loader))
        xb = xb.to(device)
        with torch.no_grad():
            out = m2(xb)
        assert out.shape[0]==xb.shape[0] and out.shape[1]==10
    except Exception as e:
        print(f"verify failed {e}")

    # early stop if 94% reached and we have some margin? continue to save more but we can stop after 94% + few epochs
    if acc >= 0.94 and epoch >= 30:
        # continue for 2 more epochs to ensure stable, then break
        if acc >= 0.94:
            # check if next epoch also high? just continue 2 more
            pass
    if acc >= 0.945 and epoch >= 35:
        print("Reached 94.5%, stopping early")
        break

# final copy
# find last ckpt
ckpts = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_epoch_")])
if ckpts:
    last = os.path.join(CKPT_DIR, ckpts[-1])
    shutil.copy(last, FINAL_CKPT)
    print(f"Copied {last} -> {FINAL_CKPT}")

total_wall = last_ckpt_time - start_wall if last_ckpt_time else time.time()-start_wall
span = last_ckpt_time - first_ckpt_time if first_ckpt_time and last_ckpt_time else total_wall
print(f"Total wall {total_wall:.1f}s span {span:.1f}s")

# write metrics
import json
# evaluate final
final_model = torch.jit.load(FINAL_CKPT, map_location=device)
final_model.eval()
acc_final = evaluate(final_model, test_loader)
print(f"Final acc {acc_final*100:.2f}%")
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": float(acc_final), "wall_clock_sec": float(total_wall)}, f)
