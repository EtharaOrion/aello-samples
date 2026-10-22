import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import autoaugment

# --- Config ---
DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 50
BATCH = 512
NUM_WORKERS = 8
LR = 0.5
WD = 5e-4
LABEL_SMOOTH = 0.1

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

# --- Model: WRN-28-10 style but optimized for CIFAR ---
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        # need to handle shortcut after bn+relu for WRN style? Use preact
        # For simplicity, use standard preact WRN
        # Actually WRN preact: bn-relu-conv-bn-relu-drop-conv + shortcut
        # shortcut should be from x if in_ch != out_ch else identity, but with preact the shortcut is from x directly
        # We'll do it correctly:
        # out already is relu(bn(x)), conv1
        # But we need shortcut from x
        shortcut = self.shortcut(x)
        out = self.conv1(out)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        return out + shortcut

class WRN(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        c1, c2, c3 = 16*k, 32*k, 64*k
        # Actually for WRN-28-10: 16, 160, 320, 640
        # But 16*k where k=10 => 160 etc. First conv is 16, then stages.
        # Let's do: stage1: 16 -> 160, stage2: 320, stage3: 640
        self.stage1 = self._make_layer(16, c1, n, 1, dropout)
        self.stage2 = self._make_layer(c1, c2, n, 2, dropout)
        self.stage3 = self._make_layer(c2, c3, n, 2, dropout)
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
        # x is already normalized [-2,2], do NOT renormalize
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.fc(x)
        return x

# Use smaller WRN for speed: WRN-28-10 is heavy but H100 can handle. Let's use WRN-28-10 with dropout 0
# Alternative: WRN-16-8 is faster. We'll use 28-10 for accuracy.
# To speed up, use channels_last and compile? We'll use channels_last.

# --- Data ---
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# For speed, use TrivialAugmentWide instead of AutoAugment? AutoAugment is fine.
# Add Cutout via RandomErasing
train_tf_with_erase = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf_with_erase)
test_ds = datasets.ImageFolder(DATA_TEST, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)} Test: {len(test_ds)}")

model = WRN(depth=28, widen=10, dropout=0.0).to(device)
model = model.to(memory_format=torch.channels_last)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
# Cosine with warmup
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# Warmup manually for first 5 epochs
def warmup_lr(epoch):
    if epoch < 5:
        return (epoch + 1) / 5
    return 1.0

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

scaler = torch.amp.GradScaler('cuda')

# CutMix / MixUp
def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[-1]
    H = size[-2]
    cut_rat = (1. - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

torch.backends.cudnn.benchmark = True

# Clean ckpt dir
if os.path.exists(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
os.makedirs(CKPT_DIR, exist_ok=True)

start_wall = time.time()
first_ckpt_time = None
last_ckpt_time = None

for epoch in range(1, EPOCHS+1):
    model.train()
    # warmup LR scaling
    if epoch <= 5:
        warm = (epoch) / 5
        for pg in optimizer.param_groups:
            pg['lr'] = LR * warm
    else:
        # cosine after warmup: we use scheduler step after epoch, but adjust
        # scheduler was init with T_max=EPOCHS, we step each epoch after warmup
        pass

    epoch_start = time.time()
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        # apply mixup/cutmix randomly
        r = random.random()
        use_cutmix = r < 0.5
        use_mixup = 0.5 <= r < 0.7
        lam = 1.0
        y_a = y_b = y
        if use_cutmix:
            x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
        elif use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            if lam != 1.0:
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if epoch > 5:
        scheduler.step()
    # else scheduler not stepped during warmup, but we need to keep it in sync
    # Actually CosineAnnealingLR expects step each epoch; we skip warmup steps
    # So after warmup, we want cosine from epoch 5 to EPOCHS
    # We'll just use manual cosine:
    # Let's override LR with cosine after warmup
    if epoch > 5:
        # cosine decay from LR to 1e-4
        progress = (epoch - 5) / (EPOCHS - 5)
        lr = 1e-4 + 0.5 * (LR - 1e-4) * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    elapsed = time.time() - start_wall
    print(f"Epoch {epoch:02d}/{EPOCHS} acc={acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} time={elapsed:.1f}s")

    # save scripted checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    # script requires model on cpu? script on gpu also works but save cpu version for verifier
    m_cpu = model.to("cpu")
    # need to ensure model is in eval mode for scripting? Keep eval
    m_cpu.eval()
    scripted = torch.jit.script(m_cpu)
    scripted.save(ckpt_path)
    # move back to gpu
    model.to(device).to(memory_format=torch.channels_last)
    model.train()
    if first_ckpt_time is None:
        first_ckpt_time = time.time()
    last_ckpt_time = time.time()

    # early stop if 94% reached and we have at least some epochs? Continue to save more but we can break after 94% + a bit
    # Keep training to ensure best, but we want fast time-to-94, so we don't need extra epochs after crossing
    # We'll continue for 2 more epochs after crossing then break to save time
    if acc >= 0.94:
        print(f"Reached 94% at epoch {epoch}")
        # save final and break after 1 more epoch for safety? break now
        # Let's do 1 more epoch then break
        if acc >= 0.94 and epoch >= 10:  # ensure not too early fluke
            # check if we want to break immediately to minimize t_star
            # t_star is based on epoch N, so earlier N is better even if same wall time per epoch
            # So break now to keep N minimal
            break

# copy last to /workspace/checkpoint.pt
# find last ckpt
ckpts = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_epoch_")])
if ckpts:
    last = os.path.join(CKPT_DIR, ckpts[-1])
    shutil.copy(last, FINAL_CKPT)
    print(f"Copied {last} to {FINAL_CKPT}")

total_wall = last_ckpt_time - start_wall if last_ckpt_time else time.time() - start_wall
print(f"Total wall {total_wall:.1f}s from start to last ckpt, span {(last_ckpt_time - first_ckpt_time) if first_ckpt_time else 0:.1f}s")

# write metrics
import json
# evaluate final acc again for report
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
acc = correct/total
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": acc, "wall_clock_sec": total_wall}, f)
print(f"Final acc {acc:.4f} wall {total_wall:.1f}")

# verify scripted load
try:
    m = torch.jit.load(FINAL_CKPT)
    m.eval()
    print("Scripted load OK")
    # test with standardized input
    x = torch.randn(2,3,32,32)
    with torch.no_grad():
        y = m(x)
    print(f"Forward OK shape {y.shape}")
except Exception as e:
    print(f"Scripted load failed: {e}")
