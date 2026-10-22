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
EPOCHS = 60
BATCH_TRAIN = 512
BATCH_TEST = 512
NUM_WORKERS = 8
LR_MAX = 1.0
WD = 5e-4
MOMENTUM = 0.9
LABEL_SMOOTH = 0.1

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

torch.backends.cudnn.benchmark = True

# --- Model: ResNet9 style optimized for CIFAR ---
class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = ConvBN(c, c)
        self.c2 = ConvBN(c, c)
        # second conv without relu before add? we do relu after add
        # To keep scriptable, do explicit
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)

class FastCIFARResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Residual(128),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Residual(256),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Residual(512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = random.betavariate(alpha, alpha)
    # ensure lam >=0.5 for stability? not needed
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    batch_size = x.size(0)
    lam = random.betavariate(alpha, alpha)
    index = torch.randperm(batch_size, device=x.device)
    # generate bbox
    W = x.size(3)
    H = x.size(2)
    cut_rat = (1 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W-1)
    cy = random.randint(0, H-1)
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    # adjust lam by area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    x_cutmix = x.clone()
    x_cutmix[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    y_a, y_b = y, y[index]
    return x_cutmix, y_a, y_b, lam

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    # clean old checkpoints if any (ensure contiguous)
    for f in os.listdir(CKPT_DIR):
        fp = os.path.join(CKPT_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    # Simpler without AutoAugment for speed? Keep it - helps accuracy
    # Use alternative: if AutoAugment too slow, fallback
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
    test_ds = datasets.ImageFolder(DATA_TEST, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model = FastCIFARResNet().to(device)
    model = model.to(memory_format=torch.channels_last)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_MAX, momentum=MOMENTUM, weight_decay=WD, nesterov=True)
    # Cosine annealing with warmup? Use OneCycle or Cosine
    # Use CosineAnnealingLR over EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # Alternative: OneCycle
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)

    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    start_wall = time.time()
    first_ckpt_time = None
    last_ckpt_time = None

    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_start = time.time()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            # Apply MixUp/CutMix randomly
            r = random.random()
            use_mix = False
            lam = 1.0
            y_a = y
            y_b = y
            if r < 0.35:
                # CutMix
                x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
                use_mix = True
            elif r < 0.5:
                # MixUp
                x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
                use_mix = True

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(x)
                if use_mix:
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                else:
                    loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{EPOCHS}  acc={acc*100:.2f}%  lr={lr:.4f}  time={epoch_time:.2f}s")

        # Save scripted checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        # Need to script model - ensure eval mode and on cpu? Script on current device but save should be loadable on GPU
        # Use torch.jit.script
        scripted = torch.jit.script(model.eval())
        scripted.save(ckpt_path)
        # Ensure mtime is set by save (don't touch)
        if first_ckpt_time is None:
            first_ckpt_time = os.path.getmtime(ckpt_path)
        last_ckpt_time = os.path.getmtime(ckpt_path)
        # Also verify load
        try:
            m2 = torch.jit.load(ckpt_path, map_location=device)
            m2.eval()
            # quick sanity: run one batch
            with torch.no_grad():
                xb, _ = next(iter(test_loader))
                xb = xb.to(device).to(memory_format=torch.channels_last)
                _ = m2(xb[:4])
        except Exception as e:
            print(f"Checkpoint verify failed: {e}")

        # Early stop if we already hit 94% and have some margin? Continue to ensure best
        # But we want to keep training to improve time-to-94? Actually we want earliest epoch to hit 94.
        # So we don't early stop too early; we need to let it run but we already have checkpoint.
        # If acc >=0.94 and epoch >= 30, we could continue a bit then break to save time
        # Let's not break early; run all epochs

        model.train()

    # Final checkpoint copy
    last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
    if os.path.exists(last_ckpt):
        shutil.copy(last_ckpt, FINAL_CKPT)
        print(f"Copied {last_ckpt} -> {FINAL_CKPT}")
    else:
        # find latest
        files = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_epoch_")])
        if files:
            shutil.copy(os.path.join(CKPT_DIR, files[-1]), FINAL_CKPT)

    total_wall = time.time() - start_wall
    # Find best acc checkpoint
    # We already printed, but compute final acc
    # Write agent_reported_metrics.json
    # Need to find first epoch >=94% to report wall_clock_sec as total wall
    # wall_clock_sec must be >= span between first and last checkpoint
    # So use total_wall
    # Also compute best acc
    best_acc = 0
    # Re-eval last checkpoint for reporting
    # Use last acc from loop (approx)
    # Let's re-eval final model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    best_acc = correct / total
    print(f"Final acc: {best_acc*100:.2f}%  total_wall={total_wall:.2f}s")
    import json
    with open("/workspace/agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": best_acc, "wall_clock_sec": total_wall}, f)
    print("Done")

if __name__ == "__main__":
    main()
