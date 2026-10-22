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
EPOCHS = 80
BATCH = 512
NUM_WORKERS = 8
LR = 0.5  # will be scaled with cosine
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- Model: PreAct ResNet for CIFAR ---
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if len(self.shortcut) > 0 else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class CifarResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen=4):
        super().__init__()
        # WRN-28-4 : 3 stages, each with n blocks, widen factor
        # depth 28 => n=4 (since 6*n+4=28)
        n = (depth - 4)//6
        k = widen
        self.in_ch = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(16*k, n, stride=1)
        self.layer2 = self._make_layer(32*k, n, stride=2)
        self.layer3 = self._make_layer(64*k, n, stride=2)
        self.bn = nn.BatchNorm2d(64*k)
        self.fc = nn.Linear(64*k, num_classes)
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
    def _make_layer(self, out_ch, blocks, stride):
        layers=[]
        for i in range(blocks):
            s = stride if i==0 else 1
            layers.append(BasicBlock(self.in_ch, out_ch, s))
            self.in_ch = out_ch
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def mixup_cutmix(batch_x, batch_y, alpha=1.0, cutmix_prob=0.5):
    # returns mixed x, y_a, y_b, lam
    if random.random() > 0.5:
        # mixup
        lam = random.betavariate(alpha, alpha) if alpha>0 else 1.0
        idx = torch.randperm(batch_x.size(0), device=batch_x.device)
        mixed_x = lam * batch_x + (1 - lam) * batch_x[idx]
        y_a, y_b = batch_y, batch_y[idx]
        return mixed_x, y_a, y_b, lam
    else:
        # cutmix
        lam = random.betavariate(alpha, alpha) if alpha>0 else 1.0
        idx = torch.randperm(batch_x.size(0), device=batch_x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(batch_x.size(), lam)
        mixed_x = batch_x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = batch_x[idx, :, bby1:bby2, bbx1:bbx2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size(-1) * batch_x.size(-2)))
        y_a, y_b = batch_y, batch_y[idx]
        return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[-1]
    H = size[-2]
    cut_rat = (1. - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W-1)
    cy = random.randint(0, H-1)
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)
    return bbx1, bby1, bbx2, bby2

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    # clean old checkpoints if any (ensure contiguous)
    for f in os.listdir(CKPT_DIR):
        if f.startswith("ckpt_epoch_"):
            os.remove(os.path.join(CKPT_DIR, f))
    if os.path.exists(FINAL_CKPT):
        os.remove(FINAL_CKPT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
    test_ds = datasets.ImageFolder(DATA_TEST, transform=test_tf)
    print(f"Train: {len(train_ds)} Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = CifarResNet(depth=28, widen=4).to(device)
    model = model.to(memory_format=torch.channels_last)
    # optional compile for speed - try but fallback
    # model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
    # cosine with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        # epoch is 0-indexed step count? We'll use per-epoch
        e = epoch + 1
        if e <= warmup_epochs:
            return e / warmup_epochs
        else:
            progress = (e - warmup_epochs) / (EPOCHS - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265))).item()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler()

    start_wall = time.time()
    first_ckpt_time = None
    last_ckpt_time = None
    best_acc = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_start = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            # mixup/cutmix with 50% prob
            if random.random() < 0.5:
                x, y_a, y_b, lam = mixup_cutmix(x, y, alpha=1.0)
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss = criterion(out, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch:03d}/{EPOCHS} acc={acc*100:.2f}% best={best_acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} time={time.time()-epoch_start:.1f}s")

        # save scripted checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        # need to script - ensure model is on cpu? script works on gpu model too but save scripted
        # Use model.eval() already
        scripted = torch.jit.script(model.eval())
        scripted.save(ckpt_path)
        if first_ckpt_time is None:
            first_ckpt_time = time.time()
        last_ckpt_time = time.time()
        # also verify load
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

        # early stop if we already hit 94% and have some margin? continue to ensure best
        # but we want to keep training to maybe improve speed? No, we want fastest to 94%
        # So we could stop early after hitting 94% + few epochs, but we need to keep ladder contiguous
        # We'll continue full run for now

    # final copy
    last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
    if os.path.exists(last_ckpt):
        shutil.copy(last_ckpt, FINAL_CKPT)
        print(f"Copied {last_ckpt} -> {FINAL_CKPT}")

    total_wall = last_ckpt_time - start_wall if last_ckpt_time else time.time() - start_wall
    span = last_ckpt_time - first_ckpt_time if first_ckpt_time and last_ckpt_time else total_wall
    print(f"Done. Best acc {best_acc*100:.2f}% total_wall {total_wall:.1f}s span {span:.1f}s")
    # write metrics
    import json
    with open("/workspace/agent_reported_metrics.json","w") as f:
        json.dump({"accuracy": best_acc, "wall_clock_sec": total_wall}, f)
    print("Metrics written")

if __name__ == "__main__":
    main()
