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
BATCH_EVAL = 512
LR = 0.6
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1
WARMUP = 5
NUM_WORKERS = 8

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# --- Model: WRN-28-4 style adapted for CIFAR ---
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
        # need to handle shortcut after bn+relu for preact
        # For preact, shortcut is from x after bn+relu? Actually original WRN does bn-relu-conv
        # We'll do: out = conv1(relu(bn(x))), then bn-relu-dropout-conv2
        # shortcut: if needed, conv on x (or on relu(bn(x)))? Use x directly for simplicity
        # To keep preact correct, shortcut should be on x after first bn+relu if dimensions change?
        # Simpler: use standard post-act block for scripting stability
        # Let's implement post-act for simplicity and speed
        return out

# Actually use standard ResNet BasicBlock (post-act) for simplicity and jit friendliness
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CifarResNet(nn.Module):
    def __init__(self, num_classes=10, widen=4):
        super().__init__()
        # widen factor scales channels: base 16* widen
        c1 = 16 * widen  # 64
        c2 = 32 * widen  # 128
        c3 = 64 * widen  # 256
        self.conv1 = nn.Conv2d(3, c1, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        # 3 stages, each 4 blocks for depth 28 (4*6+4=28) -> 4 blocks per stage
        self.layer1 = self._make_layer(c1, c1, 4, 1)
        self.layer2 = self._make_layer(c1, c2, 4, 2)
        self.layer3 = self._make_layer(c2, c3, 4, 2)
        self.bn_final = nn.BatchNorm2d(c3)
        self.fc = nn.Linear(c3, num_classes)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []
        layers.append(Block(in_ch, out_ch, stride))
        for _ in range(1, blocks):
            layers.append(Block(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x is already standardized, do NOT renormalize
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def mixup_cutmix(batch_x, batch_y, alpha_mix=0.2, alpha_cut=1.0, prob=0.5):
    # randomly choose mixup or cutmix or none
    r = random.random()
    if r < 0.5:
        # mixup
        lam = random.betavariate(alpha_mix, alpha_mix) if alpha_mix>0 else 1.0
        idx = torch.randperm(batch_x.size(0), device=batch_x.device)
        mixed_x = lam * batch_x + (1 - lam) * batch_x[idx]
        return mixed_x, batch_y, batch_y[idx], lam
    elif r < 1.0:
        # cutmix
        lam = random.betavariate(alpha_cut, alpha_cut)
        idx = torch.randperm(batch_x.size(0), device=batch_x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(batch_x.size(), lam)
        batch_x[:, :, bby1:bby2, bbx1:bbx2] = batch_x[idx, :, bby1:bby2, bbx1:bbx2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size(-1) * batch_x.size(-2)))
        return batch_x, batch_y, batch_y[idx], lam
    else:
        return batch_x, batch_y, None, 1.0

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
    # clean old checkpoints if any (ensure single run ladder)
    for f in os.listdir(CKPT_DIR):
        if f.startswith("ckpt_epoch_"):
            os.remove(os.path.join(CKPT_DIR, f))
    if os.path.exists(FINAL_CKPT):
        os.remove(FINAL_CKPT)

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

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=4, pin_memory=True)

    model = CifarResNet(widen=4).to(device)

    # Use SGD with Nesterov
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)

    # Cosine schedule with warmup
    def lr_lambda(epoch):
        # epoch 0-indexed
        e = epoch + 1
        if e <= WARMUP:
            return e / WARMUP
        else:
            progress = (e - WARMUP) / (EPOCHS - WARMUP)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler()
    criterion_smooth = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    start_wall = time.time()
    print(f"Training start: {start_wall}, epochs={EPOCHS}, batch={BATCH}, lr={LR}, device={device}")

    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_start = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # MixUp/CutMix with 80% prob
            if random.random() < 0.8:
                # need to do mixup/cutmix on GPU tensors
                # reuse function but operate on tensors
                r = random.random()
                if r < 0.5:
                    lam = random.betavariate(0.2, 0.2)
                    idx = torch.randperm(x.size(0), device=device)
                    mixed_x = lam * x + (1 - lam) * x[idx]
                    y_a, y_b = y, y[idx]
                    lam_a = lam
                else:
                    lam = random.betavariate(1.0, 1.0)
                    idx = torch.randperm(x.size(0), device=device)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                    mixed_x = x.clone()
                    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
                    y_a, y_b = y, y[idx]
                    lam_a = lam
                # forward with mixed
                with torch.cuda.amp.autocast():
                    logits = model(mixed_x)
                    loss = lam_a * criterion_smooth(logits, y_a) + (1 - lam_a) * criterion_smooth(logits, y_b)
            else:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion_smooth(logits, y)

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
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        elapsed = time.time() - start_wall
        print(f"Epoch {epoch:03d}/{EPOCHS} acc={acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} time={elapsed:.1f}s")

        # save scripted checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted = torch.jit.script(model.eval())
        scripted.save(ckpt_path)
        # also ensure model back to train for next epoch (script doesn't affect original)
        model.train()

        # early stop if we already hit 94 and have some buffer? Continue to ensure best
        # but we must keep all epochs contiguous 1..N

    # final copy
    last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
    shutil.copy(last_ckpt, FINAL_CKPT)
    total_time = time.time() - start_wall
    print(f"Done total_time={total_time:.1f}s")
    # write metrics
    # find best acc epoch by re-eval? Use last acc as proxy, but we should compute best
    # For metrics, report last acc and total time
    import json
    with open("/workspace/agent_reported_metrics.json","w") as f:
        json.dump({"accuracy": acc, "wall_clock_sec": total_time}, f)
    print(f"Saved final checkpoint and metrics acc={acc}")

if __name__ == "__main__":
    main()
