import os, time, shutil, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/scratch_ckpt"
EPOCHS = 35
BATCH = 512
NUM_WORKERS = 8
LR = 0.8
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
    def __init__(self, num_classes=10, depth=28, widen=10):
        super().__init__()
        n = (depth - 4)//6
        k = widen
        self.in_ch = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(16*k, n, stride=1)
        self.layer2 = self._make_layer(32*k, n, stride=2)
        self.layer3 = self._make_layer(64*k, n, stride=2)
        self.bn = nn.BatchNorm2d(64*k)
        self.fc = nn.Linear(64*k, num_classes)
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

def rand_bbox(size, lam):
    W = size[-1]; H = size[-2]
    cut_rat = (1. - lam) ** 0.5
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = random.randint(0, W-1); cy = random.randint(0, H-1)
    bbx1 = max(0, cx - cut_w // 2); bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2); bby2 = min(H, cy + cut_h // 2)
    return bbx1, bby1, bbx2, bby2

def mixup_cutmix(batch_x, batch_y, alpha=1.0):
    if random.random() > 0.5:
        lam = random.betavariate(alpha, alpha) if alpha>0 else 1.0
        idx = torch.randperm(batch_x.size(0), device=batch_x.device)
        mixed_x = lam * batch_x + (1 - lam) * batch_x[idx]
        y_a, y_b = batch_y, batch_y[idx]
        return mixed_x, y_a, y_b, lam
    else:
        lam = random.betavariate(alpha, alpha) if alpha>0 else 1.0
        idx = torch.randperm(batch_x.size(0), device=batch_x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(batch_x.size(), lam)
        mixed_x = batch_x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = batch_x[idx, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size(-1) * batch_x.size(-2)))
        y_a, y_b = batch_y, batch_y[idx]
        return mixed_x, y_a, y_b, lam

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        if f.startswith("ckpt_epoch_"):
            os.remove(os.path.join(CKPT_DIR, f))
    device = torch.device("cuda")
    print(f"WRN28-10 BATCH {BATCH} LR {LR} EPOCHS {EPOCHS}")

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
    test_ds = datasets.ImageFolder(DATA_TEST, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = CifarResNet(depth=28, widen=10).to(device)
    model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
    warmup_epochs = 3
    def lr_lambda(epoch):
        e = epoch + 1
        if e <= warmup_epochs:
            return e / warmup_epochs
        else:
            progress = (e - warmup_epochs) / (EPOCHS - warmup_epochs)
            return 0.5 * (1 + math.cos(progress * math.pi))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    start_wall = time.time()
    first_ckpt_time = None
    last_ckpt_time = None
    best_acc = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            if random.random() < 0.5:
                x, y_a, y_b, lam = mixup_cutmix(x, y, alpha=1.0)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
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
        correct=0; total=0
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
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch:03d}/{EPOCHS} acc={acc*100:.2f}% best={best_acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} time={time.time()-t0:.1f}s")
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted = torch.jit.script(model.eval())
        scripted.save(ckpt_path)
        if first_ckpt_time is None:
            first_ckpt_time = time.time()
        last_ckpt_time = time.time()
    total_wall = last_ckpt_time - start_wall if last_ckpt_time else time.time() - start_wall
    print(f"Done best {best_acc*100:.2f}% total_wall {total_wall:.1f}s")

if __name__ == "__main__":
    main()
