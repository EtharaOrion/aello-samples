import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 50
BATCH_SIZE = 1024
NUM_WORKERS = 8
LR = 1.2
WD = 5e-4
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 3

torch.backends.cudnn.benchmark = True

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
        return F.relu(out)

class FastResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 3, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        layers.append(Block(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(Block(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_dataloaders():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_ds = torchvision.datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
    test_ds = torchvision.datasets.ImageFolder(DATA_TEST, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        if f.startswith("ckpt_epoch_"):
            os.remove(os.path.join(CKPT_DIR, f))
    if os.path.exists(FINAL_CKPT):
        os.remove(FINAL_CKPT)
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    train_loader, test_loader = get_dataloaders()
    print(f"Train {len(train_loader.dataset)} Test {len(test_loader.dataset)} Steps/epoch {len(train_loader)}")
    model = FastResNet().to(device)
    model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD, nesterov=True)
    def lr_lambda(epoch):
        e = epoch + 1
        if e <= WARMUP_EPOCHS:
            return e / WARMUP_EPOCHS
        else:
            progress = (e - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)).item())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler('cuda')
    start_wall = time.time()
    first_ckpt_time = None
    last_ckpt_time = None
    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_start = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Mixup/CutMix 50% of batches
            do_cutmix = random.random() < 0.5
            do_mixup = (not do_cutmix) and (random.random() < 0.3)
            if do_cutmix:
                lam = random.betavariate(1.0, 1.0)
                b = x.size(0)
                index = torch.randperm(b, device=device)
                y_a, y_b = y, y[index]
                W = x.size(3); H = x.size(2)
                cut_rat = (1 - lam) ** 0.5
                cut_w = int(W * cut_rat)
                cut_h = int(H * cut_rat)
                cx = random.randint(0, W-1)
                cy = random.randint(0, H-1)
                bbx1 = max(cx - cut_w // 2, 0)
                bby1 = max(cy - cut_h // 2, 0)
                bbx2 = min(cx + cut_w // 2, W)
                bby2 = min(cy + cut_h // 2, H)
                if bbx2 > bbx1 and bby2 > bby1:
                    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif do_mixup:
                lam = random.betavariate(1.0, 1.0)
                b = x.size(0)
                index = torch.randperm(b, device=device)
                y_a, y_b = y, y[index]
                x = lam * x + (1 - lam) * x[index]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{EPOCHS} lr={lr:.4f} acc={acc*100:.2f}% time={time.time()-epoch_start:.1f}s")
        # save checkpoint: scripted model (raw)
        model.eval()
        cpu_model = FastResNet()
        cpu_model.load_state_dict(model.state_dict())
        cpu_model.eval()
        scripted = torch.jit.script(cpu_model)
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(ckpt_path)
        if first_ckpt_time is None:
            first_ckpt_time = time.time()
        last_ckpt_time = time.time()
        # verify
        try:
            m2 = torch.jit.load(ckpt_path)
            m2.eval()
            with torch.no_grad():
                dummy = torch.randn(2,3,32,32)
                _ = m2(dummy)
        except Exception as e:
            print(f"Verify failed {e}")
        model.train()
    src = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
    shutil.copy(src, FINAL_CKPT)
    total_wall = last_ckpt_time - start_wall
    print(f"Done total {total_wall:.1f}s")
    final_acc = evaluate(model, test_loader, device)
    print(f"Final acc {final_acc*100:.2f}%")
    import json
    with open("/workspace/agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": float(final_acc), "wall_clock_sec": float(total_wall)}, f)

if __name__ == "__main__":
    main()
