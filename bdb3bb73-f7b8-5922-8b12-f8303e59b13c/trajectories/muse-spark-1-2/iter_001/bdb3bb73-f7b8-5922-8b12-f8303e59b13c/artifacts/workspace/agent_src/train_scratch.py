import os, time, shutil, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.multiprocessing.set_sharing_strategy('file_system')

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/scratch_ckpt"
EPOCHS = 30
BATCH_TRAIN = 2048
BATCH_TEST = 1024
LR_MAX = 2.0
WD = 5e-4
LS = 0.1

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda")

class Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 128, 2, 1)
        self.layer2 = self._make(128, 256, 2, 2)
        self.layer3 = self._make(256, 512, 2, 2)
        self.fc = nn.Linear(512, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make(self, in_c, out_c, n, stride):
        layers = [Block(in_c, out_c, stride)]
        for _ in range(1, n):
            layers.append(Block(out_c, out_c, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def main():
    start_wall = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        fp = os.path.join(CKPT_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
    test_ds = datasets.ImageFolder(DATA_TEST, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model = Net().to(device)
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True
    # try compile
    try:
        model = torch.compile(model, mode="max-autotune")
        print("compiled", flush=True)
    except Exception as e:
        print(f"compile fail {e}", flush=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR_MAX, momentum=0.9, weight_decay=WD, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100
    )
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=LS)

    best_acc = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                _, pred = outputs.max(1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch}/{EPOCHS} acc={acc*100:.2f}% best={best_acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.5f}", flush=True)

        # save scripted - unwrap compile
        to_script = model._orig_mod if hasattr(model, '_orig_mod') else model
        to_script.eval()
        scripted = torch.jit.script(to_script)
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(ckpt_path)
        to_script.train()
        model.train()

    end_wall = time.time()
    wall = end_wall - start_wall
    print(f"Done best {best_acc*100:.2f}% wall {wall:.1f}s", flush=True)
    import json
    with open("/tmp/scratch_metrics.json", "w") as f:
        json.dump({"accuracy": best_acc, "wall_clock_sec": wall}, f)

if __name__ == "__main__":
    main()
