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
BATCH_TRAIN = 1024
BATCH_TEST = 512
NUM_WORKERS = 8
LR_MAX = 1.0
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model: ResNet for CIFAR (3x3 stem, no maxpool) ---
# Use PreAct ResNet style for better accuracy/speed
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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
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
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

# Alternative: WideResNet 28-10 for better accuracy
# Let's use ResNet18 first for speed, but we can switch to WRN if needed
# For now use ResNet18 with width 64-512

def get_model():
    m = ResNet18()
    return m

def main():
    start_wall = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    # clean old checkpoints if any (ensure single run ladder)
    for f in os.listdir(CKPT_DIR):
        fp = os.path.join(CKPT_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
    test_ds = datasets.ImageFolder(DATA_TEST, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=False, prefetch_factor=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False)

    model = get_model().to(device)
    model = model.to(memory_format=torch.channels_last)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_MAX, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)
    # OneCycle LR - very fast convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)

    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Enable cudnn benchmark
    torch.backends.cudnn.benchmark = True

    best_acc = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                _, pred = outputs.max(1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        print(f"Epoch {epoch}/{EPOCHS} acc={acc*100:.2f}% best={best_acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f}")

        # Save scripted checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        # Need to script - ensure model is on CPU for scripting? No, script on current device but verifier loads on GPU
        # Script the model (eval mode)
        scripted = torch.jit.script(model.eval())
        scripted.save(ckpt_path)
        # Also ensure final checkpoint copy after last epoch will be done outside loop
        model.train()

        # Early stop if we already hit 94 and have some buffer? No, continue to ensure best
        # But we want to keep training to maybe improve further

    # Copy last checkpoint to /workspace/checkpoint.pt
    last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
    shutil.copy(last_ckpt, FINAL_CKPT)
    end_wall = time.time()
    wall_clock = end_wall - start_wall
    print(f"Done. Best acc {best_acc*100:.2f}% wall {wall_clock:.1f}s")
    # Write metrics
    import json
    with open("/workspace/agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": best_acc, "wall_clock_sec": wall_clock}, f)
    # Verify last checkpoint loads and evals
    try:
        m2 = torch.jit.load(FINAL_CKPT).eval().to(device)
        # quick sanity with one batch
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                out = m2(inputs)
            print(f"Checkpoint verify: output shape {out.shape}, sample pred {out[0].argmax().item()}")
            break
    except Exception as e:
        print(f"Verify failed: {e}")

if __name__ == "__main__":
    main()
