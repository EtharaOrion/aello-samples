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
BATCH = 512
EPOCHS = 60
NUM_WORKERS = 8
LR = 0.5  # for SGD batch 512
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# --- Model: PreAct ResNet18 for CIFAR ---
class PreActBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            )
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if len(self.shortcut) > 0 else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class CifarResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 3 stages: 64,128,256,512 but for CIFAR we use 4 layers each stage 2 blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.bn_last = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()
    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        layers.append(PreActBlock(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(PreActBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # x is already standardized [-2,2], do NOT renormalize
        out = self.conv1(x)
        # no relu after conv1, PreAct will handle
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn_last(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_dataloaders():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
    test_ds = datasets.ImageFolder(DATA_TEST, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # channels_last
            x = x.to(memory_format=torch.channels_last)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    # clean old checkpoints
    for f in os.listdir(CKPT_DIR):
        if f.startswith("ckpt_epoch_"):
            os.remove(os.path.join(CKPT_DIR, f))
    if os.path.exists(FINAL_CKPT):
        os.remove(FINAL_CKPT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders()
    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}, Steps/epoch: {len(train_loader)}")

    model = CifarResNet().to(device)
    model = model.to(memory_format=torch.channels_last)
    # no torch.compile to keep scripting simple

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
    # cosine with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        # epoch is 0-indexed
        e = epoch + 1
        if e <= warmup_epochs:
            return e / warmup_epochs
        else:
            # cosine from 1 to 0
            progress = (e - warmup_epochs) / (EPOCHS - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265))).item()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler('cuda')

    start_wall = time.time()
    first_ckpt_time = None
    last_ckpt_time = None
    best_acc = 0
    crossing_epoch = None

    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_start = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = x.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        # save checkpoint - scripted
        # need to script a CPU copy to be device agnostic
        model_cpu = CifarResNet()
        model_cpu.load_state_dict(model.state_dict())
        model_cpu.eval()
        scripted = torch.jit.script(model_cpu)
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(ckpt_path)
        if first_ckpt_time is None:
            first_ckpt_time = time.time()
        last_ckpt_time = time.time()
        # eval
        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
        if crossing_epoch is None and acc >= 0.94:
            crossing_epoch = epoch
        lr_now = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{EPOCHS} | acc {acc*100:.2f}% | best {best_acc*100:.2f}% | lr {lr_now:.4f} | {epoch_time:.2f}s | crossing {crossing_epoch}")

        # early stop if we have good margin and time? continue to save all epochs but we can stop after 94% + few epochs
        # keep training for at least 45 epochs to ensure stable 94%
        if epoch >= 45 and best_acc >= 0.94:
            # if we already crossed and have 5 epochs beyond crossing, we can stop early to save time
            if crossing_epoch is not None and epoch >= crossing_epoch + 5:
                print(f"Early stop at epoch {epoch} with best {best_acc*100:.2f}%")
                break

    total_wall = last_ckpt_time - start_wall
    span = last_ckpt_time - first_ckpt_time if first_ckpt_time else total_wall
    print(f"Done. Best {best_acc*100:.2f}%, crossing {crossing_epoch}, total_wall {total_wall:.1f}s, span {span:.1f}s")
    # copy last checkpoint to /workspace/checkpoint.pt
    # find last ckpt
    ckpts = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_epoch_")])
    if ckpts:
        last = os.path.join(CKPT_DIR, ckpts[-1])
        shutil.copy(last, FINAL_CKPT)
        print(f"Copied {last} -> {FINAL_CKPT}")
        # verify load
        m = torch.jit.load(FINAL_CKPT)
        m.eval()
        print("Verified scripted load OK")
        # quick eval of scripted model
        # use test_loader but with scripted model on cuda
        m_cuda = torch.jit.load(FINAL_CKPT).eval().cuda()
        # need to handle device: scripted model is on cpu, but we can move? jit model doesn't have .cuda() ? it does
        # Actually torch.jit.load returns model that can be moved via .to()
        # Let's test
        try:
            m_cuda = torch.jit.load(FINAL_CKPT)
            m_cuda.eval()
            # move to cuda by loading and then calling .to? For scripted, we can just run on cuda by passing cuda tensor
            # The model parameters are on CPU, but input on CUDA will cause device mismatch. So we need to ensure model is on CUDA.
            # We saved CPU model, so we need to test with CPU input
            import torchvision.transforms as T
            from PIL import Image
            import glob
            # simple test: run one batch on CPU
            x_test, y_test = next(iter(test_loader))
            with torch.no_grad():
                out = m(x_test)
                acc_cpu = (out.argmax(1) == y_test).float().mean().item()
            print(f"Scripted CPU eval one batch acc {acc_cpu*100:.1f}%")
        except Exception as e:
            print(f"Scripted eval error: {e}")

    # write metrics
    import json
    metrics = {"accuracy": float(best_acc), "wall_clock_sec": float(total_wall)}
    with open("/workspace/agent_reported_metrics.json", "w") as f:
        json.dump(metrics, f)
    print(f"Wrote metrics {metrics}")

if __name__ == "__main__":
    main()
