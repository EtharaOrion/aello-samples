import os, time, shutil, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

EPOCHS = 60
BATCH = 1024
LR = 1.2
WD = 5e-4
CKPT_DIR = "/workspace/periodic_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

device = torch.device("cuda")
print(torch.cuda.get_device_name(0))

# --- Preload data to RAM as tensors ---
# Load all train images as uint8 CHW tensors
import pathlib
train_root = pathlib.Path("/workspace/data/cifar10/train")
test_root = pathlib.Path("/workspace/data/cifar10/test")
classes = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
class_to_idx = {c:i for i,c in enumerate(classes)}
print("Classes:", classes)

def load_folder(root):
    imgs = []
    labels = []
    for cls in classes:
        idx = class_to_idx[cls]
        for p in (root/cls).glob("*.png"):
            # Load as PIL then to tensor uint8
            img = Image.open(p).convert("RGB")
            # Convert to tensor uint8 CHW
            import numpy as np
            arr = np.array(img)  # HWC uint8
            t = torch.from_numpy(arr).permute(2,0,1)  # CHW
            imgs.append(t)
            labels.append(idx)
    imgs = torch.stack(imgs)  # NCHW uint8
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels

print("Loading train...")
train_imgs, train_labels = load_folder(train_root)
print(f"Train: {train_imgs.shape}, {train_labels.shape}")
print("Loading test...")
test_imgs, test_labels = load_folder(test_root)
print(f"Test: {test_imgs.shape}")

# Normalize constants for GPU augmentation
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

# Keep test as normalized float on GPU for fast eval
test_imgs_f = test_imgs.float().div(255.0)
# We'll normalize on the fly in eval loop

# Custom dataset that returns uint8 CHW + label, augmentation done in collate/GPU
class FastCIFAR(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs  # NCHW uint8
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

train_ds = FastCIFAR(train_imgs, train_labels)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

# For test, use simple dataset
test_ds = torch.utils.data.TensorDataset(test_imgs_f, test_labels)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

# --- Model ---
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False)
        else:
            self.convShortcut = None
    def forward(self, x):
        if self.equalInOut:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
            return out + x
        else:
            out = self.relu1(self.bn1(x))
            out = self.conv1(out)
            out = self.relu2(self.bn2(out))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
            assert self.convShortcut is not None
            return out + self.convShortcut(x)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i==0 and in_planes or out_planes, out_planes, i==0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropRate=0.0, num_classes=10):
        super().__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

model = WideResNet(28, 10, dropRate=0.0).to(device)
model = model.to(memory_format=torch.channels_last)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Verify scripting
s = torch.jit.script(model)
print("jit.script OK")
del s

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler('cuda')

mean_d = mean.to(device)
std_d = std.to(device)

def augment_batch(x):  # x: NCHW float [0,1] on GPU
    # RandomCrop with padding 4
    # x is NCHW float [0,1] on GPU, we pad and crop
    # Use F.pad then random crop via indexing
    # For speed, do batched random crop using torch
    N = x.size(0)
    # pad 4 on each side -> 40x40
    x_padded = F.pad(x, (4,4,4,4), value=0)  # actually pad does (left,right,top,bottom) for 4D? Use F.pad with 4 values
    # Random crop: generate random offsets
    # Use torch.randint for each sample
    # To avoid loop, we can use unfold or just loop (N=1024, loop is okay but we can vectorize)
    # Simple loop on GPU is still fast enough? Let's do vectorized via grid_sample or just use torch random crop via indexing
    # We'll do per-sample crop using random ints on CPU then slice on GPU - need to move to GPU slicing
    # Instead, do augmentation on GPU with batched operation: create random crops via torch
    # Fastest: use torch's random crop by generating offsets and using advanced indexing
    # We'll implement with a loop in Python but slicing is on GPU (fast)
    crops = torch.empty_like(x)
    for i in range(N):
        h = random.randint(0, 8)
        w = random.randint(0, 8)
        crops[i] = x_padded[i, :, h:h+32, w:w+32]
    x = crops
    # Random horizontal flip
    flip_mask = torch.rand(N, device=x.device) < 0.5
    # flip where mask true
    if flip_mask.any():
        x[flip_mask] = x[flip_mask].flip(dims=[3])
    # Normalize
    x = (x - mean_d) / std_d
    # RandomErasing approx: with p=0.25, erase random patch
    # We'll do simple Cutout: erase 8x8 patch with p=0.25
    # For speed, skip for now or do vectorized
    # Do random erasing with p=0.25
    erase_mask = torch.rand(N, device=x.device) < 0.25
    if erase_mask.any():
        for i in torch.where(erase_mask)[0]:
            h = random.randint(0, 24)
            w = random.randint(0, 24)
            x[i, :, h:h+8, w:w+8] = 0  # after normalize, 0 corresponds to mean
    return x

def mixup_data(x, y, alpha=0.2):
    lam = random.betavariate(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def cutmix_data(x, y, alpha=1.0):
    lam = random.betavariate(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    W, H = x.size(2), x.size(3)
    cut_rat = (1 - lam) ** 0.5
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

start_wall = time.time()

for epoch in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    for imgs_u8, targets in train_loader:
        # imgs_u8: NCHW uint8 on CPU pinned, targets on CPU
        # Move to GPU and convert to float [0,1]
        x = imgs_u8.to(device, non_blocking=True).float().div(255.0).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        # Augment on GPU
        x = augment_batch(x)
        # Mixup/CutMix
        r = random.random()
        use_mix = r < 0.5
        is_cutmix = r < 0.25
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            if use_mix:
                if is_cutmix:
                    x, y_a, y_b, lam = cutmix_data(x, targets, alpha=1.0)
                else:
                    x, y_a, y_b, lam = mixup_data(x, targets, alpha=0.2)
                outputs = model(x)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(x)
                loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    epoch_train = time.time() - t0

    # eval
    model.eval()
    correct = total = 0
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for imgs_f, targets in test_loader:
                # imgs_f is [0,1] float on CPU, need normalize
                x = imgs_f.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                x = (x - mean_d) / std_d
                targets = targets.to(device, non_blocking=True)
                outputs = model(x)
                _, pred = outputs.max(1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
    acc = correct / total
    total_time = time.time() - start_wall
    print(f"Epoch {epoch}/{EPOCHS} | acc {acc*100:.2f}% | lr {optimizer.param_groups[0]['lr']:.4f} | train {epoch_train:.1f}s | total {total_time:.1f}s")

    # save scripted
    model.eval()
    scripted = torch.jit.script(model)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    loaded = torch.jit.load(ckpt_path, map_location="cpu")
    loaded.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, 32, 32)
        _ = loaded(dummy)
    print(f"  -> saved {ckpt_path}")
    model.train()
    if acc >= 0.94:
        print(f"*** Hit 94% at epoch {epoch} ***")

shutil.copy(os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt"), "/workspace/checkpoint.pt")
wall = time.time() - start_wall
with open("/workspace/agent_reported_metrics.json", "w") as f:
    json.dump({"accuracy": acc, "wall_clock_sec": wall}, f)
print(f"Done wall={wall:.1f}s acc={acc:.4f}")
