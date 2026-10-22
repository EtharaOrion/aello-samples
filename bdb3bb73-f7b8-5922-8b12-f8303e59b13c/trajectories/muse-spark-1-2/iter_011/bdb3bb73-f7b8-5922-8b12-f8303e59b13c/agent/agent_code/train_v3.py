import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 50
BATCH_SIZE = 1024
LR = 1.0
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.makedirs(CKPT_DIR, exist_ok=True)

print("Preloading train data...")
tmp_ds = ImageFolder(DATA_TRAIN)
N = len(tmp_ds)
train_data = torch.empty(N, 3, 32, 32, dtype=torch.uint8)
train_labels = torch.empty(N, dtype=torch.long)
for i, (path, label) in enumerate(tmp_ds.samples):
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    train_data[i] = torch.from_numpy(arr).permute(2,0,1)
    train_labels[i] = label
print(f"Train {train_data.shape}")

print("Preloading test data...")
tmp_ds_test = ImageFolder(DATA_TEST)
Nt = len(tmp_ds_test)
test_data = torch.empty(Nt, 3, 32, 32, dtype=torch.uint8)
test_labels = torch.empty(Nt, dtype=torch.long)
for i, (path, label) in enumerate(tmp_ds_test.samples):
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    test_data[i] = torch.from_numpy(arr).permute(2,0,1)
    test_labels[i] = label
print(f"Test {test_data.shape}")

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.layer1 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128, 1))
        self.layer2 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256, 1))
        self.layer3 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device("cuda")
model = ResNet9().to(device)
model = model.to(memory_format=torch.channels_last)
# Use torch.compile without max-autotune to avoid huge autotune overhead
try:
    model = torch.compile(model)
    print("torch.compile enabled (default)")
except Exception as e:
    print(f"compile failed {e}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)
steps_per_epoch = (N + BATCH_SIZE -1)//BATCH_SIZE
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)
scaler = torch.amp.GradScaler('cuda')
MEAN_CUDA = MEAN.to(device)
STD_CUDA = STD.to(device)

train_ds_tensor = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_ds_tensor, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

def evaluate():
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for i in range(0, Nt, 512):
            x = test_data[i:i+512].to(device, non_blocking=True).float()/255.0
            x = (x - MEAN_CUDA)/STD_CUDA
            x = x.to(memory_format=torch.channels_last)
            y = test_labels[i:i+512].to(device)
            with torch.amp.autocast('cuda'):
                out = model(x)
            correct += (out.argmax(1)==y).sum().item()
            total+=y.size(0)
    return correct/total

# GPU aug helpers - vectorized per-sample crop via unfold
def gpu_augment(x):
    # x: B C 32 32 float [0,1] on GPU
    B = x.size(0)
    # pad 4 reflect -> 40x40
    x = F.pad(x, (4,4,4,4), mode='reflect')  # B C 40 40
    # per-sample random crop using unfold trick
    # unfold to get all 9x9 crops: B C 9 9 32 32
    # Use unfold: x.unfold(2,32,1).unfold(3,32,1) -> B C 9 9 32 32
    # But unfold on 40x40 with kernel 32 stride 1 gives 9 positions
    crops = x.unfold(2, 32, 1).unfold(3, 32, 1)  # B C 9 9 32 32
    # random indices per sample
    ho = torch.randint(0, 9, (B,), device=x.device)
    wo = torch.randint(0, 9, (B,), device=x.device)
    # gather: need to index crops[:, :, ho[i], wo[i]] per i
    # crops is B C 9 9 32 32, we want B C 32 32
    # Use advanced indexing
    # Create batch indices
    b_idx = torch.arange(B, device=x.device)
    # crops[b_idx, :, ho, wo] -> need to handle C dim
    # crops shape: B C 9 9 32 32, indexing with b_idx and ho/wo on dims 0,2,3
    # Use: crops[b_idx, :, ho, wo] not directly possible due to C dim
    # Instead reshape: crops is B C 9 9 32 32, we can do:
    # crops_permuted = crops.permute(0,2,3,1,4,5) -> B 9 9 C 32 32 then index
    # Simpler: use torch.gather or loop over B with vectorized gather via indexing
    # Let's do: crops_selected = crops[b_idx, :, ho, wo] via advanced indexing with broadcasting
    # We need to index dim 0 with b_idx, dim 2 with ho, dim 3 with wo, keep dim1 (C) and 4,5
    # Use: crops[b_idx[:,None], torch.arange(crops.size(1), device=x.device)[None,:], ho[:,None], wo[:,None]] -> complex
    # Easier: flatten 9*9=81 crops and gather
    crops_flat = crops.reshape(B, crops.size(1), 81, 32, 32)  # B C 81 32 32 where 81 = 9*9, order is row-major: idx = ho*9+wo
    idx = ho*9 + wo  # B
    # gather along dim 2
    # Use torch.gather: need to expand idx to match dims
    # crops_flat: B C 81 32 32, idx: B -> need B C 1 32 32
    idx_exp = idx.view(B,1,1,1,1).expand(B, crops.size(1), 1, 32, 32)
    # But gather expects same number of dims, and index on dim 2
    # Use gather
    x_cropped = torch.gather(crops_flat, 2, idx_exp).squeeze(2)  # B C 32 32
    x = x_cropped
    # per-sample horizontal flip
    flip_mask = torch.rand(B, device=x.device) < 0.5
    x_flipped = torch.flip(x, dims=[3])
    mask = flip_mask.view(B,1,1,1)
    x = torch.where(mask, x_flipped, x)
    # Cutout: random 16x16 patch zeroed with p=0.5 per sample, vectorized
    # Generate random center per sample
    cut_mask = torch.rand(B, device=x.device) < 0.5
    if cut_mask.any():
        # For samples where cut, generate h,w center
        # We'll create a mask tensor B 1 32 32 with 0 where cutout
        # Generate h1,w1 per sample
        h_center = torch.randint(0, 32, (B,), device=x.device)
        w_center = torch.randint(0, 32, (B,), device=x.device)
        # Create coordinate grids
        # For each sample, zero out [h-8:h+8, w-8:w+8]
        # Vectorized via broadcasting: create H and W indices
        # Use meshgrid
        # Create mask of ones, then set cutout region to 0
        # We can create a B 32 32 mask via broadcasting
        ys = torch.arange(32, device=x.device).view(1,32,1).expand(B,32,32)
        xs = torch.arange(32, device=x.device).view(1,1,32).expand(B,32,32)
        h1 = (h_center - 8).clamp(0,32).view(B,1,1)
        h2 = (h_center + 8).clamp(0,32).view(B,1,1)
        w1 = (w_center - 8).clamp(0,32).view(B,1,1)
        w2 = (w_center + 8).clamp(0,32).view(B,1,1)
        # mask where inside cutout
        inside = (ys >= h1) & (ys < h2) & (xs >= w1) & (xs < w2)  # B 32 32 bool
        inside = inside & cut_mask.view(B,1,1)
        # Apply: set to 0 where inside
        # x is B C 32 32, inside is B 32 32 -> expand to B 1 32 32
        x = x.masked_fill(inside.unsqueeze(1), 0)
    return x

# Test gpu_augment speed
print("Testing gpu_augment...")
with torch.no_grad():
    dummy = torch.rand(1024,3,32,32, device=device)
    import time as t2
    s=t2.time()
    for _ in range(10):
        _ = gpu_augment(dummy)
    torch.cuda.synchronize()
    print(f"gpu_augment 10 batches: {t2.time()-s:.2f}s")

start_wall = time.time()
first_ckpt_time=None
last_ckpt_time=None
print(f"Training {EPOCHS} epochs steps {steps_per_epoch} batch {BATCH_SIZE}")

for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_start=time.time()
    for xb_uint8, yb in train_loader:
        x = xb_uint8.to(device, non_blocking=True).float()/255.0
        y = yb.to(device, non_blocking=True)
        # GPU augment before normalize
        x = gpu_augment(x)
        # Mixup/CutMix with p=0.5
        if random.random() < 0.5:
            # Mixup alpha 0.2 or CutMix
            if random.random() < 0.5:
                # Mixup
                lam = np.random.beta(0.2, 0.2)
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1-lam) * x[perm]
                # For loss, we need to handle mixup labels: use two losses
                # We'll do mixed loss via criterion with soft labels? Instead do two forward? Simpler: do label mixing via loss interpolation
                # We'll compute loss as lam*CE + (1-lam)*CE_perm
                # So we need to keep y_perm
                y_perm = y[perm]
                # Normalize after mixup
                x = (x - MEAN_CUDA)/STD_CUDA
                x = x.to(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = lam * F.cross_entropy(out, y, label_smoothing=LABEL_SMOOTHING) + (1-lam) * F.cross_entropy(out, y_perm, label_smoothing=LABEL_SMOOTHING)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                continue
            else:
                # CutMix
                lam = np.random.beta(1.0, 1.0)
                perm = torch.randperm(x.size(0), device=device)
                # generate bbox
                W=32; H=32
                cut_rat = np.sqrt(1. - lam)
                cut_w = int(W * cut_rat)
                cut_h = int(H * cut_rat)
                cx = np.random.randint(W)
                cy = np.random.randint(H)
                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)
                x[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
                y_perm = y[perm]
                x = (x - MEAN_CUDA)/STD_CUDA
                x = x.to(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = lam * F.cross_entropy(out, y, label_smoothing=LABEL_SMOOTHING) + (1-lam) * F.cross_entropy(out, y_perm, label_smoothing=LABEL_SMOOTHING)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                continue
        # No mixup path
        x = (x - MEAN_CUDA)/STD_CUDA
        x = x.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    to_script = model._orig_mod if hasattr(model, '_orig_mod') else model
    to_script.eval()
    scripted = torch.jit.script(to_script)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    to_script.train()
    if hasattr(model, '_orig_mod'):
        model.train()
    last_ckpt_time=time.time()
    if first_ckpt_time is None:
        first_ckpt_time=last_ckpt_time
    acc=evaluate()
    lr_cur=optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:03d}/{EPOCHS} acc={acc*100:.2f}% lr={lr_cur:.4f} epoch_time={time.time()-epoch_start:.1f}s wall={time.time()-start_wall:.1f}s")

last_ckpt=os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
shutil.copy(last_ckpt, FINAL_CKPT)
print(f"Done copy {last_ckpt} -> {FINAL_CKPT}")
total_wall=last_ckpt_time-start_wall
best_acc=evaluate()
print(f"Final acc {best_acc*100:.2f}% wall {total_wall:.1f}s")
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": float(best_acc), "wall_clock_sec": float(total_wall)}, f)
