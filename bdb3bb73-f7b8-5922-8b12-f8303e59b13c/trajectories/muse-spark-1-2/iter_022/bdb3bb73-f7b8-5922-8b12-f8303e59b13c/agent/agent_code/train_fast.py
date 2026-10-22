import os, time, shutil, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

EPOCHS = 50
BATCH = 1024
LR = 1.0
WD = 5e-4
WARMUP = 5
CKPT_DIR = "/workspace/periodic_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
device = torch.device("cuda")

# Load data to GPU as uint8 tensors for fast GPU augmentation
# Use ImageFolder to get paths, then load all images into memory
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

# Preload train/test as tensors on CPU first, then move batches to GPU with augmentation
# Instead, preload as numpy and do GPU aug

# Simple: load all images as tensors (3,32,32) uint8 on CPU, then each epoch do GPU aug
print("Preloading data...")
train_ds_raw = ImageFolder("/workspace/data/cifar10/train")
test_ds_raw = ImageFolder("/workspace/data/cifar10/test")

def load_folder(ds):
    imgs = []
    labels = []
    for path, label in ds.samples:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)  # HWC
        imgs.append(arr)
        labels.append(label)
    imgs = np.stack(imgs)  # N,H,W,C
    labels = np.array(labels, dtype=np.int64)
    return torch.from_numpy(imgs), torch.from_numpy(labels)

train_imgs, train_labels = load_folder(train_ds_raw)  # N,32,32,3 uint8
test_imgs, test_labels = load_folder(test_ds_raw)
print(f"Train {train_imgs.shape} Test {test_imgs.shape}")

# Normalize constants for GPU
mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1,3,1,1)

# Move test to GPU normalized
test_imgs_gpu = test_imgs.permute(0,3,1,2).to(device, dtype=torch.float32) / 255.0
test_imgs_gpu = (test_imgs_gpu - mean) / std
test_labels_gpu = test_labels.to(device)

# Train stays on CPU as uint8, we batch and move to GPU per batch with aug
train_imgs_cpu = train_imgs  # N,H,W,C uint8
train_labels_cpu = train_labels

# Model
class Block(nn.Module):
    def __init__(self, ci, co, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ci)
        self.conv1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(co)
        self.conv2 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or ci != co:
            self.shortcut = nn.Sequential(nn.Conv2d(ci, co, 1, stride, bias=False), nn.BatchNorm2d(co))
    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WRN(nn.Module):
    def __init__(self, depth=28, widen=10):
        super().__init__()
        n = (depth-4)//6; k=widen
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.layer1 = self._make(16, 16*k, n, 1)
        self.layer2 = self._make(16*k, 32*k, n, 2)
        self.layer3 = self._make(32*k, 64*k, n, 2)
        self.bn = nn.BatchNorm2d(64*k)
        self.fc = nn.Linear(64*k, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'); nn.init.constant_(m.bias,0)
    def _make(self, ci, co, n, stride):
        layers=[Block(ci,co,stride)]
        for _ in range(1,n): layers.append(Block(co,co,1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out=self.conv1(x)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=F.relu(self.bn(out))
        out=F.adaptive_avg_pool2d(out,1).flatten(1)
        return self.fc(out)

model = WRN(28,10).to(device)
model = model.to(memory_format=torch.channels_last)
model = torch.compile(model)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
def lr_fn(epoch):
    if epoch < WARMUP: return (epoch+1)/WARMUP
    p=(epoch-WARMUP)/(EPOCHS-WARMUP)
    return 0.5*(1+math.cos(math.pi*p))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
scaler = torch.amp.GradScaler('cuda')

# GPU augmentation function
def gpu_augment(x):  # x: B,3,32,32 float in [0,1] before normalize, already on GPU
    B = x.size(0)
    # Random crop with padding 4: pad to 40x40 then crop
    x = F.pad(x, (4,4,4,4), mode='reflect')
    # random crop offsets
    h_off = torch.randint(0, 9, (B,), device=x.device)
    w_off = torch.randint(0, 9, (B,), device=x.device)
    # Use grid sampling or manual slicing - manual loop is slow, use batched indexing
    # Efficient: use torch.stack with slicing via unfold? Simpler: use for loop but B=1024, 1024 iterations is okay? No.
    # Use vectorized crop via affine grid
    # Alternative: use F.unfold trick or just do random crop via rolling
    # Fastest: generate crops via indexing with advanced indexing
    # We'll do: create output tensor and fill via loop in Python but with torch ops it's still 1024 iterations of slicing which is okay on GPU (1024*50 epochs = 50k iterations, each slice is fast)
    # Let's try vectorized using torch.nn.functional.affine? Simpler to use Kornia-style: use random ints and gather
    # Implement via batched crop using unfold
    # x padded: B,3,40,40 -> unfold to get 9x9 crops? Actually we need random per sample.
    # Quick hack: use for loop with torch - it's okay, 1024 iterations per batch * 49 batches = 50k slices per epoch, each slice is a kernel launch -> too many launches.
    # Better: use random horizontal flip + cutout only, skip crop for speed? But crop is important.
    # Let's implement crop via torch.randint and then use torch.gather with precomputed indices
    # Simpler: use torchvision's GPU aug via batched operations using torch.roll? No.
    # Let's do efficient crop: generate random offsets and use grid_sample
    # Create affine matrices for crop
    # For each sample, we want to crop 32x32 from 40x40 at (h_off, w_off)
    # Normalized coordinates: use affine_grid
    # theta for crop: scale 32/40=0.8, translation = (off - 4)/16 *0.8? Let's compute.
    # Instead simpler: just do pad+random crop on CPU before? But we want GPU speed.
    # Let's just do flip + normalize on GPU and do crop on CPU via preloading with padding? Hmm.
    # For now, implement simple GPU crop via indexing with a single kernel using torch's advanced indexing
    # Create index tensors
    out = torch.empty(B, 3, 32, 32, device=x.device, dtype=x.dtype)
    for i in range(B):
        out[i] = x[i, :, h_off[i]:h_off[i]+32, w_off[i]:w_off[i]+32]
    # Horizontal flip
    flip_mask = torch.rand(B, device=x.device) < 0.5
    out[flip_mask] = torch.flip(out[flip_mask], dims=[3])
    return out

# Clean checkpoints
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

# Training loop with GPU aug
N = train_imgs_cpu.shape[0]
indices = torch.arange(N)

t0 = time.time()
print(f"Training {EPOCHS}ep B{BATCH} LR{LR} N={N}")

for epoch in range(EPOCHS):
    model.train()
    te = time.time()
    perm = torch.randperm(N)
    # iterate batches
    for i in range(0, N, BATCH):
        idx = perm[i:i+BATCH]
        # Load batch HWC uint8 -> B,3,32,32 float [0,1] on GPU
        batch_hwc = train_imgs_cpu[idx]  # B,H,W,C
        batch = batch_hwc.permute(0,3,1,2).to(device, non_blocking=True).float() / 255.0
        labels = train_labels_cpu[idx].to(device, non_blocking=True)
        # GPU augment
        batch = gpu_augment(batch)
        # Normalize
        batch = (batch - mean) / std
        batch = batch.to(memory_format=torch.channels_last)

        # CutMix/Mixup 50% chance
        lam = 1.0
        y1 = labels
        y2 = labels
        r = random.random()
        if r < 0.3:  # cutmix
            lam_b = np.random.beta(1.0, 1.0)
            perm2 = torch.randperm(batch.size(0), device=device)
            y2 = labels[perm2]
            W = 32; H = 32
            cut_rat = math.sqrt(1 - lam_b)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            cx = random.randint(0, W-1)
            cy = random.randint(0, H-1)
            x1 = max(cx - cut_w//2, 0); x2 = min(cx + cut_w//2, W)
            y1c = max(cy - cut_h//2, 0); y2c = min(cy + cut_h//2, H)
            batch[:, :, y1c:y2c, x1:x2] = batch[perm2, :, y1c:y2c, x1:x2]
            lam = 1 - ((x2-x1)*(y2c-y1c)/(W*H))
        elif r < 0.5:  # mixup
            lam = float(np.random.beta(0.4, 0.4))
            perm2 = torch.randperm(batch.size(0), device=device)
            y2 = labels[perm2]
            batch = lam * batch + (1-lam) * batch[perm2]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(batch)
            if lam == 1.0:
                loss = criterion(out, y1)
            else:
                loss = lam * criterion(out, y1) + (1-lam) * criterion(out, y2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

    # Eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, test_imgs_gpu.size(0), 512):
            xb = test_imgs_gpu[i:i+512].to(memory_format=torch.channels_last)
            yb = test_labels_gpu[i:i+512]
            with torch.amp.autocast('cuda'):
                out = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    acc = correct/total
    dt = time.time()-te
    wall = time.time()-t0
    print(f"Ep {epoch+1:02d} acc {acc:.4f} lr {optimizer.param_groups[0]['lr']:.4f} {dt:.1f}s wall {wall:.1f}s")
    raw = model._orig_mod if hasattr(model,'_orig_mod') else model
    raw.eval()
    torch.jit.script(raw).save(os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch+1:04d}.pt"))
    raw.train()

shutil.copy(os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt"), "/workspace/checkpoint.pt")
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": acc, "wall_clock_sec": time.time()-t0}, f)
print(f"Done acc {acc:.4f} wall {time.time()-t0:.1f}s")
