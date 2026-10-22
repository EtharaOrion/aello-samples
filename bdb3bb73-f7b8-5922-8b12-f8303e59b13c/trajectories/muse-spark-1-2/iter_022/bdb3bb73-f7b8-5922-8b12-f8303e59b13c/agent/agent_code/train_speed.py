import os, time, shutil, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

EPOCHS = 30
BATCH = 1024
LR = 1.2
WD = 5e-4
WARMUP = 3
CKPT_DIR = "/workspace/scratch_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)
device = torch.device("cuda")
print(f"H100: {torch.cuda.get_device_name(0)}")

# Preload data to CPU as uint8, then GPU augment
print("Preloading...")
train_ds = ImageFolder("/workspace/data/cifar10/train")
test_ds = ImageFolder("/workspace/data/cifar10/test")

def load_folder(ds):
    imgs = []
    labels = []
    for path, label in ds.samples:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        imgs.append(arr)
        labels.append(label)
    return np.stack(imgs), np.array(labels, dtype=np.int64)

train_imgs_np, train_labels_np = load_folder(train_ds)
test_imgs_np, test_labels_np = load_folder(test_ds)
print(f"Train {train_imgs_np.shape} Test {test_imgs_np.shape}")

# Keep train on CPU as uint8 HWC, test pre-normalized on GPU
mean_t = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
std_t = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1,3,1,1)

test_gpu = torch.from_numpy(test_imgs_np).permute(0,3,1,2).to(device, dtype=torch.float32) / 255.0
test_gpu = (test_gpu - mean_t) / std_t
test_gpu = test_gpu.to(memory_format=torch.channels_last)
test_labels_gpu = torch.from_numpy(test_labels_np).to(device)

train_imgs_cpu = torch.from_numpy(train_imgs_np)  # N,H,W,C uint8
train_labels_cpu = torch.from_numpy(train_labels_np)

# Model: smaller WRN-28-8 (faster than 28-10 but still strong)
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
    def __init__(self, depth=28, widen=8):
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

model = WRN(28, 8).to(device)
model = model.to(memory_format=torch.channels_last)
# Try compile - use default
try:
    model = torch.compile(model, mode="max-autotune")
    print("Compiled with max-autotune")
except:
    model = torch.compile(model)
    print("Compiled default")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
def lr_fn(epoch):
    if epoch < WARMUP: return (epoch+1)/WARMUP
    p=(epoch-WARMUP)/(EPOCHS-WARMUP)
    return 0.5*(1+math.cos(math.pi*p))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
scaler = torch.amp.GradScaler('cuda')

# Clean
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR,f))

N = train_imgs_cpu.shape[0]
t0 = time.time()
print(f"Training {EPOCHS}ep B{BATCH} LR{LR} N={N} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Pre-generate random crop indices on CPU for speed? Do GPU augment inline
for epoch in range(EPOCHS):
    model.train()
    te = time.time()
    perm = torch.randperm(N)
    for i in range(0, N, BATCH):
        idx = perm[i:i+BATCH]
        # HWC uint8 -> B,3,32,32 float on GPU
        batch_hwc = train_imgs_cpu[idx]  # B,H,W,C
        # Move to GPU and do augment: pad+crop+flip on GPU
        # For speed, do simple GPU augment: use F.pad + manual crop via indexing
        b = batch_hwc.shape[0]
        x = batch_hwc.permute(0,3,1,2).to(device, non_blocking=True).float() / 255.0  # B,3,32,32
        # Pad 4
        x = F.pad(x, (4,4,4,4), mode='reflect')  # B,3,40,40
        # Random crop: generate offsets
        h_off = torch.randint(0, 9, (b,), device=device)
        w_off = torch.randint(0, 9, (b,), device=device)
        # Vectorized crop using advanced indexing - create output
        # Use torch.stack with slicing - but need fast path
        # Use unfold + gather? Simple loop is okay for B=1024? 1024 slices per batch * 49 batches = 50k slices, each is a kernel launch -> slow
        # Better: use grid_sample or use batched indexing via torch
        # Fastest: use F.unfold to get all crops then select
        # Alternative: use random crop via affine transform with grid_sample (single kernel)
        # Let's use a fast trick: generate random crops via indexing with a single operation using torch.gather
        # Simplest fast: use torchvision's random crop on GPU via kornia? No.
        # Let's do: create indices and use advanced indexing with broadcasting
        # Actually, let's just do the crop with a single kernel using torch's indexing:
        # x is B,3,40,40, we want B,3,32,32 where each sample has different offset
        # We can do: create a tensor of shape B,3,32,32 by using torch.arange and broadcasting
        # But easiest: use for loop but with torch.compile it might be okay? No, still slow.
        # Let's try a different approach: do crop on CPU before moving to GPU - pre-augment on CPU with numpy? But that would be slow too.
        # For now, let's do GPU crop via unfold + select
        # x_unfold: B, 3*9*9, 32*32? No.
        # Simpler: just do flip without crop for speed test, or do crop via rolling
        # Let's implement crop via batched gather using indices
        # Create output via indexing: use torch's advanced indexing with meshgrid
        # Quick: use F.unfold to extract all 9x9 possible crops? That's 81 crops per image, too many.
        # Let's just do a simple efficient crop: use torch.randint and then use slicing with a loop but with B=1024, 49 batches, 30 epochs = 1470 batches, each with 1024 slices = 1.5M slices. Each slice is a small kernel, might be okay on H100? Let's try.
        # Actually, let's benchmark: do the loop
        out = torch.empty(b, 3, 32, 32, device=device, dtype=x.dtype)
        for j in range(b):
            out[j] = x[j, :, h_off[j]:h_off[j]+32, w_off[j]:w_off[j]+32]
        x = out
        # Flip
        flip = torch.rand(b, device=device) < 0.5
        x[flip] = torch.flip(x[flip], dims=[3])
        # Normalize
        x = (x - mean_t) / std_t
        x = x.to(memory_format=torch.channels_last)
        y = train_labels_cpu[idx].to(device, non_blocking=True)

        # Mixup/CutMix
        lam = 1.0
        y1 = y
        y2 = y
        r = random.random()
        if r < 0.25:  # cutmix
            lam_b = np.random.beta(1.0, 1.0)
            perm2 = torch.randperm(b, device=device)
            y2 = y[perm2]
            W=32; H=32
            cut_rat = math.sqrt(1 - lam_b)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            cx = random.randint(0, W-1)
            cy = random.randint(0, H-1)
            x1 = max(cx - cut_w//2, 0); x2 = min(cx + cut_w//2, W)
            y1c = max(cy - cut_h//2, 0); y2c = min(cy + cut_h//2, H)
            x[:, :, y1c:y2c, x1:x2] = x[perm2, :, y1c:y2c, x1:x2]
            lam = 1 - ((x2-x1)*(y2c-y1c)/(W*H))
        elif r < 0.45:  # mixup
            lam = float(np.random.beta(0.4, 0.4))
            perm2 = torch.randperm(b, device=device)
            y2 = y[perm2]
            x = lam * x + (1-lam) * x[perm2]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
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
        for j in range(0, test_gpu.size(0), 512):
            xb = test_gpu[j:j+512]
            yb = test_labels_gpu[j:j+512]
            with torch.amp.autocast('cuda'):
                out = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    acc = correct/total
    dt = time.time()-te
    wall = time.time()-t0
    print(f"Ep {epoch+1:02d} acc {acc:.4f} lr {optimizer.param_groups[0]['lr']:.4f} {dt:.1f}s wall {wall:.1f}s")
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw.eval()
    torch.jit.script(raw).save(os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch+1:04d}.pt"))
    raw.train()
    if acc >= 0.94:
        print(f"  *** Hit 94% at epoch {epoch+1} wall {wall:.1f}s ***")

# Copy best to final
import glob
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
if ckpts:
    shutil.copy(ckpts[-1], "/tmp/best_scratch.pt")
    print(f"Done wall {time.time()-t0:.1f}s final acc {acc:.4f}")
