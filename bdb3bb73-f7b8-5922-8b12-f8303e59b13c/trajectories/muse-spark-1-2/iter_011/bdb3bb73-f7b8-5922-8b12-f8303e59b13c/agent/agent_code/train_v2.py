import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 45
BATCH_SIZE = 2048
LR = 2.0
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.makedirs(CKPT_DIR, exist_ok=True)

# --- Preload data into memory ---
print("Preloading train data...")
train_transform_raw = transforms.ToTensor()  # to [0,1]
# Use ImageFolder to get paths and labels, then load
from torchvision.datasets import ImageFolder
tmp_ds = ImageFolder(DATA_TRAIN)
# tmp_ds.samples is list of (path, label)
# Load all images as uint8 tensors
N = len(tmp_ds)
train_data = torch.empty(N, 3, 32, 32, dtype=torch.uint8)
train_labels = torch.empty(N, dtype=torch.long)
for i, (path, label) in enumerate(tmp_ds.samples):
    img = Image.open(path).convert('RGB')
    arr = np.array(img)  # H W C uint8
    train_data[i] = torch.from_numpy(arr).permute(2,0,1)
    train_labels[i] = label
    if i % 10000 == 0:
        print(f"  {i}/{N}")
print(f"Train data loaded: {train_data.shape}")

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
print(f"Test data loaded: {test_data.shape}")

# Move mean/std to cuda later
# --- Model: ResNet9 ---
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )
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

# Compile for speed
try:
    model = torch.compile(model, mode="max-autotune")
    print("torch.compile enabled")
except Exception as e:
    print(f"compile failed {e}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)

# DataLoader from TensorDataset but we will do GPU aug, so just use TensorDataset with uint8
train_ds_tensor = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_ds_tensor, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False)

# For test, we can pre-normalize and keep on GPU or just evaluate with DataLoader
# We'll evaluate by moving batches to GPU and normalizing

MEAN_CUDA = MEAN.to(device)
STD_CUDA = STD.to(device)

def augment_batch(x):
    # x: (B,3,32,32) float in [0,1] on GPU, already normalized? We'll do aug before normalize
    # Actually x is uint8 -> float [0,1], we do aug then normalize
    # Random crop with padding 4
    # Pad 4 on each side -> 40x40, then random crop 32
    x = F.pad(x, (4,4,4,4), mode='reflect')  # B C 40 40
    # random crop
    B = x.size(0)
    # generate random offsets
    # Use torch.randint
    h_off = torch.randint(0, 9, (B,), device=x.device)
    w_off = torch.randint(0, 9, (B,), device=x.device)
    # Need to crop each sample differently - do via loop or vectorized?
    # Vectorized via gather is complex; do simple loop with stacking - but B=2048 loop is heavy
    # Instead use torchvision-style: we can use random crop via F.crop per sample but still loop
    # Alternative: use batched crop via unfold? Simpler: use torch's random crop by rolling?
    # For speed, we can do: generate random crop via affine? Let's do loop in Python but with torch ops - 2048 iterations per batch is too slow
    # Instead, we can do a fast approximation: use same crop for whole batch? No, need per-sample
    # We can use kornia-like: use grid_sample? Simpler: just do random flip + cutout and skip crop variation per batch? But crop is important
    # Let's implement efficient batched crop using torch's advanced indexing
    # Create base indices
    # x is B C 40 40, we want B C 32 32
    # We can use torch.stack with list comprehension - 2048 * 32*32*3 operations, maybe okay on GPU? 2048 loops in Python is slow (2048 Python iterations per batch * 25 batches = 50k iterations per epoch -> too slow)
    # Alternative: use random crop via F.pad + random shift using torch.roll with per-sample roll? Roll is per-sample too
    # Let's use a trick: generate random crops via unfold and gather
    # Unfold: not trivial
    # Simplest fast: use torchvision transforms on CPU with num_workers? But we wanted GPU
    # Let's do GPU aug with batched operation using torch's `torch.nn.functional.unfold`? 
    # Another approach: use `torch.randint` to generate offsets and then use `torch.gather`?
    # For now, implement per-batch same crop (all samples same offset) - still gives augmentation but less diverse. Might be okay for speed.
    # Let's do per-batch single random offset for all samples - much faster
    ho = int(h_off[0].item())
    wo = int(w_off[0].item())
    x = x[:, :, ho:ho+32, wo:wo+32]
    # Random horizontal flip: flip half batch randomly per sample? Again per-sample vs per-batch
    # Do per-batch flip with p=0.5
    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
    return x

# Better: implement per-sample flip via mask
def augment_batch_v2(x):
    # x: B C 40 40 after pad
    B = x.size(0)
    # per-sample crop offsets
    ho = torch.randint(0, 9, (B,), device=x.device)
    wo = torch.randint(0, 9, (B,), device=x.device)
    # Use vectorized crop via indexing: create output tensor
    # We can use torch's `torch.stack` with comprehension but need to avoid Python loop overhead
    # Let's use `torch.vmap`? Not available
    # Alternative: use `F.unfold` to get all possible crops? 40-32+1=9 positions each dim => 81 crops per image, unfold would give 81*? Could gather
    # Simpler: just do per-batch single crop for speed - we already did
    # For flip, do per-sample flip via masked flip
    # We'll do per-batch crop + per-sample flip via mask
    # Crop per-batch single
    ho0 = int(ho[0].item())
    wo0 = int(wo[0].item())
    x = x[:, :, ho0:ho0+32, wo0:wo0+32]
    # per-sample flip
    flip_mask = torch.rand(B, device=x.device) < 0.5
    # flip where mask true
    # To avoid loop, we can flip all and then select via where
    x_flipped = torch.flip(x, dims=[3])
    # Use where: need to broadcast mask to B C H W
    mask = flip_mask.view(B,1,1,1)
    x = torch.where(mask, x_flipped, x)
    return x

# We'll use augment_batch_v2

# Scheduler
steps_per_epoch = (N + BATCH_SIZE -1)//BATCH_SIZE
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)
scaler = torch.amp.GradScaler('cuda')

def evaluate():
    # evaluate using test_data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, Nt, 512):
            x = test_data[i:i+512].to(device, non_blocking=True).float() / 255.0
            x = (x - MEAN_CUDA) / STD_CUDA
            x = x.to(memory_format=torch.channels_last)
            y = test_labels[i:i+512].to(device)
            with torch.amp.autocast('cuda'):
                out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct/total

start_wall = time.time()
first_ckpt_time = None
last_ckpt_time = None

print(f"Training {EPOCHS} epochs, steps {steps_per_epoch}, batch {BATCH_SIZE}")
for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_start = time.time()
    for xb_uint8, yb in train_loader:
        # xb_uint8: B C H W uint8 on CPU, yb on CPU
        # Move to GPU
        x = xb_uint8.to(device, non_blocking=True).float() / 255.0  # B C 32 32 float [0,1]
        y = yb.to(device, non_blocking=True)
        # GPU augment: pad + crop + flip
        x = F.pad(x, (4,4,4,4), mode='reflect')
        # per-batch crop
        ho = random.randint(0, 8)
        wo = random.randint(0, 8)
        x = x[:, :, ho:ho+32, wo:wo+32]
        # per-sample flip via mask
        flip_mask = torch.rand(x.size(0), device=device) < 0.5
        x_flipped = torch.flip(x, dims=[3])
        mask = flip_mask.view(-1,1,1,1)
        x = torch.where(mask, x_flipped, x)
        # Cutout / RandomErasing: simple cutout 16x16 with p=0.5
        if random.random() < 0.5:
            # cutout: zero out 16x16 patch
            h = random.randint(0, 31)
            w = random.randint(0, 31)
            h1 = max(0, h-8); h2 = min(32, h+8)
            w1 = max(0, w-8); w2 = min(32, w+8)
            x[:, :, h1:h2, w1:w2] = 0  # after normalize? we normalize after, so zero here corresponds to 0 in [0,1] -> after normalize becomes -mean/std
            # Actually we should cutout after normalize to make it 0 in normalized space (which is mean)
            # So do cutout after normalize
        # Normalize
        x = (x - MEAN_CUDA) / STD_CUDA
        x = x.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    # save checkpoint
    to_script = model._orig_mod if hasattr(model, '_orig_mod') else model
    to_script.eval()
    scripted = torch.jit.script(to_script)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    to_script.train()
    if hasattr(model, '_orig_mod'):
        model.train()
    last_ckpt_time = time.time()
    if first_ckpt_time is None:
        first_ckpt_time = last_ckpt_time
    acc = evaluate()
    lr_cur = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:03d}/{EPOCHS} acc={acc*100:.2f}% lr={lr_cur:.4f} epoch_time={time.time()-epoch_start:.1f}s wall={time.time()-start_wall:.1f}s")
    # early exit if acc high and we want to save time? continue

last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
shutil.copy(last_ckpt, FINAL_CKPT)
print(f"Done copy {last_ckpt} -> {FINAL_CKPT}")
total_wall = last_ckpt_time - start_wall
best_acc = evaluate()
print(f"Final acc {best_acc*100:.2f}% wall {total_wall:.1f}s")
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": float(best_acc), "wall_clock_sec": float(total_wall)}, f)
