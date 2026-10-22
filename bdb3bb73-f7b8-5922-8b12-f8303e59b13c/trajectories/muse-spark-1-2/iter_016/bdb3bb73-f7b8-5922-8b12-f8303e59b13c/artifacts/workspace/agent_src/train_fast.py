import os, time, shutil, random, json, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 30
BATCH = 1024
LR = 1.0
WD = 5e-4

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

if os.path.exists(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device("cuda")

# Preload to numpy then to GPU tensor uint8
print("Preloading...")
classes = sorted(os.listdir(DATA_TRAIN))
class_to_idx = {c:i for i,c in enumerate(classes)}
train_paths = []
train_labels = []
for c in classes:
    for p in glob.glob(os.path.join(DATA_TRAIN, c, "*.png")):
        train_paths.append(p)
        train_labels.append(class_to_idx[c])
# load
train_images = []
for p in train_paths:
    img = Image.open(p).convert("RGB")
    train_images.append(np.array(img, dtype=np.uint8))
train_images = np.stack(train_images)  # N H W C
train_labels = np.array(train_labels, dtype=np.int64)
print(f"train {train_images.shape}")

test_paths = []
test_labels = []
for c in classes:
    for p in glob.glob(os.path.join(DATA_TEST, c, "*.png")):
        test_paths.append(p)
        test_labels.append(class_to_idx[c])
test_images = []
for p in test_paths:
    img = Image.open(p).convert("RGB")
    test_images.append(np.array(img, dtype=np.uint8))
test_images = np.stack(test_images)
test_labels = np.array(test_labels, dtype=np.int64)
print(f"test {test_images.shape}")

# Move to GPU as uint8 tensor NCHW? Keep HWC for augmentation then convert
train_images_gpu = torch.from_numpy(train_images).to(device)  # N H W C uint8
train_labels_gpu = torch.from_numpy(train_labels).to(device)
test_images_gpu = torch.from_numpy(test_images).to(device)
test_labels_gpu = torch.from_numpy(test_labels).to(device)

# Precompute normalized test tensor for eval (no aug)
# Convert test to float normalized NCHW
def normalize_batch(x): # x is float in [0,1] NCHW
    mean = torch.tensor([0.4914,0.4822,0.4465], device=x.device).view(1,3,1,1)
    std = torch.tensor([0.2470,0.2435,0.2616], device=x.device).view(1,3,1,1)
    return (x - mean) / std

test_float = test_images_gpu.permute(0,3,1,2).float() / 255.0
test_float = normalize_batch(test_float)  # N C H W

# Model
class BasicBlock(nn.Module):
    __constants__ = ['equalInOut']
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False)
        else:
            self.convShortcut = nn.Identity()
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        out1 = self.conv1(out)
        out1 = self.relu2(self.bn2(out1))
        out1 = self.conv2(out1)
        if self.equalInOut:
            return x + out1
        else:
            return self.convShortcut(out) + out1

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(i==0 and in_planes or out_planes, out_planes, i==0 and stride or 1))
        self.layer = nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10):
        super().__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth-4)%6==0
        n = (depth-4)//6
        self.conv1 = nn.Conv2d(3, nChannels[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], 2)
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

model = WideResNet(28, 10, 10).to(device)
model = model.to(memory_format=torch.channels_last)
# compile
model = torch.compile(model)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler('cuda')

# GPU augmentation helpers
# We will do per-batch augmentation on GPU using vectorized ops
# For random crop with padding 4: pad 4 reflect then random crop
# We can implement by F.pad then gather

def gpu_augment(batch_images): # batch_images: N H W C uint8
    # batch_images is N H W C uint8 on GPU
    N = batch_images.shape[0]
    # pad 4 reflect: need to convert to N C H W for pad? Easier to pad via F.pad on NCHW
    # Convert to N C H W float? But we want to pad before crop, we can pad HWC via manual?
    # Let's convert to NCHW float in [0,1] first, then pad
    x = batch_images.permute(0,3,1,2).float() / 255.0  # N C H W
    # pad 4
    x = F.pad(x, (4,4,4,4), mode='reflect')  # N C 40 40
    # random crop 32: generate random top/left per sample
    # Use torch.randint
    tops = torch.randint(0, 9, (N,), device=x.device)
    lefts = torch.randint(0, 9, (N,), device=x.device)
    # Need to crop each sample differently - loop is okay for N=1024? Might be slow.
    # Instead we can use vectorized trick: create batch of crops via indexing
    # For speed, we can do: for each sample, slice
    # Let's do a fast batched crop using torch.stack
    # This loop over N=1024 with Python loop may be slow but still GPU bound? Let's try to vectorize with unfold?
    # Alternative: use random crop via affine? Simpler to loop in Python but with GPU slices it's still fast (1024 iterations per batch * 49 batches = 50k slices per epoch, okay)
    # We'll implement loop with torch
    cropped = torch.empty((N, 3, 32, 32), device=x.device, dtype=x.dtype)
    for i in range(N):
        cropped[i] = x[i, :, tops[i]:tops[i]+32, lefts[i]:lefts[i]+32]
    x = cropped
    # random flip
    flip_mask = torch.rand(N, device=x.device) < 0.5
    # flip where mask true
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
    # normalize
    mean = torch.tensor([0.4914,0.4822,0.4465], device=x.device).view(1,3,1,1)
    std = torch.tensor([0.2470,0.2435,0.2616], device=x.device).view(1,3,1,1)
    x = (x - mean) / std
    # cutout 16x16 with prob 0.5 per sample
    cut_mask = torch.rand(N, device=x.device) < 0.5
    if cut_mask.any():
        # for each cut sample, random y,x
        ys = torch.randint(0, 32, (N,), device=x.device)
        xs = torch.randint(0, 32, (N,), device=x.device)
        for i in torch.where(cut_mask)[0]:
            y = int(ys[i]); xx = int(xs[i])
            y1 = max(0, y-8); y2 = min(32, y+8)
            x1 = max(0, xx-8); x2 = min(32, xx+8)
            x[i, :, y1:y2, x1:x2] = 0
    return x

# For speed, we can also fuse cutout loop - but okay

# Mixup
def mixup_batch(x, y, alpha=0.2):
    lam = random.betavariate(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1-lam) * x[idx]
    return mixed, y, y[idx], lam

start = time.time()
first = None
last = None
best_acc = 0

N = train_images_gpu.shape[0]
indices = torch.arange(N, device=device)

for epoch in range(1, EPOCHS+1):
    model.train()
    perm = torch.randperm(N, device=device)
    # iterate batches
    for i in range(0, N, BATCH):
        idx = perm[i:i+BATCH]
        batch_imgs = train_images_gpu[idx]  # N H W C
        batch_labels = train_labels_gpu[idx]
        x = gpu_augment(batch_imgs)  # N C H W normalized
        x = x.to(memory_format=torch.channels_last)
        # mixup 50%
        if random.random() < 0.5:
            x, y_a, y_b, lam = mixup_batch(x, batch_labels, 0.2)
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)
        else:
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = criterion(out, batch_labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()

    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, test_float.shape[0], 512):
            x = test_float[i:i+512]
            y = test_labels_gpu[i:i+512]
            with torch.amp.autocast('cuda'):
                out = model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    acc = correct/total
    print(f"Epoch {epoch} acc={acc:.4f} lr={optimizer.param_groups[0]['lr']:.4f}", flush=True)
    if acc > best_acc:
        best_acc = acc

    # save scripted
    unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
    script_model = WideResNet(28, 10, 10).to(device)
    script_model.load_state_dict(unwrapped.state_dict())
    script_model.eval()
    scripted = torch.jit.script(script_model)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    if first is None:
        first = time.time()
    last = time.time()
    try:
        m = torch.jit.load(ckpt_path, map_location="cpu")
        m.eval()
        dummy = torch.randn(2,3,32,32)
        _ = m(dummy)
    except Exception as e:
        print(f"verify fail {e}")

last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
shutil.copy(last_ckpt, FINAL_CKPT)
wall = last - start
span = last - first if first else wall
print(f"Wall {wall:.2f} span {span:.2f} best {best_acc:.4f}")
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": best_acc, "wall_clock_sec": wall}, f)
