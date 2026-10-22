import os, time, shutil, random, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import torchvision.transforms as transforms

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/scratch_ckpt"
FINAL_CKPT = "/workspace/scratch_ckpt_final.pt"
EPOCHS = 30
BATCH_TRAIN = 2048
BATCH_TEST = 2048
LR = 1.5
WD = 5e-4
WARMUP = 2

os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR,f))
if os.path.exists(FINAL_CKPT):
    os.remove(FINAL_CKPT)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
print("Loading data into RAM...")

# Preload train
import pathlib
classes = sorted(os.listdir(DATA_TRAIN))
class_to_idx = {c:i for i,c in enumerate(classes)}
train_paths = []
train_labels = []
for c in classes:
    for p in glob.glob(os.path.join(DATA_TRAIN, c, "*.png")):
        train_paths.append(p)
        train_labels.append(class_to_idx[c])
# shuffle
combined = list(zip(train_paths, train_labels))
random.shuffle(combined)
train_paths, train_labels = zip(*combined)

# Load all images as uint8 tensors (N,3,32,32)
# Use PIL to load, convert to tensor uint8
import numpy as np
N = len(train_paths)
train_data = torch.empty(N, 3, 32, 32, dtype=torch.uint8)
for i, p in enumerate(train_paths):
    img = Image.open(p).convert("RGB")
    arr = np.array(img) # HWC
    t = torch.from_numpy(arr).permute(2,0,1) # 3,32,32
    train_data[i] = t
    if i % 10000 == 0:
        print(f"loaded {i}/{N}")
train_labels_t = torch.tensor(train_labels, dtype=torch.long)
print(f"Train data {train_data.shape} {train_data.dtype}")

# Test data preload normalized
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)
# For test, preload normalized float
test_paths = []
test_labels = []
for c in classes:
    for p in glob.glob(os.path.join(DATA_TEST, c, "*.png")):
        test_paths.append(p)
        test_labels.append(class_to_idx[c])
Nt = len(test_paths)
test_data = torch.empty(Nt, 3, 32, 32, dtype=torch.float32)
for i, p in enumerate(test_paths):
    img = Image.open(p).convert("RGB")
    arr = np.array(img).astype(np.float32)/255.0
    t = torch.from_numpy(arr).permute(2,0,1)
    # normalize
    t = (t - mean.squeeze().view(3,1,1)) / std.squeeze().view(3,1,1)
    test_data[i] = t
test_labels_t = torch.tensor(test_labels, dtype=torch.long)
print(f"Test data {test_data.shape}")

# Datasets that return uint8 for train (aug on GPU)
class InMemTrain(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_ds = InMemTrain(train_data, train_labels_t)
test_ds = TensorDataset(test_data, test_labels_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False, num_workers=0, pin_memory=False)

# Model WRN 28-4
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class WRN(nn.Module):
    def __init__(self, depth=28, widen=4, num_classes=10):
        super().__init__()
        assert (depth-4)%6==0
        n=(depth-4)//6
        k=widen
        self.in_planes=16
        nStages=[16,16*k,32*k,64*k]
        self.conv1=nn.Conv2d(3,nStages[0],3,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(nStages[0])
        self.layer1=self._make(BasicBlock,nStages[1],n,1)
        self.layer2=self._make(BasicBlock,nStages[2],n,2)
        self.layer3=self._make(BasicBlock,nStages[3],n,2)
        self.bn_final=nn.BatchNorm2d(nStages[3])
        self.linear=nn.Linear(nStages[3],num_classes)
    def _make(self, block, planes, nb, stride):
        strides=[stride]+[1]*(nb-1)
        layers=[]
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes=planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=F.relu(self.bn_final(out))
        out=F.adaptive_avg_pool2d(out,1)
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        return out

model = WRN(28,4).to(device)
model = model.to(memory_format=torch.channels_last)
print(f"Params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
def lr_lambda(epoch):
    if epoch < WARMUP:
        return (epoch+1)/WARMUP
    else:
        p=(epoch-WARMUP)/(EPOCHS-WARMUP)
        return 0.5*(1+torch.cos(torch.tensor(p*3.14159265)).item())
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda')
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# GPU augmentation helpers
def augment_batch(x): # x is float [0,1] normalized? we will handle
    # x: (B,3,32,32) float in [0,1] before normalize, or normalized?
    # We'll do: input is uint8 -> float/255 -> pad+crop+flip -> normalize
    # So do pad 4, random crop, flip here
    B = x.size(0)
    # pad 4
    x = F.pad(x, (4,4,4,4), mode='reflect') # B,3,40,40
    # random crop 32
    # generate random offsets
    # Use torch.randint
    # For speed, use vectorized crop via grid? Simple loop with random ints per sample using slicing is slow.
    # Instead use torch's random crop via unfold? Easier: generate random top/left per sample and use advanced indexing
    # We'll do batched crop using torch
    # Create random offsets
    tops = torch.randint(0, 9, (B,), device=x.device)
    lefts = torch.randint(0, 9, (B,), device=x.device)
    # Use trick: for each sample, slice
    # This loop is okay for B=2048, 2048 iterations per batch is heavy but still maybe okay on GPU? Let's try vectorized via torch.stack
    # Alternative: use F.unfold? Simpler to do loop in python but with GPU tensors slicing is still GPU op
    # We'll do loop but with torch ops - should be okay
    # To avoid python loop overhead, we can use torch's crop via affine? Let's just loop with small B
    # Actually we can use torchvision's RandomCrop on GPU via Kornia? Let's implement fast batched crop using gather
    # Simpler: use torch.randint and then use torch.stack with list comprehension (still python loop)
    cropped = torch.empty(B, 3, 32, 32, device=x.device, dtype=x.dtype)
    for i in range(B):
        cropped[i] = x[i, :, tops[i]:tops[i]+32, lefts[i]:lefts[i]+32]
    x = cropped
    # flip
    flip_mask = torch.rand(B, device=x.device) < 0.5
    # flip where mask true
    if flip_mask.any():
        x[flip_mask] = x[flip_mask].flip(dims=[3])
    # normalize
    mean_d = torch.tensor([0.4914,0.4822,0.4465], device=x.device).view(1,3,1,1)
    std_d = torch.tensor([0.2470,0.2435,0.2616], device=x.device).view(1,3,1,1)
    x = (x - mean_d) / std_d
    return x

# For speed, pre-create mean/std
mean_d = torch.tensor([0.4914,0.4822,0.4465], device=device).view(1,3,1,1)
std_d = torch.tensor([0.2470,0.2435,0.2616], device=device).view(1,3,1,1)

def mixup_data(x, y, alpha=0.8):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1-lam) * x[idx]
    return mixed, y, y[idx], lam

def cutmix_data(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    W=x.size(3); H=x.size(2)
    cut_rat=(1-lam)**0.5
    cut_w=int(W*cut_rat); cut_h=int(H*cut_rat)
    cx=torch.randint(W,(1,),device=x.device).item()
    cy=torch.randint(H,(1,),device=x.device).item()
    bbx1=max(cx-cut_w//2,0); bby1=max(cy-cut_h//2,0)
    bbx2=min(cx+cut_w//2,W); bby2=min(cy+cut_h//2,H)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
    lam=1-((bbx2-bbx1)*(bby2-bby1)/(W*H))
    return x, y, y[idx], lam

def mix_criterion(pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam)*criterion(pred, y_b)

start_wall=time.time()
first_ckpt=None
last_ckpt=None
best_acc=0

for epoch in range(1, EPOCHS+1):
    model.train()
    t0=time.time()
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True).float() / 255.0  # B,3,32,32 [0,1]
        yb = yb.to(device, non_blocking=True)
        # augment on GPU
        # pad+crop+flip+normalize
        # Use fast augment
        # pad
        xb = F.pad(xb, (4,4,4,4), mode='reflect')
        # random crop via random offsets - vectorized using torch
        B = xb.size(0)
        tops = torch.randint(0, 9, (B,), device=device)
        lefts = torch.randint(0, 9, (B,), device=device)
        # Use batched indexing via loop - but we can try to use torch's unfold for speed
        # Let's use unfold trick: xb is B,3,40,40 -> we want random 32x32 crop
        # Could use F.unfold but simpler to keep loop; for B=2048 loop 2048 times per batch is heavy (25 batches per epoch => 50k iterations)
        # Let's try to implement crop via grid_sample or via slicing with advanced indexing using torch
        # Alternative: use torch's random crop by generating a single crop per batch? No need per-sample randomness? But we need per-sample.
        # We'll implement using torch's efficient batched crop via `torch.stack` with list comp - still loop but GPU slicing is fast
        # Let's try to optimize: use `torch.nn.functional.unfold` to get all possible crops? Not.
        # Keep loop but maybe okay: 2048*25=51200 slices per epoch, each slice is GPU op, might be okay (~0.5s)
        cropped = torch.empty(B, 3, 32, 32, device=device, dtype=xb.dtype)
        for i in range(B):
            cropped[i] = xb[i, :, tops[i]:tops[i]+32, lefts[i]:lefts[i]+32]
        xb = cropped
        # flip
        flip_mask = torch.rand(B, device=device) < 0.5
        if flip_mask.any():
            xb[flip_mask] = xb[flip_mask].flip(dims=[3])
        xb = (xb - mean_d) / std_d

        # mixup/cutmix 50%
        use_mix = random.random() < 0.5
        if use_mix:
            if random.random() < 0.5:
                xb, y_a, y_b, lam = mixup_data(xb, yb, 0.8)
            else:
                xb, y_a, y_b, lam = cutmix_data(xb, yb, 1.0)
            use_mix_flag=True
        else:
            use_mix_flag=False

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(xb)
            if use_mix_flag:
                loss = lam * criterion(out, y_a) + (1-lam)*criterion(out, y_b)
            else:
                loss = criterion(out, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    # eval
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(xb)
            pred = out.argmax(1)
            correct += (pred==yb).sum().item()
            total += yb.size(0)
    acc=correct/total
    print(f"Epoch {epoch:02d} acc {acc*100:.2f}% lr {optimizer.param_groups[0]['lr']:.4f} {time.time()-t0:.1f}s")
    # save
    m_copy = WRN(28,4)
    m_copy.load_state_dict(model.state_dict())
    m_copy.eval()
    scripted=torch.jit.script(m_copy)
    ckpt_path=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    if epoch==1:
        first_ckpt=time.time()
    last_ckpt=time.time()
    best_acc=acc

# copy final
shutil.copy(os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt"), FINAL_CKPT)
wall=last_ckpt-start_wall
print(f"Wall {wall:.1f}s span {last_ckpt-first_ckpt:.1f}s best {best_acc:.4f}")
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": best_acc, "wall_clock_sec": wall}, f)
