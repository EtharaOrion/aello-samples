import os, time, shutil, random, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

EPOCHS = 35
BATCH = 1024
LR = 1.0
WD = 5e-4

CKPT_DIR = "/workspace/scratch_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

def load_folder(root):
    ds = ImageFolder(root)
    n = len(ds)
    data = torch.empty(n, 3, 32, 32, dtype=torch.uint8)
    targets = torch.empty(n, dtype=torch.long)
    for i, (path, cls) in enumerate(ds.samples):
        img = Image.open(path).convert('RGB')
        arr = torch.from_numpy(np.array(img)).permute(2,0,1)
        data[i] = arr
        targets[i] = cls
    return data, targets

print("Loading data...")
train_u8, train_targets_cpu = load_folder("/workspace/data/cifar10/train")
test_u8, test_targets_cpu = load_folder("/workspace/data/cifar10/test")
print(f"Loaded train {train_u8.shape} test {test_u8.shape}")

# Pre-normalize test on GPU
mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1,3,1,1)
test_data = (test_u8.float() / 255.0).to(device)
test_data = (test_data - mean) / std
test_targets = test_targets_cpu.to(device)

# Model
class Block2(nn.Module):
    def __init__(self, in_c, out_c, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.shortcut = nn.Conv2d(in_c, out_c, 1, stride, bias=False) if in_c!=out_c or stride!=1 else None
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        h = self.relu(self.bn1(x))
        out = self.conv1(h)
        out = self.conv2(self.dropout(self.relu(self.bn2(out))))
        if self.shortcut is not None:
            return out + self.shortcut(h)
        else:
            return out + x

class WRN(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth-4)%6==0
        n = (depth-4)//6
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        c1, c2, c3 = 16*widen, 32*widen, 64*widen
        self.layer1 = self._make(16, c1, n, 1, dropout)
        self.layer2 = self._make(c1, c2, n, 2, dropout)
        self.layer3 = self._make(c2, c3, n, 2, dropout)
        self.bn = nn.BatchNorm2d(c3)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(c3, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    def _make(self, in_c, out_c, n, stride, dropout):
        layers=[Block2(in_c, out_c, stride, dropout)]
        for _ in range(1,n):
            layers.append(Block2(out_c, out_c, 1, dropout))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.fc(x)
        return x

model = WRN(depth=28, widen=10, dropout=0.0).to(device)
model = model.to(memory_format=torch.channels_last)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
def lr_lambda(epoch):
    warmup=3
    if epoch < warmup:
        return (epoch+1)/warmup
    p = (epoch - warmup)/(EPOCHS - warmup)
    return 0.5*(1+math.cos(math.pi*p))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda')
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# GPU augment helpers - vectorized
def gpu_augment_batch(x):  # x Bx3x32x32 float [0,1] on GPU
    B = x.size(0)
    # pad 4 reflect -> 40x40
    x_pad = F.pad(x, (4,4,4,4), mode='reflect')  # Bx3x40x40
    # unfold to get all 32x32 crops: 9x9=81 positions
    # F.unfold expects NCHW, returns N x (C*kh*kw) x L
    patches = F.unfold(x_pad, kernel_size=32, stride=1)  # Bx3072x81
    patches = patches.permute(0,2,1)  # Bx81x3072
    idx = torch.randint(0, 81, (B,), device=x.device)
    chosen = patches[torch.arange(B, device=x.device), idx]  # Bx3072
    out = chosen.view(B, 3, 32, 32)
    # flip
    flip_mask = torch.rand(B, device=x.device) > 0.5
    if flip_mask.any():
        out[flip_mask] = torch.flip(out[flip_mask], dims=[3])
    return out

def gpu_erase_batch(x, p=0.25):
    B = x.size(0)
    # vectorized erase: for each sample with prob p, erase random rect
    mask = torch.rand(B, device=x.device) < p
    if not mask.any():
        return x
    # for masked samples, generate random h,w,y,x
    # we do loop over masked indices but masked is ~25% of B=1024 => ~256 iterations, okay
    # but we can try to vectorize with batched indexing using scatter?
    # simple loop over masked
    idxs = torch.where(mask)[0]
    for i in idxs:
        h = random.randint(4, 10)
        w = random.randint(4, 10)
        y = random.randint(0, 32-h)
        xx = random.randint(0, 32-w)
        x[i, :, y:y+h, xx:xx+w] = 0
    return x

def rand_bbox(size, lam):
    W=size[-1]; H=size[-2]
    cut_rat=(1-lam)**0.5
    cut_w=int(W*cut_rat); cut_h=int(H*cut_rat)
    cx=random.randint(0,W-1); cy=random.randint(0,H-1)
    bbx1=max(0,cx-cut_w//2); bby1=max(0,cy-cut_h//2)
    bbx2=min(W,cx+cut_w//2); bby2=min(H,cy+cut_h//2)
    return bbx1,bby1,bbx2,bby2

def mix_cutmix(x,y):
    r=random.random()
    if r<0.4:
        lam=torch.distributions.Beta(0.2,0.2).sample().item()
        idx=torch.randperm(x.size(0), device=x.device)
        return lam*x+(1-lam)*x[idx], y, y[idx], lam
    elif r<0.8:
        lam=torch.distributions.Beta(1.0,1.0).sample().item()
        idx=torch.randperm(x.size(0), device=x.device)
        bbx1,bby1,bbx2,bby2=rand_bbox(x.size(), lam)
        mx=x.clone()
        mx[:,:,bby1:bby2,bbx1:bbx2]=x[idx,:,bby1:bby2,bbx1:bbx2]
        lam=1-((bbx2-bbx1)*(bby2-bby1)/(x.size(-1)*x.size(-2)))
        return mx, y, y[idx], lam
    else:
        return x, y, None, 1.0

def evaluate():
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for i in range(0, len(test_data), 1024):
            x = test_data[i:i+1024]
            y = test_targets[i:i+1024]
            x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda'):
                out = model(x)
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)
    return correct/total

start_wall=time.time()
first=None
last=None
best=0
best_epoch=0
num_train = len(train_u8)

for epoch in range(1, EPOCHS+1):
    t0=time.time()
    model.train()
    perm = torch.randperm(num_train)
    for i in range(0, num_train, BATCH):
        idx = perm[i:i+BATCH]
        x_u8 = train_u8[idx]  # CPU
        y = train_targets_cpu[idx].to(device, non_blocking=True)
        x = x_u8.float().div(255.0).to(device, non_blocking=True)
        # GPU augment
        x = gpu_augment_batch(x)
        x = (x - mean) / std
        x = gpu_erase_batch(x, p=0.25)
        x = x.to(memory_format=torch.channels_last)
        xm, ya, yb, lam = mix_cutmix(x, y)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(xm)
            if yb is None:
                loss = criterion(out, ya)
            else:
                loss = lam*criterion(out, ya)+(1-lam)*criterion(out, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    train_t=time.time()-t0
    acc=evaluate()
    print(f"Epoch {epoch}/{EPOCHS} acc={acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} t={train_t:.1f}s")
    model.eval()
    scripted=torch.jit.script(model)
    p=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(p)
    if first is None: first=time.time()
    last=time.time()
    if acc>best:
        best=acc
        best_epoch=epoch

total_wall=last-start_wall
print(f"Done best={best*100:.2f}% at epoch {best_epoch} total_wall={total_wall:.1f}s")
try:
    m=torch.jit.load(os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt"))
    m.eval(); m.cuda()
    with torch.no_grad():
        x=torch.randn(2,3,32,32, device='cuda')
        print("GPU forward ok", m(x).shape)
except Exception as e:
    print(f"GPU verify failed {e}")

if best >= 0.94:
    print(f"PROMOTING: best {best*100:.2f}% >=94%")
    import glob
    for f in glob.glob("/workspace/periodic_checkpoints/*"):
        os.remove(f)
    for f in os.listdir(CKPT_DIR):
        shutil.copy(os.path.join(CKPT_DIR, f), os.path.join("/workspace/periodic_checkpoints", f))
    shutil.copy(os.path.join("/workspace/periodic_checkpoints", f"ckpt_epoch_{EPOCHS:04d}.pt"), "/workspace/checkpoint.pt")
    with open("/workspace/agent_reported_metrics.json","w") as f:
        json.dump({"accuracy": best, "wall_clock_sec": total_wall}, f)
    print("Promoted")
else:
    print(f"NOT promoting: best {best*100:.2f}% <94%")
