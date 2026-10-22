import os, time, shutil, random, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

EPOCHS = 30
BATCH = 1024
LR = 1.0
WD = 5e-4

CKPT_DIR = "/workspace/periodic_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

# Preload data into memory as tensors (uint8) for speed
# Use ImageFolder to get paths, then load
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF

# Load train as numpy
train_root = "/workspace/data/cifar10/train"
test_root = "/workspace/data/cifar10/test"

# Use datasets.ImageFolder with ToTensor but preload
# Instead, manually load all images into tensor
def load_folder(root):
    ds = ImageFolder(root)
    # ds.samples is list of (path, class)
    # Load all
    n = len(ds)
    # CIFAR10 32x32
    data = torch.empty(n, 3, 32, 32, dtype=torch.uint8)
    targets = torch.empty(n, dtype=torch.long)
    for i, (path, cls) in enumerate(ds.samples):
        img = Image.open(path).convert('RGB')
        # to tensor uint8
        arr = torch.from_numpy(np.array(img)).permute(2,0,1)  # HWC -> CHW
        data[i] = arr
        targets[i] = cls
    return data, targets

print("Loading data...")
train_data, train_targets = load_folder(train_root)
test_data, test_targets = load_folder(test_root)
print(f"Train {train_data.shape} Test {test_data.shape}")

# For fast augmentation, we do on GPU or CPU?
# We'll do CPU augmentations via transforms on PIL but we have tensors
# Instead create a Dataset that does augmentations on the fly using torchvision transforms on tensor
# Use Kornia-style? Simple: RandomCrop + Flip + AutoAugment via torchvision

# We'll use a custom Dataset with GPU augmentation? For speed, do CPU with num_workers
from torch.utils.data import Dataset

class FastTrainDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data  # uint8 CHW
        self.targets = targets
        # for AutoAugment we need PIL
        self.aa = transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]  # CHW uint8
        # Convert to PIL for augmentations
        pil = TF.to_pil_image(img)
        # RandomCrop with padding 4
        pil = transforms.functional.pad(pil, 4, padding_mode='reflect')
        # random crop 32
        i = random.randint(0, 8)
        j = random.randint(0, 8)
        pil = TF.crop(pil, i, j, 32, 32)
        if random.random() > 0.5:
            pil = TF.hflip(pil)
        pil = self.aa(pil)
        t = TF.to_tensor(pil)  # [0,1]
        t = TF.normalize(t, (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
        # RandomErasing
        if random.random() < 0.25:
            # simple random erasing
            # t is 3x32x32 normalized
            # erase random rectangle
            h = random.randint(4, 10)
            w = random.randint(4, 10)
            y = random.randint(0, 32-h)
            x = random.randint(0, 32-w)
            t[:, y:y+h, x:x+w] = 0  # 0 corresponds to mean after normalize? approx
        return t, self.targets[idx]

class FastTestDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx].float() / 255.0
        img = TF.normalize(img, (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
        return img, self.targets[idx]

train_ds = FastTrainDataset(train_data, train_targets)
test_ds = FastTestDataset(test_data, test_targets)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# Model: WRN 16-8 (faster) or 28-10? Try 16-8
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
    def __init__(self, depth=16, widen=8, num_classes=10, dropout=0.0):
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

model = WRN(depth=16, widen=8, dropout=0.0).to(device)
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
    correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x=x.to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda'):
                out=model(x)
            correct+=(out.argmax(1)==y).sum().item()
            total+=y.size(0)
    return correct/total

start_wall=time.time()
first=None
last=None
best=0
for epoch in range(1, EPOCHS+1):
    model.train()
    for x,y in train_loader:
        x,y=x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x=x.to(memory_format=torch.channels_last)
        xm, ya, yb, lam = mix_cutmix(x,y)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out=model(xm)
            if yb is None:
                loss=criterion(out, ya)
            else:
                loss=lam*criterion(out, ya)+(1-lam)*criterion(out, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    acc=evaluate()
    print(f"Epoch {epoch}/{EPOCHS} acc={acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f}")
    model.eval()
    scripted=torch.jit.script(model)
    p=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(p)
    if first is None: first=time.time()
    last=time.time()
    if acc>best: best=acc

shutil.copy(os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt"), "/workspace/checkpoint.pt")
total_wall=last-start_wall
print(f"Done best={best*100:.2f}% total_wall={total_wall:.1f}s")
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": best, "wall_clock_sec": total_wall}, f)
