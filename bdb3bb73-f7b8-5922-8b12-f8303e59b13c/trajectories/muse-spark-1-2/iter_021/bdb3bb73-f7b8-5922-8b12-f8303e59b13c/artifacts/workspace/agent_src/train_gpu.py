import os, time, random, shutil
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import torchvision

EPOCHS=30
BATCH=1024
LR=1.5
WD=5e-4
MEAN=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1).cuda()
STD=torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1).cuda()
CKPT="/workspace/scratch_ckpt"
os.makedirs(CKPT,exist_ok=True)
for f in os.listdir(CKPT): os.remove(os.path.join(CKPT,f))
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

class Block(nn.Module):
    def __init__(self,i,o,s):
        super().__init__()
        self.bn1=nn.BatchNorm2d(i); self.r1=nn.ReLU(inplace=True)
        self.c1=nn.Conv2d(i,o,3,s,1,bias=False)
        self.bn2=nn.BatchNorm2d(o); self.r2=nn.ReLU(inplace=True)
        self.c2=nn.Conv2d(o,o,3,1,1,bias=False)
        self.sc=nn.Conv2d(i,o,1,s,bias=False) if i!=o or s!=1 else nn.Identity()
    def forward(self,x):
        out=self.r1(self.bn1(x)); sc=self.sc(x); out=self.c1(out); out=self.c2(self.r2(self.bn2(out))); return out+sc
class WRN(nn.Module):
    def __init__(self,d=28,w=10,nc=10):
        super().__init__()
        n=(d-4)//6
        self.ic=nn.Conv2d(3,16,3,1,1,bias=False)
        c1,c2,c3=16*w,32*w,64*w
        def mk(i,o,n,s): return nn.Sequential(*([Block(i,o,s)]+[Block(o,o,1) for _ in range(n-1)]))
        self.l1=mk(16,c1,n,1); self.l2=mk(c1,c2,n,2); self.l3=mk(c2,c3,n,2)
        self.bn=nn.BatchNorm2d(c3); self.r=nn.ReLU(inplace=True); self.fc=nn.Linear(c3,nc)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.zeros_(m.bias)
    def forward(self,x):
        o=self.ic(x); o=self.l1(o); o=self.l2(o); o=self.l3(o); o=self.r(self.bn(o)); o=F.adaptive_avg_pool2d(o,1).flatten(1); return self.fc(o)

# Preload data to GPU-friendly format: load as uint8 tensors on CPU, then augment on GPU
# Use torchvision CIFAR10 dataset to get PIL, convert to tensor quickly
# Instead, load ImageFolder and cache as tensors
from torchvision.datasets import ImageFolder
import pathlib

# Load train data as tensors (uint8) on CPU RAM
print("Loading data...")
train_ds_raw = ImageFolder("/workspace/data/cifar10/train", transform=transforms.ToTensor())  # [0,1]
# Stack all
train_images = torch.stack([img for img,_ in train_ds_raw])  # N,3,32,32 float [0,1]
train_labels = torch.tensor([lbl for _,lbl in train_ds_raw])
print(f"train {train_images.shape} {train_images.dtype}")
# Keep on CPU, will transfer per batch and augment on GPU

test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
test_ds = ImageFolder("/workspace/data/cifar10/test", transform=test_tf)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

# Create DataLoader that yields raw [0,1] tensors
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

device=torch.device("cuda")
model=WRN().to(device).to(memory_format=torch.channels_last)
try:
    model=torch.compile(model, mode="max-autotune")
    print("compile ok")
except Exception as e: print(e)

crit=nn.CrossEntropyLoss(label_smoothing=0.1)
opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,epochs=EPOCHS,steps_per_epoch=len(train_loader),pct_start=0.3,anneal_strategy='cos',div_factor=10,final_div_factor=100)
scaler=torch.amp.GradScaler('cuda')

# GPU augmentation functions
def gpu_augment(x):  # x: [B,3,32,32] in [0,1] on GPU
    # RandomCrop 32 with padding 4: pad 4 then random crop
    # pad
    x = F.pad(x, (4,4,4,4), mode='reflect')  # [B,3,40,40]
    # random crop
    B = x.size(0)
    # generate random offsets
    off_h = torch.randint(0, 9, (B,), device=x.device)
    off_w = torch.randint(0, 9, (B,), device=x.device)
    # Use batched crop via grid? Simple loop is slow but B=1024, 32x32 small, we can use unfold trick
    # Use torch.stack with slicing - vectorized via advanced indexing
    # Create output
    out = torch.empty(B, 3, 32, 32, device=x.device, dtype=x.dtype)
    for i in range(B):
        out[i] = x[i, :, off_h[i]:off_h[i]+32, off_w[i]:off_w[i]+32]
    # Random flip
    flip_mask = torch.rand(B, device=x.device) < 0.5
    out[flip_mask] = torch.flip(out[flip_mask], dims=[3])
    # Normalize
    out = (out - MEAN) / STD
    return out

# For speed, use torch's random crop via F.pad + batched indexing using torch
# Alternative faster: use Kornia if available, but we do manual

def mixup(x,y,a=0.4):
    lam=random.betavariate(a,a); idx=torch.randperm(x.size(0),device=x.device); return lam*x+(1-lam)*x[idx], y, y[idx], lam

start=time.time()
best=0
for epoch in range(1,EPOCHS+1):
    model.train(); t0=time.time()
    for xb,yb in train_loader:
        xb=xb.to(device, non_blocking=True)  # [B,3,32,32] [0,1]
        yb=yb.to(device, non_blocking=True)
        # GPU augment
        with torch.no_grad():
            xb = gpu_augment(xb)
            # RandomErasing 25% - simple
            if random.random()<0.25:
                # erase random patch
                B = xb.size(0)
                for i in range(B):
                    if random.random()<0.5:
                        h = random.randint(4,10); w = random.randint(4,10)
                        y1 = random.randint(0,32-h); x1 = random.randint(0,32-w)
                        xb[i,:,y1:y1+h,x1:x1+w]=0  # after normalize, 0 is mean
        do=random.random()<0.5
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            if do:
                xm,ya,yb2,lam=mixup(xb,yb,0.4)
                out=model(xm); loss=lam*crit(out,ya)+(1-lam)*crit(out,yb2)
            else:
                out=model(xb); loss=crit(out,yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
            with torch.amp.autocast('cuda',dtype=torch.bfloat16): out=model(x)
            correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    acc=correct/total; best=max(best,acc)
    print(f"E{epoch:02d} acc={acc*100:.2f} best={best*100:.2f} dt={time.time()-t0:.1f}s")
    raw = model._orig_mod if hasattr(model,'_orig_mod') else model
    cpu_m=WRN().to("cpu"); cpu_m.load_state_dict({k.replace("_orig_mod.",""):v for k,v in raw.state_dict().items()}); cpu_m.eval()
    torch.jit.script(cpu_m).save(os.path.join(CKPT,f"ckpt_epoch_{epoch:04d}.pt"))
wall=time.time()-start
print(f"wall {wall:.1f} best {best*100:.2f}")
