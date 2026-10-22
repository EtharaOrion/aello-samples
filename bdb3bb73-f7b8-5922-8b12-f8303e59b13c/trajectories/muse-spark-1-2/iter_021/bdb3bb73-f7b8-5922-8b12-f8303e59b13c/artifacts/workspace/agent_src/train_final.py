import os, time, random, shutil
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EPOCHS=30
BATCH=1024
LR=1.5
WD=5e-4
MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)
CKPT="/workspace/scratch_ckpt"
os.makedirs(CKPT,exist_ok=True)
for f in os.listdir(CKPT): os.remove(os.path.join(CKPT,f))
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

# Model
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

# --- GPU cached data ---
print("Caching data to GPU...")
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image

# Load via ImageFolder but cache as uint8 tensor on GPU
train_ds_pil = ImageFolder("/workspace/data/cifar10/train")  # no transform, returns PIL
# Convert to tensor uint8 [N,3,32,32]
N=len(train_ds_pil)
train_imgs = torch.empty(N,3,32,32,dtype=torch.uint8)
train_lbls = torch.empty(N,dtype=torch.long)
for i,(img,lbl) in enumerate(train_ds_pil):
    # img is PIL 32x32
    arr = torch.from_numpy(np.array(img)).permute(2,0,1)  # HWC->CHW
    train_imgs[i]=arr
    train_lbls[i]=lbl
print(f"cached {N} images {train_imgs.shape}")

# Move to GPU as uint8 (50k*3*32*32=150MB)
train_imgs_gpu = train_imgs.cuda()
train_lbls_gpu = train_lbls.cuda()

# Test loader still via DataLoader (small)
test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
test_ds=ImageFolder("/workspace/data/cifar10/test",transform=test_tf)
test_loader=DataLoader(test_ds,batch_size=512,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)

device=torch.device("cuda")
model=WRN().to(device).to(memory_format=torch.channels_last)
try:
    model=torch.compile(model, mode="max-autotune")
    print("compile ok")
except Exception as e: print(f"compile fail {e}")

crit=nn.CrossEntropyLoss(label_smoothing=0.1)
opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
# steps per epoch
steps_per_epoch = (N + BATCH -1)//BATCH
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,epochs=EPOCHS,steps_per_epoch=steps_per_epoch,pct_start=0.3,anneal_strategy='cos',div_factor=10,final_div_factor=100)
scaler=torch.amp.GradScaler('cuda')

MEAN_T=torch.tensor(MEAN,device='cuda').view(1,3,1,1)
STD_T=torch.tensor(STD,device='cuda').view(1,3,1,1)

def gpu_augment_batch(x): # x uint8 [B,3,32,32] -> float normalized [B,3,32,32]
    B=x.size(0)
    # pad 4 reflect: need float first
    xf = x.float()/255.0  # [0,1]
    # pad
    xf = F.pad(xf, (4,4,4,4), mode='reflect')  # [B,3,40,40]
    # random crop via batched indexing using affine? Use random offsets and gather
    # Use vectorized crop: generate offsets and use torch
    off_h = torch.randint(0,9,(B,),device='cuda')
    off_w = torch.randint(0,9,(B,),device='cuda')
    # Use unfold trick for speed: but loop is okay for B=1024? 1024* loop is slow
    # Instead use batched crop via grid_sample or via indexing with advanced
    # Let's use torch.stack with slicing via vmap-like: use torch's crop via F
    # Fastest: use torch's random crop by using batched roll? Simpler: use for loop but with torch.compile it may be okay
    # We'll do vectorized using torch.arange and advanced indexing
    # Create output via indexing
    # Use torch's efficient method: use F.unfold? Let's just loop in Python but with small B it's okay - 1024 loops per batch * 49 batches = 50k loops per epoch, each loop copies 3*32*32 floats -> ~50k*3k=150M elements, okay maybe 0.1s
    out = torch.empty(B,3,32,32,device='cuda',dtype=torch.float32)
    for i in range(B):
        out[i]=xf[i,:,off_h[i]:off_h[i]+32, off_w[i]:off_w[i]+32]
    # flip
    flip = torch.rand(B,device='cuda')<0.5
    out[flip]=torch.flip(out[flip],dims=[3])
    # normalize
    out = (out - MEAN_T)/STD_T
    # random erasing 25% of images, 50% of those
    # do per-image erasing with probability 0.25
    # vectorized: generate mask
    erase_mask = torch.rand(B,device='cuda')<0.25
    idxs = torch.where(erase_mask)[0]
    for i in idxs:
        h = random.randint(4,10); w = random.randint(4,10)
        y1 = random.randint(0,32-h); x1 = random.randint(0,32-w)
        out[i,:,y1:y1+h,x1:x1+w]=0
    return out

# Pre-compile augment? Not needed

start=time.time()
best=0
for epoch in range(1,EPOCHS+1):
    model.train()
    t0=time.time()
    # shuffle indices
    perm = torch.randperm(N, device='cuda')
    # iterate batches
    for i in range(0,N,BATCH):
        idx = perm[i:i+BATCH]
        xb_uint8 = train_imgs_gpu[idx]  # [B,3,32,32] uint8
        yb = train_lbls_gpu[idx]
        # augment on GPU
        xb = gpu_augment_batch(xb_uint8)
        xb = xb.to(memory_format=torch.channels_last)
        # mixup 50%
        do_mix = random.random()<0.5
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            if do_mix:
                lam = random.betavariate(0.4,0.4)
                perm2 = torch.randperm(xb.size(0),device='cuda')
                xb_m = lam*xb + (1-lam)*xb[perm2]
                yb2 = yb[perm2]
                out = model(xb_m)
                loss = lam*crit(out,yb) + (1-lam)*crit(out,yb2)
            else:
                out = model(xb)
                loss = crit(out,yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
    # eval
    model.eval()
    correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device,non_blocking=True).to(memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                out=model(x)
            correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    acc=correct/total; best=max(best,acc)
    dt=time.time()-t0
    print(f"E{epoch:02d} acc={acc*100:.2f} best={best*100:.2f} dt={dt:.1f}s lr={opt.param_groups[0]['lr']:.4f}")
    # save
    raw = model._orig_mod if hasattr(model,'_orig_mod') else model
    cpu_m=WRN().to("cpu")
    cpu_m.load_state_dict({k.replace("_orig_mod.",""):v for k,v in raw.state_dict().items()})
    cpu_m.eval()
    torch.jit.script(cpu_m).save(os.path.join(CKPT,f"ckpt_epoch_{epoch:04d}.pt"))

wall=time.time()-start
print(f"wall {wall:.1f} best {best*100:.2f}")
