import os, time, shutil, random, glob
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import numpy as np

CKPT_DIR="/workspace/scratch_ckpt"
DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
BATCH=1024
EPOCHS=35
LR=1.0
WD=5e-4
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.allow_tf32=True
torch.backends.cuda.matmul.allow_tf32=True

class BasicBlock(nn.Module):
    def __init__(self,i,o,s):
        super().__init__()
        self.bn1=nn.BatchNorm2d(i); self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(i,o,3,s,1,bias=False)
        self.bn2=nn.BatchNorm2d(o); self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(o,o,3,1,1,bias=False)
        self.eq=(i==o)
        self.short=nn.Identity() if self.eq else nn.Conv2d(i,o,1,s,bias=False)
    def forward(self,x):
        if self.eq:
            out=self.relu1(self.bn1(x))
            out=self.relu2(self.bn2(self.conv1(out)))
            out=self.conv2(out)
            return x+out
        else:
            out=self.relu1(self.bn1(x))
            sh=self.short(out)
            out=self.conv1(out)
            out=self.relu2(self.bn2(out))
            out=self.conv2(out)
            return sh+out
class NetBlock(nn.Module):
    def __init__(self,n,i,o,s):
        super().__init__()
        ls=[BasicBlock(i if k==0 else o, o, s if k==0 else 1) for k in range(n)]
        self.l=nn.Sequential(*ls)
    def forward(self,x): return self.l(x)
class WRN(nn.Module):
    def __init__(self,d=28,w=10,nc=10):
        super().__init__()
        ch=[16,16*w,32*w,64*w]
        n=(d-4)//6
        self.conv1=nn.Conv2d(3,ch[0],3,1,1,bias=False)
        self.b1=NetBlock(n,ch[0],ch[1],1)
        self.b2=NetBlock(n,ch[1],ch[2],2)
        self.b3=NetBlock(n,ch[2],ch[3],2)
        self.bn=nn.BatchNorm2d(ch[3]); self.relu=nn.ReLU(inplace=True)
        self.fc=nn.Linear(ch[3],nc); self.nch=ch[3]
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m,nn.Linear): m.bias.data.zero_()
    def forward(self,x):
        o=self.conv1(x); o=self.b1(o); o=self.b2(o); o=self.b3(o)
        o=self.relu(self.bn(o)); o=F.adaptive_avg_pool2d(o,1); o=o.view(-1,self.nch)
        return self.fc(o)

def load_folder(root):
    classes=sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
    c2i={c:i for i,c in enumerate(classes)}
    imgs=[]; labs=[]
    for c in classes:
        for f in sorted(glob.glob(os.path.join(root,c,"*.png"))):
            imgs.append(np.array(Image.open(f).convert("RGB"),dtype=np.uint8))
            labs.append(c2i[c])
    return torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.array(labs,dtype=np.int64))

print("loading")
train_im_cpu,train_lb_cpu=load_folder(DATA_TRAIN)
test_im_cpu,test_lb_cpu=load_folder(DATA_TEST)
print("loaded",train_im_cpu.shape)
mean=torch.tensor([0.4914,0.4822,0.4465],device=device).view(1,3,1,1)
std=torch.tensor([0.2470,0.2435,0.2616],device=device).view(1,3,1,1)
test_gpu=test_im_cpu.permute(0,3,1,2).float().div(255).to(device)
test_gpu=(test_gpu-mean)/std
test_lb_gpu=test_lb_cpu.to(device)
N=train_im_cpu.shape[0]
# Keep train on CPU as uint8, but use pinned memory for fast transfer
train_im_cpu = train_im_cpu.pin_memory() if hasattr(train_im_cpu, 'pin_memory') else train_im_cpu

model=WRN().to(device).to(memory_format=torch.channels_last)
# no compile for now to avoid overhead
criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-4)
warmup=2
scaler=torch.amp.GradScaler('cuda')

# Pre-allocate augment buffers
def augment_fast(x):
    # x B C 32 32 float [0,1] on GPU
    B=x.shape[0]
    # pad 4
    xp=F.pad(x,(4,4,4,4),mode='reflect')  # B C 40 40
    # random crop via unfold + indexing (fast)
    # Use unfold
    xu=xp.unfold(2,32,1).unfold(3,32,1)  # B C 9 9 32 32
    ho=torch.randint(0,9,(B,),device=x.device)
    wo=torch.randint(0,9,(B,),device=x.device)
    x=xu[torch.arange(B,device=x.device),:,ho,wo]
    # flip
    flip=torch.rand(B,device=x.device)<0.5
    if flip.any():
        x[flip]=torch.flip(x[flip],dims=[3])
    return x

def rand_bbox(W,H,lam):
    cr=(1-lam)**0.5
    cw=int(W*cr); ch=int(H*cr)
    cx=random.randint(0,W-1); cy=random.randint(0,H-1)
    x1=max(0,cx-cw//2); y1=max(0,cy-ch//2)
    x2=min(W,cx+cw//2); y2=min(H,cy+ch//2)
    return x1,y1,x2,y2

def eval_acc():
    model.eval()
    cor=tot=0
    with torch.no_grad():
        for i in range(0,test_gpu.shape[0],512):
            x=test_gpu[i:i+512]; y=test_lb_gpu[i:i+512]
            x=x.to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda'):
                o=model(x)
            cor+=(o.argmax(1)==y).sum().item(); tot+=y.size(0)
    return cor/tot

os.makedirs(CKPT_DIR,exist_ok=True)
for f in os.listdir(CKPT_DIR):
    p=os.path.join(CKPT_DIR,f)
    if os.path.isfile(p): os.remove(p)

start=time.time()
first=None; last=None
# Use CUDA streams for data transfer overlap
for epoch in range(1,EPOCHS+1):
    model.train()
    if epoch<=warmup:
        lr=LR*epoch/warmup
        for pg in opt.param_groups: pg['lr']=lr
    else:
        sched.step()
    perm=torch.randperm(N)
    loss_sum=0; steps=0
    # iterate
    for i in range(0,N,BATCH):
        idx=perm[i:i+BATCH]
        # fast transfer: use pinned memory + non_blocking
        bi_cpu = train_im_cpu[idx]  # B H W C uint8
        bl = train_lb_cpu[idx].to(device, non_blocking=True)
        bi = bi_cpu.permute(0,3,1,2).float().div(255).to(device, non_blocking=True)
        bi = augment_fast(bi)
        bi = (bi - mean) / std
        bi = bi.to(memory_format=torch.channels_last)
        r=random.random()
        if r<0.25:
            lam=random.betavariate(0.2,0.2)
            p2=torch.randperm(bi.size(0),device=device)
            mx=lam*bi+(1-lam)*bi[p2]
            ya, yb=bl, bl[p2]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                o=model(mx)
                loss=lam*criterion(o,ya)+(1-lam)*criterion(o,yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        elif r<0.5:
            lam=random.betavariate(1.0,1.0)
            p2=torch.randperm(bi.size(0),device=device)
            x1,y1,x2,y2=rand_bbox(32,32,lam)
            mx=bi.clone()
            mx[:,:,y1:y2,x1:x2]=bi[p2][:,:,y1:y2,x1:x2]
            lam2=1-((x2-x1)*(y2-y1)/(32*32))
            ya,yb=bl,bl[p2]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                o=model(mx)
                loss=lam2*criterion(o,ya)+(1-lam2)*criterion(o,yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                o=model(bi)
                loss=criterion(o,bl)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        loss_sum+=loss.item(); steps+=1
    # eval every epoch but skip first 10 to save time
    if epoch>=10 or epoch%2==0:
        acc=eval_acc()
    else:
        acc=0
    print(f"Epoch {epoch:03d}/{EPOCHS} loss {loss_sum/steps:.4f} acc {acc*100:.2f}% lr {opt.param_groups[0]['lr']:.4f} time {time.time()-start:.1f}s",flush=True)
    # save checkpoint every epoch
    cpu=WRN()
    cpu.load_state_dict(model.state_dict())
    cpu.eval()
    scripted=torch.jit.script(cpu)
    p=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(p)
    if first is None: first=time.time()
    last=time.time()

ckpts=sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_epoch_")])
if ckpts:
    shutil.copy(os.path.join(CKPT_DIR,ckpts[-1]),"/workspace/scratch_final.pt")
    print("copied")
total=last-start if last else time.time()-start
print(f"total {total:.1f}s")
import json
try:
    fm=torch.jit.load("/workspace/scratch_final.pt").to(device).eval()
    cor=tot=0
    with torch.no_grad():
        for i in range(0,test_gpu.shape[0],512):
            x=test_gpu[i:i+512]; y=test_lb_gpu[i:i+512]
            with torch.amp.autocast('cuda'):
                o=fm(x)
            cor+=(o.argmax(1)==y).sum().item(); tot+=y.size(0)
    acc=cor/tot
    print(f"final {acc*100:.2f}%")
    with open("/workspace/scratch_metrics.json","w") as f: json.dump({"accuracy":float(acc),"wall_clock_sec":float(total)},f)
except Exception as e:
    print(e)
    import traceback; traceback.print_exc()
