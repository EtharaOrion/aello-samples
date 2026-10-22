import os, time, shutil, random, json, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
CKPT_DIR="/workspace/scratch_ckpt"
FINAL_CKPT="/workspace/scratch_ckpt_final.pt"
EPOCHS=60
BATCH_SIZE=1024
LR=1.0
WD=5e-4
LS=0.1
MEAN=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
STD=torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR,f))
print("Preloading...")
tmp=ImageFolder(DATA_TRAIN)
N=len(tmp)
train_data=torch.empty(N,3,32,32,dtype=torch.uint8)
train_labels=torch.empty(N,dtype=torch.long)
for i,(p,l) in enumerate(tmp.samples):
    train_data[i]=torch.from_numpy(np.array(Image.open(p).convert('RGB'))).permute(2,0,1)
    train_labels[i]=l
tmp2=ImageFolder(DATA_TEST)
Nt=len(tmp2)
test_data=torch.empty(Nt,3,32,32,dtype=torch.uint8)
test_labels=torch.empty(Nt,dtype=torch.long)
for i,(p,l) in enumerate(tmp2.samples):
    test_data[i]=torch.from_numpy(np.array(Image.open(p).convert('RGB'))).permute(2,0,1)
    test_labels[i]=l
print(f"loaded {N} {Nt}")
class ResBlock(nn.Module):
    def __init__(self,c_in,c_out,s=1):
        super().__init__()
        self.c1=nn.Conv2d(c_in,c_out,3,stride=s,padding=1,bias=False)
        self.b1=nn.BatchNorm2d(c_out)
        self.c2=nn.Conv2d(c_out,c_out,3,padding=1,bias=False)
        self.b2=nn.BatchNorm2d(c_out)
        self.sc=nn.Sequential()
        if s!=1 or c_in!=c_out:
            self.sc=nn.Sequential(nn.Conv2d(c_in,c_out,1,stride=s,bias=False),nn.BatchNorm2d(c_out))
    def forward(self,x):
        o=F.relu(self.b1(self.c1(x)))
        o=self.b2(self.c2(o))
        o+=self.sc(x)
        return F.relu(o)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep=nn.Sequential(nn.Conv2d(3,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU())
        self.l1=nn.Sequential(ResBlock(64,128,2),ResBlock(128,128,1))
        self.l2=nn.Sequential(ResBlock(128,256,2),ResBlock(256,256,1))
        self.l3=nn.Sequential(ResBlock(256,512,2),ResBlock(512,512,1))
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(512,10)
    def forward(self,x):
        x=self.prep(x);x=self.l1(x);x=self.l2(x);x=self.l3(x)
        x=self.pool(x).view(x.size(0),-1)
        return self.fc(x)
device=torch.device("cuda")
model=Net().to(device).to(memory_format=torch.channels_last)
criterion=nn.CrossEntropyLoss(label_smoothing=LS)
opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
steps=(N+BATCH_SIZE-1)//BATCH_SIZE
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,epochs=EPOCHS,steps_per_epoch=steps,pct_start=0.3,anneal_strategy='cos',div_factor=10,final_div_factor=100)
scaler=torch.amp.GradScaler('cuda')
MEAN_C=MEAN.to(device); STD_C=STD.to(device)
train_ds=TensorDataset(train_data,train_labels)
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
def evaluate():
    model.eval()
    correct=0
    with torch.no_grad():
        for i in range(0,Nt,512):
            x=test_data[i:i+512].to(device).float()/255.0
            x=(x-MEAN_C)/STD_C
            x=x.to(memory_format=torch.channels_last)
            y=test_labels[i:i+512].to(device)
            with torch.amp.autocast('cuda'):
                out=model(x)
            correct+=(out.argmax(1)==y).sum().item()
    model.train()
    return correct/Nt
def gpu_augment_fast(x):
    x=F.pad(x,(4,4,4,4),mode='reflect')
    ho=random.randint(0,8)
    wo=random.randint(0,8)
    x=x[:,:,ho:ho+32, wo:wo+32]
    flip=torch.rand(x.size(0),device=x.device)<0.5
    xf=torch.flip(x,dims=[3])
    x=torch.where(flip.view(-1,1,1,1), xf, x)
    B=x.size(0)
    cut=torch.rand(B,device=x.device)<0.5
    if cut.any():
        hc=torch.randint(0,32,(B,),device=x.device)
        wc=torch.randint(0,32,(B,),device=x.device)
        ys=torch.arange(32,device=x.device).view(1,32,1).expand(B,32,32)
        xs=torch.arange(32,device=x.device).view(1,1,32).expand(B,32,32)
        h1=(hc-8).clamp(0,32).view(B,1,1); h2=(hc+8).clamp(0,32).view(B,1,1)
        w1=(wc-8).clamp(0,32).view(B,1,1); w2=(wc+8).clamp(0,32).view(B,1,1)
        inside=(ys>=h1)&(ys<h2)&(xs>=w1)&(xs<w2)
        inside=inside & cut.view(B,1,1)
        x=x.masked_fill(inside.unsqueeze(1),0)
    return x
start=time.time()
first=None; last=None
for epoch in range(1,EPOCHS+1):
    model.train()
    es=time.time()
    for xb,yb in train_loader:
        x=xb.to(device).float()/255.0
        y=yb.to(device)
        x=gpu_augment_fast(x)
        if random.random()<0.5:
            if random.random()<0.5:
                lam=np.random.beta(0.2,0.2)
                perm=torch.randperm(x.size(0),device=device)
                x_mix=lam*x+(1-lam)*x[perm]
                y_perm=y[perm]
                x_mix=(x_mix-MEAN_C)/STD_C
                x_mix=x_mix.to(memory_format=torch.channels_last)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out=model(x_mix)
                    loss=lam*F.cross_entropy(out,y,label_smoothing=LS)+(1-lam)*F.cross_entropy(out,y_perm,label_smoothing=LS)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                continue
            else:
                perm=torch.randperm(x.size(0),device=device)
                lam=np.random.beta(1.0,1.0)
                W=32; H=32
                cut_rat=np.sqrt(1.-lam)
                cut_w=int(W*cut_rat); cut_h=int(H*cut_rat)
                cx=np.random.randint(W); cy=np.random.randint(H)
                bbx1=np.clip(cx-cut_w//2,0,W); bby1=np.clip(cy-cut_h//2,0,H)
                bbx2=np.clip(cx+cut_w//2,0,W); bby2=np.clip(cy+cut_h//2,0,H)
                x[:, :, bby1:bby2, bbx1:bbx2]=x[perm, :, bby1:bby2, bbx1:bbx2]
                lam=1-((bbx2-bbx1)*(bby2-bby1)/(W*H))
                y_perm=y[perm]
                x=(x-MEAN_C)/STD_C
                x=x.to(memory_format=torch.channels_last)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out=model(x)
                    loss=lam*F.cross_entropy(out,y,label_smoothing=LS)+(1-lam)*F.cross_entropy(out,y_perm,label_smoothing=LS)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                continue
        x=(x-MEAN_C)/STD_C
        x=x.to(memory_format=torch.channels_last)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out=model(x)
            loss=criterion(out,y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
    model.eval()
    torch.jit.script(model).save(os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt"))
    model.train()
    last=time.time()
    if first is None: first=last
    acc=evaluate()
    print(f"Epoch {epoch:03d}/{EPOCHS} acc={acc*100:.2f}% lr={opt.param_groups[0]['lr']:.4f} et={time.time()-es:.1f}s wall={time.time()-start:.1f}s")
shutil.copy(os.path.join(CKPT_DIR,f"ckpt_epoch_{EPOCHS:04d}.pt"),FINAL_CKPT)
total=last-start
acc=evaluate()
print(f"Final {acc*100:.2f}% wall {total:.1f}s")
open("/tmp/scratch_metrics10.json","w").write(json.dumps({"accuracy":acc,"wall_clock_sec":total}))
