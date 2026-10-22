import os, time, shutil, random, glob
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image

DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
CKPT_DIR="/workspace/scratch_ckpt"
FINAL_CKPT="/workspace/scratch_ckpt_final.pt"
EPOCHS=30
BATCH_TRAIN=4096
BATCH_TEST=2048
LR=2.5
WD=5e-4
WARMUP=3

os.makedirs(CKPT_DIR,exist_ok=True)
for f in os.listdir(CKPT_DIR):
    try: os.remove(os.path.join(CKPT_DIR,f))
    except: pass
if os.path.exists(FINAL_CKPT): os.remove(FINAL_CKPT)

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision('high')
device=torch.device("cuda")
print("Loading...")

classes=sorted(os.listdir(DATA_TRAIN))
c2i={c:i for i,c in enumerate(classes)}
train_paths=[]; train_labels=[]
for c in classes:
    for p in glob.glob(os.path.join(DATA_TRAIN,c,"*.png")):
        train_paths.append(p); train_labels.append(c2i[c])
combined=list(zip(train_paths,train_labels)); random.shuffle(combined)
train_paths,train_labels=zip(*combined)
N=len(train_paths)
train_data=torch.empty(N,3,32,32,dtype=torch.uint8)
for i,p in enumerate(train_paths):
    arr=np.array(Image.open(p).convert("RGB"))
    train_data[i]=torch.from_numpy(arr).permute(2,0,1)
    if i%10000==0: print(f"loaded {i}/{N}")
train_labels_t=torch.tensor(train_labels,dtype=torch.long)
print(f"Train {train_data.shape}")

mean=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
std=torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)
test_paths=[]; test_labels=[]
for c in classes:
    for p in glob.glob(os.path.join(DATA_TEST,c,"*.png")):
        test_paths.append(p); test_labels.append(c2i[c])
Nt=len(test_paths)
test_data=torch.empty(Nt,3,32,32,dtype=torch.float32)
for i,p in enumerate(test_paths):
    arr=np.array(Image.open(p).convert("RGB")).astype(np.float32)/255.0
    t=torch.from_numpy(arr).permute(2,0,1)
    t=(t-mean.squeeze().view(3,1,1))/std.squeeze().view(3,1,1)
    test_data[i]=t
test_labels_t=torch.tensor(test_labels,dtype=torch.long)
print(f"Test {test_data.shape}")

class InMemTrain(Dataset):
    def __init__(self,d,l): self.data=d; self.labels=l
    def __len__(self): return len(self.data)
    def __getitem__(self,idx): return self.data[idx], self.labels[idx]

train_ds=InMemTrain(train_data,train_labels_t)
test_ds=TensorDataset(test_data,test_labels_t)
train_loader=DataLoader(train_ds,batch_size=BATCH_TRAIN,shuffle=True,num_workers=0,pin_memory=True)
test_loader=DataLoader(test_ds,batch_size=BATCH_TEST,shuffle=False,num_workers=0)

class BasicBlock(nn.Module):
    def __init__(self,in_planes,planes,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_planes,planes,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.shortcut=nn.Sequential()
        if stride!=1 or in_planes!=planes:
            self.shortcut=nn.Sequential(nn.Conv2d(in_planes,planes,1,stride,bias=False),nn.BatchNorm2d(planes))
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        return F.relu(out)

class WRN(nn.Module):
    def __init__(self,depth=28,widen=4,num_classes=10):
        super().__init__()
        assert (depth-4)%6==0
        n=(depth-4)//6; k=widen
        self.in_planes=16
        nStages=[16,16*k,32*k,64*k]
        self.conv1=nn.Conv2d(3,nStages[0],3,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(nStages[0])
        self.layer1=self._make(BasicBlock,nStages[1],n,1)
        self.layer2=self._make(BasicBlock,nStages[2],n,2)
        self.layer3=self._make(BasicBlock,nStages[3],n,2)
        self.bn_final=nn.BatchNorm2d(nStages[3])
        self.linear=nn.Linear(nStages[3],num_classes)
    def _make(self,block,planes,nb,stride):
        strides=[stride]+[1]*(nb-1)
        layers=[]
        for s in strides:
            layers.append(block(self.in_planes,planes,s))
            self.in_planes=planes
        return nn.Sequential(*layers)
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=F.relu(self.bn_final(out))
        out=F.adaptive_avg_pool2d(out,1)
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        return out

model=WRN(28,4).to(device)
model=model.to(memory_format=torch.channels_last)
# compile
try:
    model=torch.compile(model, mode="max-autotune")
    print("compile ok")
except Exception as e:
    print(f"compile fail {e}")
print(f"Params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
def lr_lambda(epoch):
    if epoch < WARMUP:
        return (epoch+1)/WARMUP
    else:
        p=(epoch-WARMUP)/(EPOCHS-WARMUP)
        return 0.5*(1+torch.cos(torch.tensor(p*3.14159265)).item())
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
scaler=torch.amp.GradScaler('cuda')
criterion=nn.CrossEntropyLoss(label_smoothing=0.1)

mean_d=torch.tensor([0.4914,0.4822,0.4465],device=device).view(1,3,1,1)
std_d=torch.tensor([0.2470,0.2435,0.2616],device=device).view(1,3,1,1)

def augment_batch(x):
    x=F.pad(x,(4,4,4,4),mode='reflect')
    B=x.size(0)
    unfolded=F.unfold(x,kernel_size=32,stride=1)
    idx=torch.randint(0,81,(B,),device=x.device)
    batch_idx=torch.arange(B,device=x.device)
    selected=unfolded[batch_idx,:,idx]
    x=selected.view(B,3,32,32)
    flip_mask=torch.rand(B,device=x.device)<0.5
    if flip_mask.any():
        x[flip_mask]=x[flip_mask].flip(dims=[3])
    # cutout
    if random.random()<0.3:
        for i in range(B):
            if random.random()<0.5:
                top=torch.randint(0,24,(1,)).item()
                left=torch.randint(0,24,(1,)).item()
                x[i,:,top:top+8,left:left+8]=0
    x=(x-mean_d)/std_d
    return x

start_wall=time.time()
first_ckpt=None; last_ckpt=None; best_acc=0
for epoch in range(1,EPOCHS+1):
    model.train()
    t0=time.time()
    for xb,yb in train_loader:
        xb=xb.to(device,non_blocking=True).float()/255.0
        yb=yb.to(device,non_blocking=True)
        xb=augment_batch(xb)
        use_mix=random.random()<0.5
        if use_mix:
            if random.random()<0.5:
                lam=torch.distributions.Beta(0.8,0.8).sample().item()
                idx=torch.randperm(xb.size(0),device=device)
                xb_mixed=lam*xb+(1-lam)*xb[idx]
                y_a=yb; y_b=yb[idx]
                xb=xb_mixed
                y_a_saved,y_b_saved,lam_saved=y_a,y_b,lam
                use_mix_flag=1
            else:
                lam_beta=torch.distributions.Beta(1.0,1.0).sample().item()
                idx=torch.randperm(xb.size(0),device=device)
                W=xb.size(3); H=xb.size(2)
                cut_rat=(1-lam_beta)**0.5
                cut_w=int(W*cut_rat); cut_h=int(H*cut_rat)
                cx=torch.randint(W,(1,),device=device).item()
                cy=torch.randint(H,(1,),device=device).item()
                bbx1=max(cx-cut_w//2,0); bby1=max(cy-cut_h//2,0)
                bbx2=min(cx+cut_w//2,W); bby2=min(cy+cut_h//2,H)
                xb[:,:,bby1:bby2,bbx1:bbx2]=xb[idx,:,bby1:bby2,bbx1:bbx2]
                lam=1-((bbx2-bbx1)*(bby2-bby1)/(W*H))
                y_a_saved,y_b_saved,lam_saved=yb,yb[idx],lam
                use_mix_flag=1
        else:
            use_mix_flag=0
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out=model(xb)
            if use_mix_flag:
                loss=lam_saved*criterion(out,y_a_saved)+(1-lam_saved)*criterion(out,y_b_saved)
            else:
                loss=criterion(out,yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for xb,yb in test_loader:
            xb=xb.to(device,non_blocking=True)
            yb=yb.to(device,non_blocking=True)
            with torch.amp.autocast('cuda'):
                out=model(xb)
            pred=out.argmax(1)
            correct+=(pred==yb).sum().item()
            total+=yb.size(0)
    acc=correct/total
    print(f"Epoch {epoch:02d} acc {acc*100:.2f}% lr {optimizer.param_groups[0]['lr']:.4f} {time.time()-t0:.1f}s")
    # unwrap for scripting
    base = model._orig_mod if hasattr(model,'_orig_mod') else model
    m_copy=WRN(28,4)
    m_copy.load_state_dict(base.state_dict())
    m_copy.eval()
    scripted=torch.jit.script(m_copy)
    ckpt_path=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    if epoch==1: first_ckpt=time.time()
    last_ckpt=time.time()
    best_acc=acc

shutil.copy(os.path.join(CKPT_DIR,f"ckpt_epoch_{EPOCHS:04d}.pt"),FINAL_CKPT)
wall=last_ckpt-start_wall
print(f"Wall {wall:.1f}s span {last_ckpt-first_ckpt:.1f}s best {best_acc:.4f}")
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy":best_acc,"wall_clock_sec":wall},f)
