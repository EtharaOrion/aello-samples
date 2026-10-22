import os, time, shutil, random, json, glob
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
CKPT_DIR="/workspace/periodic_checkpoints"
FINAL_CKPT="/workspace/checkpoint.pt"
EPOCHS=70
BATCH=1024
LR=1.2
WD=5e-4
MOM=0.9
MEAN=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1).cuda()
STD=torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1).cuda()
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.allow_tf32=True
torch.backends.cuda.matmul.allow_tf32=True

def load_folder(root):
    import os, glob
    classes=sorted(os.listdir(root))
    cls_to_idx={c:i for i,c in enumerate(classes)}
    imgs=[]; labels=[]
    for c in classes:
        for p in glob.glob(os.path.join(root,c,"*.png")):
            im=np.array(Image.open(p))
            imgs.append(im)
            labels.append(cls_to_idx[c])
    imgs=np.stack(imgs)
    imgs=torch.from_numpy(imgs).permute(0,3,1,2).contiguous()
    labels=torch.tensor(labels, dtype=torch.long)
    return imgs, labels

print("Loading data...")
train_imgs, train_labels = load_folder(DATA_TRAIN)
test_imgs, test_labels = load_folder(DATA_TEST)
print(f"train {train_imgs.shape} test {test_imgs.shape}")

class Mul(nn.Module):
    def __init__(self,w): super().__init__(); self.w=w
    def forward(self,x): return x*self.w
class Residual(nn.Module):
    def __init__(self,m): super().__init__(); self.m=m
    def forward(self,x): return x+self.m(x)
def conv_bn(c_in,c_out,k=3,s=1,p=1):
    return nn.Sequential(nn.Conv2d(c_in,c_out,k,s,p,bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0=conv_bn(3,64,3,1,1)
        self.conv1=conv_bn(64,128,3,1,1)
        self.pool1=nn.MaxPool2d(2)
        self.res1a=Residual(nn.Sequential(conv_bn(128,128,3,1,1),conv_bn(128,128,3,1,1)))
        self.res1b=Residual(nn.Sequential(conv_bn(128,128,3,1,1),conv_bn(128,128,3,1,1)))
        self.conv2=conv_bn(128,256,3,1,1)
        self.pool2=nn.MaxPool2d(2)
        self.conv3=conv_bn(256,512,3,1,1)
        self.pool3=nn.MaxPool2d(2)
        self.res2a=Residual(nn.Sequential(conv_bn(512,512,3,1,1),conv_bn(512,512,3,1,1)))
        self.res2b=Residual(nn.Sequential(conv_bn(512,512,3,1,1),conv_bn(512,512,3,1,1)))
        self.pool4=nn.MaxPool2d(2)
        self.fc=nn.Linear(512*2*2,10,bias=False)
        self.mul=Mul(0.2)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
    def forward(self,x):
        x=self.conv0(x);x=self.conv1(x);x=self.pool1(x)
        x=self.res1a(x);x=self.res1b(x)
        x=self.conv2(x);x=self.pool2(x)
        x=self.conv3(x);x=self.pool3(x)
        x=self.res2a(x);x=self.res2b(x)
        x=self.pool4(x)
        x=x.reshape(x.size(0),-1)
        x=self.fc(x);x=self.mul(x)
        return x

model=Net().to(device).to(memory_format=torch.channels_last)
criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer=torch.optim.SGD(model.parameters(),lr=LR,momentum=MOM,weight_decay=WD,nesterov=True)
steps_per_epoch=(len(train_imgs)+BATCH-1)//BATCH
scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=LR,epochs=EPOCHS,steps_per_epoch=steps_per_epoch,pct_start=0.3,anneal_strategy='cos',div_factor=10,final_div_factor=100)
scaler=torch.amp.GradScaler('cuda')

def gpu_augment_fast(x):
    # x: B C 32 32 float [0,1]
    x = F.pad(x, (4,4,4,4), mode='reflect') # B C 40 40
    B, C, H, W = x.shape
    # unfold to get all 81 crops
    # F.unfold expects N C H W -> N, C*kh*kw, L
    patches = F.unfold(x, kernel_size=32, stride=1) # B, C*1024, 81
    off_h = torch.randint(0, 9, (B,), device=x.device)
    off_w = torch.randint(0, 9, (B,), device=x.device)
    idx = off_h * 9 + off_w # B
    D = C*32*32
    idx_exp = idx.view(B,1,1).expand(B, D, 1)
    selected = torch.gather(patches, 2, idx_exp).squeeze(2) # B, D
    x = selected.view(B, C, 32, 32)
    # flip
    flip = torch.rand(B, device=x.device) < 0.5
    if flip.any():
        x[flip] = torch.flip(x[flip], dims=[3])
    return x

def cutmix_gpu(x,y,alpha=1.0):
    lam=np.random.beta(alpha,alpha)
    b=x.size(0)
    idx=torch.randperm(b,device=x.device)
    y_a,y_b=y,y[idx]
    W=x.size(3);H=x.size(2)
    cut_rat=np.sqrt(1.-lam)
    cut_w=int(W*cut_rat);cut_h=int(H*cut_rat)
    cx=np.random.randint(W);cy=np.random.randint(H)
    bbx1=np.clip(cx-cut_w//2,0,W);bby1=np.clip(cy-cut_h//2,0,H)
    bbx2=np.clip(cx+cut_w//2,0,W);bby2=np.clip(cy+cut_h//2,0,H)
    x[:,:,bby1:bby2,bbx1:bbx2]=x[idx][:,:,bby1:bby2,bbx1:bbx2]
    lam=1-((bbx2-bbx1)*(bby2-bby1)/(W*H))
    return x,y_a,y_b,lam

if os.path.exists(CKPT_DIR): shutil.rmtree(CKPT_DIR)
os.makedirs(CKPT_DIR,exist_ok=True)
start=time.time()
print(f"GPU fast {EPOCHS}ep B{BATCH} LR{LR} steps={steps_per_epoch}")

best=0
num_train=len(train_imgs)
for epoch in range(1,EPOCHS+1):
    model.train()
    t0=time.time()
    perm=torch.randperm(num_train)
    for i in range(0, num_train, BATCH):
        idx=perm[i:i+BATCH]
        if len(idx)<BATCH: continue
        x=train_imgs[idx].to(device, non_blocking=True).float().div(255.0)
        y=train_labels[idx].to(device, non_blocking=True)
        x=gpu_augment_fast(x)
        x=(x - MEAN) / STD
        x=x.to(memory_format=torch.channels_last)
        use=random.random()<0.5
        if use:
            x,y_a,y_b,lam=cutmix_gpu(x,y,1.0)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out=model(x)
            loss=(lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)) if use else criterion(out,y)
        scaler.scale(loss).backward()
        scaler.step(optimizer);scaler.update();scheduler.step()
    model.eval();correct=total=0
    with torch.no_grad():
        for i in range(0, len(test_imgs), 1024):
            x=test_imgs[i:i+1024].to(device, non_blocking=True).float().div(255.0)
            y=test_labels[i:i+1024].to(device, non_blocking=True)
            x=(x - MEAN)/STD
            x=x.to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda'):
                out=model(x)
            correct+=(out.argmax(1)==y).sum().item();total+=y.size(0)
    acc=correct/total;best=max(best,acc)
    print(f"Ep {epoch:02d} acc={acc*100:.2f} best={best*100:.2f} lr={optimizer.param_groups[0]['lr']:.4f} {time.time()-t0:.1f}s")
    torch.jit.script(model.eval()).save(os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt"))

shutil.copy(os.path.join(CKPT_DIR,f"ckpt_epoch_{EPOCHS:04d}.pt"),FINAL_CKPT)
wall=time.time()-start
print(f"Done wall={wall:.1f}s best={best*100:.2f}")
with open("/workspace/agent_reported_metrics.json","w") as f: json.dump({"accuracy":best,"wall_clock_sec":wall},f)
m=torch.jit.load(FINAL_CKPT);print("load ok",m(torch.randn(2,3,32,32,device=device)).shape)
