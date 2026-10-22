import os, time, shutil, random, glob
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import numpy as np

CKPT_DIR="/workspace/scratch_ckpt"
DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
BATCH=1024
EPOCHS=40
LR=1.0
WD=5e-4
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True

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
    def __init__(self,d=28,w=4,nc=10):
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

for w in [4,6,8]:
    print(f"\n=== Testing widen {w} ===")
    model=WRN(d=28,w=w).to(device).to(memory_format=torch.channels_last)
    # count params
    params=sum(p.numel() for p in model.parameters())
    print(f"Params {params/1e6:.1f}M")
    # time one epoch
    import time as t
    model.train()
    opt=torch.optim.SGD(model.parameters(),lr=1.0,momentum=0.9,weight_decay=5e-4)
    scaler=torch.amp.GradScaler('cuda')
    criterion=nn.CrossEntropyLoss()
    # dummy data
    dummy_x=torch.randn(BATCH,3,32,32,device=device).to(memory_format=torch.channels_last)
    dummy_y=torch.randint(0,10,(BATCH,),device=device)
    # warmup
    for _ in range(5):
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            o=model(dummy_x)
            loss=criterion(o,dummy_y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    start=t.time()
    for _ in range(20):
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            o=model(dummy_x)
            loss=criterion(o,dummy_y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    elapsed=t.time()-start
    print(f"20 steps time {elapsed:.2f}s per step {elapsed/20*1000:.1f}ms")
    del model, opt, scaler
    torch.cuda.empty_cache()
