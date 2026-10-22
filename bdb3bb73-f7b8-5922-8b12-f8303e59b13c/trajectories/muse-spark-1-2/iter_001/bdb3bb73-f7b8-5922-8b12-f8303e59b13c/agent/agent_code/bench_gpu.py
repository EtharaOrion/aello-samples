import torch, time, os
import torch.nn.functional as F
from PIL import Image
import numpy as np

DATA_TRAIN="/workspace/data/cifar10/train"
device=torch.device("cuda")

classes=sorted(os.listdir(DATA_TRAIN))
c2i={c:i for i,c in enumerate(classes)}
train_imgs=[]
for cls in classes:
    d=os.path.join(DATA_TRAIN, cls)
    for fname in sorted(os.listdir(d)):
        train_imgs.append(np.array(Image.open(os.path.join(d,fname)).convert('RGB')))
train_imgs=np.stack(train_imgs)
train_t=torch.from_numpy(train_imgs).permute(0,3,1,2).contiguous()  # N C H W uint8
print(f"loaded {train_t.shape}")
train_g=train_t.cuda()
print(f"moved to gpu {train_g.shape} {train_g.dtype}")

# bench augment
def augment_gpu(imgs_uint8):
    B=imgs_uint8.size(0)
    x=imgs_uint8.float()/255.0
    x=F.pad(x,(4,4,4,4),mode='reflect')
    patches=F.unfold(x,kernel_size=32,stride=1)  # B, C*1024, 81
    h_off=torch.randint(0,9,(B,),device=x.device)
    w_off=torch.randint(0,9,(B,),device=x.device)
    idx=h_off*9+w_off
    idx_exp=idx.view(B,1,1).expand(B,patches.size(1),1)
    selected=patches.gather(2,idx_exp).squeeze(2)
    x=selected.view(B,3,32,32)
    flip=torch.rand(B,device=x.device)<0.5
    if flip.any():
        flipped=torch.flip(x,dims=[3])
        mask=flip.view(B,1,1,1)
        x=torch.where(mask,flipped,x)
    mean=torch.tensor([0.4914,0.4822,0.4465],device=x.device).view(1,3,1,1)
    std=torch.tensor([0.2470,0.2435,0.2616],device=x.device).view(1,3,1,1)
    x=(x-mean)/std
    return x

# warmup
for _ in range(5):
    b=train_g[:2048]
    augment_gpu(b)
torch.cuda.synchronize()
start=time.time()
for i in range(0,50000,2048):
    b=train_g[i:i+2048]
    augment_gpu(b)
torch.cuda.synchronize()
print(f"augment 50000 with 2048 batch: {time.time()-start:.3f}s")

# bench with loop erasing
import random
def augment_gpu_erase(imgs_uint8):
    B=imgs_uint8.size(0)
    x=imgs_uint8.float()/255.0
    x=F.pad(x,(4,4,4,4),mode='reflect')
    patches=F.unfold(x,kernel_size=32,stride=1)
    h_off=torch.randint(0,9,(B,),device=x.device)
    w_off=torch.randint(0,9,(B,),device=x.device)
    idx=h_off*9+w_off
    idx_exp=idx.view(B,1,1).expand(B,patches.size(1),1)
    selected=patches.gather(2,idx_exp).squeeze(2)
    x=selected.view(B,3,32,32)
    flip=torch.rand(B,device=x.device)<0.5
    if flip.any():
        flipped=torch.flip(x,dims=[3])
        mask=flip.view(B,1,1,1)
        x=torch.where(mask,flipped,x)
    mean=torch.tensor([0.4914,0.4822,0.4465],device=x.device).view(1,3,1,1)
    std=torch.tensor([0.2470,0.2435,0.2616],device=x.device).view(1,3,1,1)
    x=(x-mean)/std
    erase=torch.rand(B,device=x.device)<0.25
    if erase.any():
        erase_idx=torch.where(erase)[0]
        for i in erase_idx:
            h=random.randint(4,10)
            w=random.randint(4,10)
            y1=random.randint(0,32-h)
            x1=random.randint(0,32-w)
            x[i,:,y1:y1+h,x1:x1+w]=0
    return x

for _ in range(5):
    b=train_g[:2048]
    augment_gpu_erase(b)
torch.cuda.synchronize()
start=time.time()
for i in range(0,50000,2048):
    b=train_g[i:i+2048]
    augment_gpu_erase(b)
torch.cuda.synchronize()
print(f"with erase: {time.time()-start:.3f}s")

# bench model forward
import torch.nn as nn
class Block(nn.Module):
    def __init__(self,in_c,out_c,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_c,out_c,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(out_c)
        self.conv2=nn.Conv2d(out_c,out_c,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(out_c)
        self.shortcut=nn.Sequential()
        if stride!=1 or in_c!=out_c:
            self.shortcut=nn.Sequential(nn.Conv2d(in_c,out_c,1,stride,bias=False),nn.BatchNorm2d(out_c))
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)),inplace=True)
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        return F.relu(out,inplace=True)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,3,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.layer1=nn.Sequential(Block(64,128,1),Block(128,128,1))
        self.layer2=nn.Sequential(Block(128,256,2),Block(256,256,1))
        self.layer3=nn.Sequential(Block(256,512,2),Block(512,512,1))
        self.fc=nn.Linear(512,10)
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)),inplace=True)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=F.adaptive_avg_pool2d(out,1)
        out=out.view(out.size(0),-1)
        return self.fc(out)

model=Net().cuda().to(memory_format=torch.channels_last)
model.eval()
x=torch.randn(2048,3,32,32,device='cuda').to(memory_format=torch.channels_last)
torch.cuda.synchronize()
start=time.time()
for _ in range(25):
    with torch.amp.autocast('cuda'):
        model(x)
torch.cuda.synchronize()
print(f"model forward 25x2048: {time.time()-start:.3f}s {(time.time()-start)/25*1000:.1f}ms per batch")

# full epoch bench: augment + forward + backward
model.train()
opt=torch.optim.SGD(model.parameters(),lr=1.0)
scaler=torch.amp.GradScaler('cuda')
crit=nn.CrossEntropyLoss()
torch.cuda.synchronize()
start=time.time()
for i in range(0,50000,2048):
    b=train_g[i:i+2048]
    x=augment_gpu(b)
    x=x.to(memory_format=torch.channels_last)
    y=torch.randint(0,10,(x.size(0),),device='cuda')
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        out=model(x)
        loss=crit(out,y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
torch.cuda.synchronize()
print(f"full epoch 2048: {time.time()-start:.3f}s")
