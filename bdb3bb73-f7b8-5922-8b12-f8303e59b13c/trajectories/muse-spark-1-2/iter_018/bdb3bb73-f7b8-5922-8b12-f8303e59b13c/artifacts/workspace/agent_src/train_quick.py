import os, time, shutil, random, json, pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

EPOCHS=50
BATCH=512
LR=0.8
WD=5e-4
CKPT_DIR="/workspace/periodic_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR,f))

device=torch.device("cuda")
print(torch.cuda.get_device_name(0))

mean=(0.4914,0.4822,0.4465)
std=(0.2470,0.2435,0.2616)

train_transform=transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])
test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])

train_set=ImageFolder("/workspace/data/cifar10/train",transform=train_transform)
test_set=ImageFolder("/workspace/data/cifar10/test",transform=test_transform)
train_loader=DataLoader(train_set,batch_size=BATCH,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
test_loader=DataLoader(test_set,batch_size=512,shuffle=False,num_workers=2,pin_memory=True)

class BasicBlock(nn.Module):
    def __init__(self,in_planes,out_planes,stride,dropRate=0.0):
        super().__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(in_planes,out_planes,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(out_planes)
        self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_planes,out_planes,3,1,1,bias=False)
        self.droprate=dropRate
        self.equalInOut=(in_planes==out_planes)
        if not self.equalInOut:
            self.convShortcut=nn.Conv2d(in_planes,out_planes,1,stride,0,bias=False)
        else:
            self.convShortcut=None
    def forward(self,x):
        if self.equalInOut:
            out=self.relu1(self.bn1(x))
            out=self.relu2(self.bn2(self.conv1(out)))
            if self.droprate>0:
                out=F.dropout(out,p=self.droprate,training=self.training)
            out=self.conv2(out)
            return out+x
        else:
            out=self.relu1(self.bn1(x))
            out=self.conv1(out)
            out=self.relu2(self.bn2(out))
            if self.droprate>0:
                out=F.dropout(out,p=self.droprate,training=self.training)
            out=self.conv2(out)
            assert self.convShortcut is not None
            return out+self.convShortcut(x)

class NetworkBlock(nn.Module):
    def __init__(self,nb_layers,in_planes,out_planes,block,stride,dropRate=0.0):
        super().__init__()
        self.layer=self._make_layer(block,in_planes,out_planes,nb_layers,stride,dropRate)
    def _make_layer(self,block,in_planes,out_planes,nb_layers,stride,dropRate):
        layers=[]
        for i in range(nb_layers):
            layers.append(block(i==0 and in_planes or out_planes,out_planes,i==0 and stride or 1,dropRate))
        return nn.Sequential(*layers)
    def forward(self,x): return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self,depth=28,widen_factor=10,dropRate=0.0,num_classes=10):
        super().__init__()
        nChannels=[16,16*widen_factor,32*widen_factor,64*widen_factor]
        assert (depth-4)%6==0
        n=(depth-4)//6
        block=BasicBlock
        self.conv1=nn.Conv2d(3,nChannels[0],3,1,1,bias=False)
        self.block1=NetworkBlock(n,nChannels[0],nChannels[1],block,1,dropRate)
        self.block2=NetworkBlock(n,nChannels[1],nChannels[2],block,2,dropRate)
        self.block3=NetworkBlock(n,nChannels[2],nChannels[3],block,2,dropRate)
        self.bn1=nn.BatchNorm2d(nChannels[3])
        self.relu=nn.ReLU(inplace=True)
        self.fc=nn.Linear(nChannels[3],num_classes)
        self.nChannels=nChannels[3]
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                m.bias.data.zero_()
    def forward(self,x):
        out=self.conv1(x)
        out=self.block1(out)
        out=self.block2(out)
        out=self.block3(out)
        out=self.relu(self.bn1(out))
        out=F.adaptive_avg_pool2d(out,1)
        out=out.view(-1,self.nChannels)
        return self.fc(out)

model=WideResNet(28,10,dropRate=0.0).to(device)
model=model.to(memory_format=torch.channels_last)
print(f"Params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
# quick script test
torch.jit.script(model)
print("script ok")

criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
scaler=torch.amp.GradScaler('cuda')

def mixup_data(x,y,alpha=0.2):
    lam=random.betavariate(alpha,alpha)
    idx=torch.randperm(x.size(0),device=x.device)
    return lam*x+(1-lam)*x[idx], y, y[idx], lam
def cutmix_data(x,y,alpha=1.0):
    lam=random.betavariate(alpha,alpha)
    idx=torch.randperm(x.size(0),device=x.device)
    W,H=x.size(2),x.size(3)
    cut_rat=(1-lam)**0.5
    cut_w,cut_h=int(W*cut_rat),int(H*cut_rat)
    cx,cy=random.randint(0,W),random.randint(0,H)
    bbx1=max(cx-cut_w//2,0); bby1=max(cy-cut_h//2,0)
    bbx2=min(cx+cut_w//2,W); bby2=min(cy+cut_h//2,H)
    x[:,:,bbx1:bbx2,bby1:bby2]=x[idx,:,bbx1:bbx2,bby1:bby2]
    lam=1-((bbx2-bbx1)*(bby2-bby1)/(W*H))
    return x,y,y[idx],lam
def mixup_criterion(criterion,pred,y_a,y_b,lam):
    return lam*criterion(pred,y_a)+(1-lam)*criterion(pred,y_b)

start_wall=time.time()
for epoch in range(1,EPOCHS+1):
    model.train()
    t0=time.time()
    for inputs,targets in train_loader:
        inputs=inputs.to(device,non_blocking=True).to(memory_format=torch.channels_last)
        targets=targets.to(device,non_blocking=True)
        r=random.random()
        use_mix=r<0.5
        is_cutmix=r<0.25
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            if use_mix:
                if is_cutmix:
                    inputs,y_a,y_b,lam=cutmix_data(inputs,targets,alpha=1.0)
                else:
                    inputs,y_a,y_b,lam=mixup_data(inputs,targets,alpha=0.2)
                outputs=model(inputs)
                loss=mixup_criterion(criterion,outputs,y_a,y_b,lam)
            else:
                outputs=model(inputs)
                loss=criterion(outputs,targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    epoch_train=time.time()-t0
    model.eval()
    correct=total=0
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for inputs,targets in test_loader:
                inputs=inputs.to(device,non_blocking=True).to(memory_format=torch.channels_last)
                targets=targets.to(device,non_blocking=True)
                outputs=model(inputs)
                _,pred=outputs.max(1)
                correct+=pred.eq(targets).sum().item()
                total+=targets.size(0)
    acc=correct/total
    total_time=time.time()-start_wall
    print(f"Epoch {epoch}/{EPOCHS} acc {acc*100:.2f}% lr {optimizer.param_groups[0]['lr']:.4f} train {epoch_train:.1f}s total {total_time:.1f}s")
    model.eval()
    scripted=torch.jit.script(model)
    ckpt_path=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    loaded=torch.jit.load(ckpt_path,map_location="cpu")
    loaded.eval()
    with torch.no_grad():
        dummy=torch.randn(2,3,32,32)
        _=loaded(dummy)
    print(f" -> saved {ckpt_path}")
    model.train()
    if acc>=0.94:
        print(f"*** Hit 94% at epoch {epoch} ***")

shutil.copy(os.path.join(CKPT_DIR,f"ckpt_epoch_{EPOCHS:04d}.pt"),"/workspace/checkpoint.pt")
wall=time.time()-start_wall
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy":acc,"wall_clock_sec":wall},f)
print(f"Done wall={wall:.1f}s acc={acc:.4f}")
