import os, time, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
CKPT_DIR="/workspace/scratch_cpu"
EPOCHS=30
BATCH=2048
BATCH_EVAL=1024
LR=1.5
WD=5e-4
MOM=0.9
LABEL_SMOOTH=0.1
WARMUP=3
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision('high')

class Block(nn.Module):
    def __init__(self,in_ch,out_ch,stride):
        super().__init__()
        self.conv1=nn.Conv2d(in_ch,out_ch,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(out_ch)
        self.conv2=nn.Conv2d(out_ch,out_ch,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(out_ch)
        self.shortcut=nn.Sequential()
        if stride!=1 or in_ch!=out_ch:
            self.shortcut=nn.Sequential(nn.Conv2d(in_ch,out_ch,1,stride,bias=False),nn.BatchNorm2d(out_ch))
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return out

class CifarResNet(nn.Module):
    def __init__(self,widen=4):
        super().__init__()
        c1=16*widen;c2=32*widen;c3=64*widen
        self.conv1=nn.Conv2d(3,c1,3,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(c1)
        self.layer1=self._make(c1,c1,4,1)
        self.layer2=self._make(c1,c2,4,2)
        self.layer3=self._make(c2,c3,4,2)
        self.bn_final=nn.BatchNorm2d(c3)
        self.fc=nn.Linear(c3,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1);nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.constant_(m.bias,0)
    def _make(self,in_ch,out_ch,n,stride):
        layers=[Block(in_ch,out_ch,stride)]
        for _ in range(1,n):
            layers.append(Block(out_ch,out_ch,1))
        return nn.Sequential(*layers)
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=F.relu(self.bn_final(out))
        out=F.adaptive_avg_pool2d(out,1)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

def rand_bbox(size,lam):
    W=size[-1];H=size[-2]
    cut_rat=(1.-lam)**0.5
    cut_w=int(W*cut_rat);cut_h=int(H*cut_rat)
    cx=random.randint(0,W-1);cy=random.randint(0,H-1)
    bbx1=max(0,cx-cut_w//2);bby1=max(0,cy-cut_h//2)
    bbx2=min(W,cx+cut_w//2);bby2=min(H,cy+cut_h//2)
    return bbx1,bby1,bbx2,bby2

def main():
    os.makedirs(CKPT_DIR,exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        if f.startswith("ckpt_epoch_"):
            os.remove(os.path.join(CKPT_DIR,f))
    train_tf=transforms.Compose([
        transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD),
    ])
    test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    train_ds=datasets.ImageFolder(DATA_TRAIN,transform=train_tf)
    test_ds=datasets.ImageFolder(DATA_TEST,transform=test_tf)
    train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=4,drop_last=True)
    test_loader=DataLoader(test_ds,batch_size=BATCH_EVAL,shuffle=False,num_workers=4,pin_memory=True)
    model=CifarResNet(widen=4).to(device)
    model=model.to(memory_format=torch.channels_last)
    optimizer=torch.optim.SGD(model.parameters(),lr=LR,momentum=MOM,weight_decay=WD,nesterov=True)
    def lr_lambda(epoch):
        e=epoch+1
        if e<=WARMUP:
            return e/WARMUP
        else:
            prog=(e-WARMUP)/(EPOCHS-WARMUP)
            return 0.5*(1+math.cos(math.pi*prog))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
    scaler=torch.amp.GradScaler('cuda')
    criterion=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    start_wall=time.time()
    print(f"Start cpu_fast epochs={EPOCHS} batch={BATCH} lr={LR}")
    for epoch in range(1,EPOCHS+1):
        model.train()
        for x,y in train_loader:
            x=x.to(device,non_blocking=True,memory_format=torch.channels_last)
            y=y.to(device,non_blocking=True)
            if random.random()<0.8:
                r=random.random()
                if r<0.5:
                    lam=random.betavariate(0.2,0.2)
                    idx=torch.randperm(x.size(0),device=device)
                    mixed_x=lam*x+(1-lam)*x[idx]
                    y_a,y_b=y,y[idx]
                    lam_a=lam
                else:
                    lam=random.betavariate(1.0,1.0)
                    idx=torch.randperm(x.size(0),device=device)
                    bbx1,bby1,bbx2,bby2=rand_bbox(x.size(),lam)
                    mixed_x=x.clone()
                    mixed_x[:,:,bby1:bby2,bbx1:bbx2]=x[idx,:,bby1:bby2,bbx1:bbx2]
                    lam=1-((bbx2-bbx1)*(bby2-bby1)/(x.size(-1)*x.size(-2)))
                    y_a,y_b=y,y[idx]
                    lam_a=lam
                with torch.amp.autocast('cuda'):
                    logits=model(mixed_x)
                    loss=lam_a*criterion(logits,y_a)+(1-lam_a)*criterion(logits,y_b)
            else:
                with torch.amp.autocast('cuda'):
                    logits=model(x)
                    loss=criterion(logits,y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        model.eval()
        correct=0;total=0
        with torch.no_grad():
            for x,y in test_loader:
                x=x.to(device,non_blocking=True,memory_format=torch.channels_last)
                y=y.to(device,non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out=model(x)
                pred=out.argmax(1)
                correct+=(pred==y).sum().item()
                total+=y.size(0)
        acc=correct/total
        elapsed=time.time()-start_wall
        print(f"Epoch {epoch:03d}/{EPOCHS} acc={acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} time={elapsed:.1f}s")
        ckpt_path=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
        scripted=torch.jit.script(model.eval())
        scripted.save(ckpt_path)
        model.train()
    total_time=time.time()-start_wall
    print(f"Done total_time={total_time:.1f}s")
    import json
    with open("/tmp/cpu_fast_metrics.json","w") as f:
        json.dump({"accuracy":acc,"wall_clock_sec":total_time},f)

if __name__=="__main__":
    main()
