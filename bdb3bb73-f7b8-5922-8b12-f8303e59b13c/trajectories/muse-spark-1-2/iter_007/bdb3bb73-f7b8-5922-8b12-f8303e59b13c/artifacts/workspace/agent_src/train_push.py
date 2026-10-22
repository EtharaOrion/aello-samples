import os, time, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
CKPT_DIR="/workspace/scratch_push"
EPOCHS=32
BATCH=2048
BATCH_EVAL=2048
LR=1.8
WD=5e-4
MOM=0.9
LABEL_SMOOTH=0.1
WARMUP=2
MEAN=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
STD=torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)
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
def preload():
    print("Preloading...")
    classes=sorted(os.listdir(DATA_TRAIN))
    c2i={c:i for i,c in enumerate(classes)}
    paths=[];labels=[]
    for c in classes:
        d=os.path.join(DATA_TRAIN,c)
        for f in os.listdir(d):
            paths.append(os.path.join(d,f));labels.append(c2i[c])
    N=len(paths)
    data=torch.empty(N,3,32,32,dtype=torch.uint8)
    for i,p in enumerate(paths):
        img=Image.open(p).convert('RGB')
        arr=np.array(img,dtype=np.uint8)
        data[i]=torch.from_numpy(arr).permute(2,0,1)
    labels=torch.tensor(labels,dtype=torch.long)
    classes_t=sorted(os.listdir(DATA_TEST))
    c2i_t={c:i for i,c in enumerate(classes_t)}
    paths_t=[];labels_t=[]
    for c in classes_t:
        d=os.path.join(DATA_TEST,c)
        for f in os.listdir(d):
            paths_t.append(os.path.join(d,f));labels_t.append(c2i_t[c])
    Nt=len(paths_t)
    data_t=torch.empty(Nt,3,32,32,dtype=torch.uint8)
    for i,p in enumerate(paths_t):
        img=Image.open(p).convert('RGB')
        arr=np.array(img,dtype=np.uint8)
        data_t[i]=torch.from_numpy(arr).permute(2,0,1)
    labels_t=torch.tensor(labels_t,dtype=torch.long)
    print(f"Preloaded {N} {Nt}")
    return data,labels,data_t,labels_t
def gpu_augment(x_uint8, mean, std):
    B=x_uint8.size(0)
    x=x_uint8.float()
    x=F.pad(x,(4,4,4,4),mode='reflect')
    ho=random.randint(0,8)
    wo=random.randint(0,8)
    x=x[:,:,ho:ho+32, wo:wo+32]
    flip=torch.rand(B,device=x.device)<0.5
    if flip.any():
        x_flipped=torch.flip(x,dims=[3])
        mask=flip.view(B,1,1,1)
        x=torch.where(mask, x_flipped, x)
    x=x/255.0
    x=(x-mean.to(x.device))/std.to(x.device)
    erase=torch.rand(B,device=x.device)<0.25
    if erase.any():
        h=torch.randint(4,10,(B,),device=x.device)
        w=torch.randint(4,10,(B,),device=x.device)
        y1=torch.randint(0,32,(B,),device=x.device)
        x1=torch.randint(0,32,(B,),device=x.device)
        y2=torch.clamp(y1+h,max=32)
        x2=torch.clamp(x1+w,max=32)
        yy=torch.arange(32,device=x.device).view(1,32,1).expand(B,32,32)
        xx=torch.arange(32,device=x.device).view(1,1,32).expand(B,32,32)
        y1e=y1.view(B,1,1);y2e=y2.view(B,1,1);x1e=x1.view(B,1,1);x2e=x2.view(B,1,1)
        mask2d=(yy>=y1e)&(yy<y2e)&(xx>=x1e)&(xx<x2e)
        mask2d=mask2d&erase.view(B,1,1)
        mask3d=mask2d.unsqueeze(1)
        x=x.masked_fill(mask3d,0)
    return x
def main():
    os.makedirs(CKPT_DIR,exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        if f.startswith("ckpt_epoch_"):
            os.remove(os.path.join(CKPT_DIR,f))
    train_data,train_labels,test_data,test_labels=preload()
    train_data_gpu=train_data.to(device)
    train_labels_gpu=train_labels.to(device)
    test_data_gpu=test_data.to(device)
    test_labels_gpu=test_labels.to(device)
    print(f"GPU data {train_data_gpu.shape}")
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
    N=train_data_gpu.size(0)
    start_wall=time.time()
    print(f"Start push epochs={EPOCHS} batch={BATCH} lr={LR}")
    for epoch in range(1,EPOCHS+1):
        model.train()
        perm=torch.randperm(N,device=device)
        for i in range(0,N,BATCH):
            idx=perm[i:i+BATCH]
            if idx.size(0)!=BATCH:
                continue
            x_uint8=train_data_gpu[idx]
            y=train_labels_gpu[idx]
            x=gpu_augment(x_uint8,MEAN,STD)
            if random.random()<0.8:
                r=random.random()
                if r<0.5:
                    lam=random.betavariate(0.2,0.2)
                    perm2=torch.randperm(BATCH,device=device)
                    mixed_x=lam*x+(1-lam)*x[perm2]
                    y_a,y_b=y,y[perm2]
                    lam_a=lam
                else:
                    lam=random.betavariate(1.0,1.0)
                    perm2=torch.randperm(BATCH,device=device)
                    W=32;H=32
                    cut_rat=(1.-lam)**0.5
                    cut_w=int(W*cut_rat);cut_h=int(H*cut_rat)
                    cx=random.randint(0,W-1);cy=random.randint(0,H-1)
                    bbx1=max(0,cx-cut_w//2);bby1=max(0,cy-cut_h//2)
                    bbx2=min(W,cx+cut_w//2);bby2=min(H,cy+cut_h//2)
                    mixed_x=x.clone()
                    mixed_x[:,:,bby1:bby2,bbx1:bbx2]=x[perm2,:,bby1:bby2,bbx1:bbx2]
                    lam=1-((bbx2-bbx1)*(bby2-bby1)/(W*H))
                    y_a,y_b=y,y[perm2]
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
            for i in range(0,test_data_gpu.size(0),BATCH_EVAL):
                x=test_data_gpu[i:i+BATCH_EVAL].float()/255.0
                x=(x-MEAN.to(device))/STD.to(device)
                x=x.to(memory_format=torch.channels_last)
                y=test_labels_gpu[i:i+BATCH_EVAL]
                with torch.amp.autocast('cuda'):
                    out=model(x)
                pred=out.argmax(1)
                correct+=(pred==y).sum().item()
                total+=y.size(0)
        acc=correct/total
        elapsed=time.time()-start_wall
        print(f"Epoch {epoch:03d}/{EPOCHS} acc={acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} time={elapsed:.1f}s")
        to_script=model
        to_script.eval()
        scripted=torch.jit.script(to_script)
        ckpt_path=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(ckpt_path)
        to_script.train()
    total_time=time.time()-start_wall
    print(f"Done total_time={total_time:.1f}s")
    import json
    with open("/tmp/push_metrics.json","w") as f:
        json.dump({"accuracy":acc,"wall_clock_sec":total_time},f)
if __name__=="__main__":
    main()
