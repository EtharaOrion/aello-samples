import os, time, shutil, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np

DATA_ROOT="/workspace/data/cifar10"
CKPT_DIR="/workspace/scratch_ckpt"
CLASSES=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CLS_TO_IDX={c:i for i,c in enumerate(CLASSES)}
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)

class CIFAR10PNG(Dataset):
    def __init__(self, root, split, transform=None):
        self.transform=transform
        self.samples=[]
        base=os.path.join(root,split)
        for cls in CLASSES:
            d=os.path.join(base,cls)
            for fname in os.listdir(d):
                self.samples.append((os.path.join(d,fname), CLS_TO_IDX[cls]))
        random.seed(0)
        random.shuffle(self.samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path,label=self.samples[idx]
        img=Image.open(path).convert("RGB")
        if self.transform:
            img=self.transform(img)
        return img,label

class BasicBlock(nn.Module):
    def __init__(self,in_planes,out_planes,stride,dropout=0.0):
        super().__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(in_planes,out_planes,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(out_planes)
        self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_planes,out_planes,3,1,1,bias=False)
        self.dropout=nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.shortcut=nn.Identity() if in_planes==out_planes and stride==1 else nn.Conv2d(in_planes,out_planes,1,stride,bias=False)
    def forward(self,x):
        out=self.conv1(self.relu1(self.bn1(x)))
        out=self.dropout(self.conv2(self.relu2(self.bn2(out))))
        out+=self.shortcut(x)
        return out

class NetworkBlock(nn.Module):
    def __init__(self,n,in_planes,out_planes,stride,dropout):
        super().__init__()
        layers=[]
        for i in range(n):
            s=stride if i==0 else 1
            inp=in_planes if i==0 else out_planes
            layers.append(BasicBlock(inp,out_planes,s,dropout))
        self.layer=nn.Sequential(*layers)
    def forward(self,x): return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self,depth=28,widen=10,num_classes=10,dropout=0.0):
        super().__init__()
        assert (depth-4)%6==0
        n=(depth-4)//6; k=widen
        self.conv1=nn.Conv2d(3,16,3,1,1,bias=False)
        self.block1=NetworkBlock(n,16,16*k,1,dropout)
        self.block2=NetworkBlock(n,16*k,32*k,2,dropout)
        self.block3=NetworkBlock(n,32*k,64*k,2,dropout)
        self.bn=nn.BatchNorm2d(64*k)
        self.relu=nn.ReLU(inplace=True)
        self.fc=nn.Linear(64*k,num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m,nn.Linear): m.bias.data.zero_()
    def forward(self,x):
        out=self.conv1(x)
        out=self.block1(out)
        out=self.block2(out)
        out=self.block3(out)
        out=self.relu(self.bn(out))
        out=F.adaptive_avg_pool2d(out,1)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

def main():
    t0=time.time()
    device=torch.device("cuda")
    print(f"GPU {torch.cuda.get_device_name(0)}")

    # Fast CPU aug: RandomCrop + Flip + TrivialAugmentWide (PIL, fast)
    train_tf=T.Compose([
        T.RandomCrop(32, padding=4, padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    test_tf=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    train_ds=CIFAR10PNG(DATA_ROOT,"train",train_tf)
    test_ds=CIFAR10PNG(DATA_ROOT,"test",test_tf)

    BATCH=512
    # Use 4 workers to avoid shm, but with prefetch
    train_loader=DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader=DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

    model=WideResNet(depth=28,widen=10,dropout=0.0).to(device).to(memory_format=torch.channels_last)

    EPOCHS=30
    # OneCycle with high max_lr for fast convergence
    optimizer=torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.4, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)

    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler=torch.amp.GradScaler('cuda')

    if os.path.exists(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR, exist_ok=True)

    best=0
    for epoch in range(1, EPOCHS+1):
        model.train()
        t_epoch=time.time()
        for x,y in train_loader:
            x,y=x.to(device, non_blocking=True).to(memory_format=torch.channels_last), y.to(device, non_blocking=True)
            # Mixup/CutMix 50%
            if random.random()<0.5:
                if random.random()<0.5:
                    lam=np.random.beta(0.2,0.2)
                    perm=torch.randperm(x.size(0), device=device)
                    x_mixed=lam*x + (1-lam)*x[perm]
                    y_a=y; y_b=y[perm]
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        out=model(x_mixed)
                        loss=lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)
                else:
                    lam=np.random.beta(1.0,1.0)
                    perm=torch.randperm(x.size(0), device=device)
                    W=32; H=32
                    cut_rat=(1-lam)**0.5
                    cut_w=int(W*cut_rat); cut_h=int(H*cut_rat)
                    cx=np.random.randint(W); cy=np.random.randint(H)
                    bbx1=np.clip(cx-cut_w//2,0,W); bby1=np.clip(cy-cut_h//2,0,H)
                    bbx2=np.clip(cx+cut_w//2,0,W); bby2=np.clip(cy+cut_h//2,0,H)
                    x[:, :, bby1:bby2, bbx1:bbx2]=x[perm, :, bby1:bby2, bbx1:bbx2]
                    lam2=1-((bbx2-bbx1)*(bby2-bby1)/(32*32))
                    y_a=y; y_b=y[perm]
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        out=model(x)
                        loss=lam2*criterion(out,y_a)+(1-lam2)*criterion(out,y_b)
            else:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out=model(x)
                    loss=criterion(out,y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        correct=0; total=0
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for x,y in test_loader:
                    x,y=x.to(device, non_blocking=True).to(memory_format=torch.channels_last), y.to(device, non_blocking=True)
                    out=model(x)
                    pred=out.argmax(1)
                    correct+=(pred==y).sum().item()
                    total+=y.size(0)
        acc=correct/total
        best=max(best,acc)
        dt=time.time()-t_epoch
        tot=time.time()-t0
        print(f"Epoch {epoch:02d}/{EPOCHS} acc={acc*100:.2f}% best={best*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} epoch_time={dt:.2f}s total={tot:.1f}s")
        model.eval()
        try:
            scripted=torch.jit.script(model)
            path=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
            scripted.save(path)
            print(f"  saved {path}")
        except Exception as e:
            print(f"  script fail {e}")
            try:
                ex=torch.randn(1,3,32,32, device=device).to(memory_format=torch.channels_last)
                traced=torch.jit.trace(model, ex)
                path=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
                traced.save(path)
            except Exception as e2:
                print(f"  trace fail {e2}")
        model.train()

    wall=time.time()-t0
    print(f"Done wall={wall:.1f}s best={best*100:.2f}%")
    with open("/tmp/scratch_metrics.json","w") as f:
        json.dump({"accuracy":best,"wall_clock_sec":wall},f)
    last=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    if os.path.exists(last):
        try:
            m=torch.jit.load(last, map_location="cuda")
            m.eval()
            with torch.no_grad():
                x=torch.randn(2,3,32,32, device="cuda")
                y=m(x)
                print(f"Verify OK {y.shape}")
        except Exception as e:
            print(f"Verify fail {e}")

if __name__=="__main__":
    main()
