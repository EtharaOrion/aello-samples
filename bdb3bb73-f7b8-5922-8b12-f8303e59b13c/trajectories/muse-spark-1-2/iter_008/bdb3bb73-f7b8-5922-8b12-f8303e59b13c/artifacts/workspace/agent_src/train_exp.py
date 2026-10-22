import os, time, random, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def run(BATCH_TRAIN, LR, EPOCHS, WD, aug="simple", compile=False):
    DATA_TRAIN="/workspace/data/cifar10/train"
    DATA_TEST="/workspace/data/cifar10/test"
    CKPT_DIR="/workspace/scratch_ckpt"
    BATCH_TEST=1024
    NUM_WORKERS=2
    MEAN=(0.4914,0.4822,0.4465)
    STD=(0.2470,0.2435,0.2616)
    device=torch.device("cuda")
    torch.backends.cudnn.benchmark=True

    class WRNBlock(nn.Module):
        def __init__(self,in_ch,out_ch,stride):
            super().__init__()
            self.bn1=nn.BatchNorm2d(in_ch)
            self.conv1=nn.Conv2d(in_ch,out_ch,3,stride,1,bias=False)
            self.bn2=nn.BatchNorm2d(out_ch)
            self.conv2=nn.Conv2d(out_ch,out_ch,3,1,1,bias=False)
            self.is_conv=stride!=1 or in_ch!=out_ch
            self.shortcut=nn.Conv2d(in_ch,out_ch,1,stride,bias=False) if self.is_conv else nn.Identity()
        def forward(self,x):
            out=F.relu(self.bn1(x),inplace=True)
            sc=self.shortcut(x) if self.is_conv else x
            out=self.conv1(out)
            out=self.conv2(F.relu(self.bn2(out),inplace=True))
            return out+sc
    class WRN(nn.Module):
        def __init__(self,depth=28,widen=10,num_classes=10):
            super().__init__()
            n=(depth-4)//6
            self.conv1=nn.Conv2d(3,16,3,1,1,bias=False)
            c1,c2,c3=16*widen,32*widen,64*widen
            self.layer1=self._make(16,c1,n,1)
            self.layer2=self._make(c1,c2,n,2)
            self.layer3=self._make(c2,c3,n,2)
            self.bn=nn.BatchNorm2d(c3)
            self.fc=nn.Linear(c3,num_classes)
            for m in self.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                elif isinstance(m,nn.BatchNorm2d):
                    nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
                elif isinstance(m,nn.Linear):
                    nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                    nn.init.constant_(m.bias,0)
        def _make(self,in_ch,out_ch,n,stride):
            layers=[WRNBlock(in_ch,out_ch,stride)]
            for _ in range(1,n):
                layers.append(WRNBlock(out_ch,out_ch,1))
            return nn.Sequential(*layers)
        def forward(self,x):
            out=self.conv1(x)
            out=self.layer1(out)
            out=self.layer2(out)
            out=self.layer3(out)
            out=F.relu(self.bn(out),inplace=True)
            out=F.adaptive_avg_pool2d(out,1).flatten(1)
            out=self.fc(out)
            return out

    if aug=="simple":
        train_tf=transforms.Compose([
            transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN,STD),
            transforms.RandomErasing(p=0.25),
        ])
    elif aug=="auto":
        train_tf=transforms.Compose([
            transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN,STD),
        ])
    elif aug=="trivial":
        train_tf=transforms.Compose([
            transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN,STD),
            transforms.RandomErasing(p=0.25),
        ])
    test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    train_ds=datasets.ImageFolder(DATA_TRAIN,transform=train_tf)
    test_ds=datasets.ImageFolder(DATA_TEST,transform=test_tf)
    train_loader=DataLoader(train_ds,batch_size=BATCH_TRAIN,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True,prefetch_factor=2,drop_last=True)
    test_loader=DataLoader(test_ds,batch_size=BATCH_TEST,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True,prefetch_factor=2)

    model=WRN(depth=28,widen=10).to(device)
    model=model.to(memory_format=torch.channels_last)
    if compile:
        try:
            model=torch.compile(model)
            print("compiled")
        except: pass
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    def rand_bbox(size,lam):
        W=size[-1]; H=size[-2]
        cut_rat=(1.-lam)**0.5
        cut_w=int(W*cut_rat); cut_h=int(H*cut_rat)
        cx=random.randint(0,W); cy=random.randint(0,H)
        bbx1=max(0,cx-cut_w//2); bby1=max(0,cy-cut_h//2)
        bbx2=min(W,cx+cut_w//2); bby2=min(H,cy+cut_h//2)
        return bbx1,bby1,bbx2,bby2
    def evaluate(m,loader):
        m.eval()
        correct=total=0
        with torch.no_grad():
            for x,y in loader:
                x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out=m(x)
                correct+=(out.argmax(1)==y).sum().item()
                total+=y.size(0)
        return correct/total
    os.makedirs(CKPT_DIR,exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        try: os.remove(os.path.join(CKPT_DIR,f))
        except: pass
    start=time.time()
    first=None; last=None
    warmup=5
    for epoch in range(1,EPOCHS+1):
        model.train()
        if epoch<=warmup:
            lr=LR*epoch/warmup
            for pg in optimizer.param_groups: pg['lr']=lr
        else:
            scheduler.step()
        tot_loss=0
        for x,y in train_loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            x=x.to(memory_format=torch.channels_last)
            use_mix=False
            if random.random()<0.5:
                lam=random.betavariate(1.0,1.0)
                if random.random()<0.5:
                    idx=torch.randperm(x.size(0),device=device)
                    x_mix=lam*x+(1-lam)*x[idx]
                    y_a,y_b=y,y[idx]
                    lam_a=lam
                    x=x_mix; use_mix=True
                else:
                    idx=torch.randperm(x.size(0),device=device)
                    bbx1,bby1,bbx2,bby2=rand_bbox(x.size(),lam)
                    if bbx2>bbx1 and bby2>bby1:
                        x[:, :, bby1:bby2, bbx1:bbx2]=x[idx, :, bby1:bby2, bbx1:bbx2]
                        lam=1-((bbx2-bbx1)*(bby2-bby1)/(x.size(-1)*x.size(-2)))
                        y_a,y_b=y,y[idx]
                        lam_a=lam
                        use_mix=True
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out=model(x)
                if use_mix:
                    loss=lam_a*criterion(out,y_a)+(1-lam_a)*criterion(out,y_b)
                else:
                    loss=criterion(out,y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(optimizer)
            scaler.update()
            tot_loss+=loss.item()
            if tot_loss!=tot_loss:
                print(f"NaN at epoch {epoch} batch"); return None
        acc=evaluate(model,test_loader)
        print(f"Epoch {epoch:02d} loss {tot_loss/len(train_loader):.3f} acc {acc*100:.2f}% lr {optimizer.param_groups[0]['lr']:.4f} time {time.time()-start:.1f}s")
        model.eval()
        to_script=model._orig_mod if hasattr(model,'_orig_mod') else model
        to_script.eval()
        scripted=torch.jit.script(to_script)
        ckpt=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(ckpt)
        to_script.train()
        model.train()
        if first is None: first=time.time()
        last=time.time()
        if acc>=0.94:
            print(f"hit 94 at {epoch} time {last-start:.1f}")
    ckpts=sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_epoch_")])
    if ckpts:
        last_ckpt=os.path.join(CKPT_DIR,ckpts[-1])
        m_final=torch.jit.load(last_ckpt,map_location=device)
        m_final.eval()
        acc_final=evaluate(m_final,test_loader)
        print(f"final acc {acc_final*100:.2f} wall {last-start:.1f} span {last-first:.1f} epochs {len(ckpts)}")
        return {"acc":acc_final, "wall":last-start, "span":last-first, "epochs":len(ckpts)}
    return None

# Try configs
import sys
configs=[
    (1024, 0.6, 25, 5e-4, "simple", False),
    (1024, 0.8, 25, 5e-4, "simple", False),
    (2048, 0.8, 25, 5e-4, "simple", False),
    (1024, 0.4, 25, 5e-4, "trivial", False),
]

for cfg in configs:
    print(f"\n=== Trying {cfg} ===")
    res=run(*cfg)
    print(f"Result {cfg} -> {res}")
    if res and res["acc"]>=0.94 and res["wall"]<120:
        print("GOOD CONFIG FOUND")
        with open("/tmp/best_exp.json","w") as f:
            json.dump({"cfg":cfg, "res":res}, f)
        break
