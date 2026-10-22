import os, time, shutil, random, json, glob, sys
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import numpy as np

def run(depth, widen, epochs, lr_max, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    for f in os.listdir(ckpt_dir):
        os.remove(os.path.join(ckpt_dir,f))
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    device=torch.device("cuda")
    MEAN=torch.tensor([0.4914,0.4822,0.4465],device=device).view(1,3,1,1)
    STD=torch.tensor([0.2470,0.2435,0.2616],device=device).view(1,3,1,1)
    train_root="/workspace/data/cifar10/train"
    test_root="/workspace/data/cifar10/test"
    classes=sorted(os.listdir(train_root))
    class_to_idx={c:i for i,c in enumerate(classes)}
    def load_folder(root):
        paths=[];labels=[]
        for c in classes:
            for p in glob.glob(os.path.join(root,c,"*.png")):
                paths.append(p);labels.append(class_to_idx[c])
        N=len(paths)
        data=torch.empty(N,3,32,32,dtype=torch.uint8)
        for i,p in enumerate(paths):
            img=np.array(Image.open(p).convert("RGB"))
            data[i]=torch.from_numpy(img).permute(2,0,1)
        return data, torch.tensor(labels,dtype=torch.long)
    train_u8,train_lbl=load_folder(train_root)
    test_u8,test_lbl=load_folder(test_root)
    train_u8_gpu=train_u8.to(device)
    train_lbl_gpu=train_lbl.to(device)
    test_f=test_u8.float().div(255.0).to(device)
    test_norm=(test_f-MEAN)/STD
    test_lbl_gpu=test_lbl.to(device)
    class BasicBlock(nn.Module):
        def __init__(self,in_ch,out_ch,stride,dropout=0.0):
            super().__init__()
            self.bn1=nn.BatchNorm2d(in_ch);self.relu1=nn.ReLU(inplace=True)
            self.conv1=nn.Conv2d(in_ch,out_ch,3,stride,1,bias=False)
            self.bn2=nn.BatchNorm2d(out_ch);self.relu2=nn.ReLU(inplace=True)
            self.conv2=nn.Conv2d(out_ch,out_ch,3,1,1,bias=False)
            self.dropout=nn.Dropout(dropout) if dropout>0 else nn.Identity()
            self.shortcut=nn.Identity()
            if stride!=1 or in_ch!=out_ch:
                self.shortcut=nn.Conv2d(in_ch,out_ch,1,stride,bias=False)
        def forward(self,x):
            out=self.relu1(self.bn1(x))
            shortcut=self.shortcut(out if isinstance(self.shortcut,nn.Conv2d) else x)
            out=self.conv1(out)
            out=self.conv2(self.dropout(self.relu2(self.bn2(out))))
            return out+shortcut
    class WideResNet(nn.Module):
        def __init__(self,depth,widen,dropout=0.0,num_classes=10):
            super().__init__()
            assert (depth-4)%6==0
            n=(depth-4)//6
            self.conv1=nn.Conv2d(3,16,3,1,1,bias=False)
            self.layer1=self._make(16,16*widen,n,1,dropout)
            self.layer2=self._make(16*widen,32*widen,n,2,dropout)
            self.layer3=self._make(32*widen,64*widen,n,2,dropout)
            self.bn=nn.BatchNorm2d(64*widen);self.relu=nn.ReLU(inplace=True)
            self.pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(64*widen,num_classes)
            for m in self.modules():
                if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                elif isinstance(m,nn.BatchNorm2d): nn.init.constant_(m.weight,1);nn.init.constant_(m.bias,0)
                elif isinstance(m,nn.Linear): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu');nn.init.constant_(m.bias,0)
        def _make(self,in_ch,out_ch,n,stride,dropout):
            layers=[BasicBlock(in_ch,out_ch,stride,dropout)]
            for _ in range(1,n): layers.append(BasicBlock(out_ch,out_ch,1,dropout))
            return nn.Sequential(*layers)
        def forward(self,x):
            out=self.conv1(x);out=self.layer1(out);out=self.layer2(out);out=self.layer3(out)
            out=self.relu(self.bn(out));out=self.pool(out);out=out.view(out.size(0),-1);out=self.fc(out);return out
    BATCH=1024;WD=5e-4;LABEL_SMOOTH=0.1;MIXUP=0.2
    model=WideResNet(depth,widen,0.0).to(device).to(memory_format=torch.channels_last)
    optimizer=torch.optim.SGD(model.parameters(),lr=lr_max,momentum=0.9,weight_decay=WD,nesterov=True)
    steps_per_epoch=(len(train_u8)+BATCH-1)//BATCH
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr_max,epochs=epochs,steps_per_epoch=steps_per_epoch,pct_start=0.3,anneal_strategy='cos',div_factor=10,final_div_factor=100)
    criterion=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler=torch.amp.GradScaler('cuda')
    def gpu_augment(x):
        x=F.pad(x,(4,4,4,4),mode='reflect')
        x_unfold=x.unfold(2,32,1).unfold(3,32,1)
        x_unfold=x_unfold.permute(0,2,3,1,4,5)
        N=x.size(0)
        h_off=torch.randint(0,9,(N,),device=x.device)
        w_off=torch.randint(0,9,(N,),device=x.device)
        idx=torch.arange(N,device=x.device)
        out=x_unfold[idx,h_off,w_off]
        flip=torch.rand(N,device=x.device)<0.5
        if flip.any(): out[flip]=torch.flip(out[flip],dims=[3])
        return out
    start_wall=time.time();first_mtime=None;last_mtime=None;last_acc=0;train_N=len(train_u8)
    hit_epoch=None
    for epoch in range(1,epochs+1):
        model.train()
        perm=torch.randperm(train_N,device=device)
        for i in range(0,train_N,BATCH):
            idx=perm[i:i+BATCH]
            if len(idx)<BATCH: continue
            x_u8=train_u8_gpu[idx];y=train_lbl_gpu[idx]
            x=x_u8.float().div(255.0);x=gpu_augment(x);x=(x-MEAN)/STD;x=x.to(memory_format=torch.channels_last)
            use_mixup=random.random()<0.5
            if use_mixup:
                lam=np.random.beta(MIXUP,MIXUP);perm2=torch.randperm(x.size(0),device=device);x2=x[perm2];y2=y[perm2];x=lam*x+(1-lam)*x2;y_a,y_b=y,y2
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out=model(x)
                if use_mixup: loss=lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)
                else: loss=criterion(out,y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();scheduler.step()
        model.eval();correct=0;total=0
        with torch.no_grad():
            for i in range(0,len(test_norm),512):
                x=test_norm[i:i+512].to(memory_format=torch.channels_last);y=test_lbl_gpu[i:i+512]
                with torch.amp.autocast('cuda'): out=model(x)
                pred=out.argmax(1);correct+=(pred==y).sum().item();total+=y.size(0)
        acc=correct/total;last_acc=acc
        print(f"epoch {epoch:02d} acc {acc*100:.2f}%")
        if hit_epoch is None and acc>=0.94:
            hit_epoch=epoch
        script_model=WideResNet(depth,widen,0.0).to(device);script_model.load_state_dict(model.state_dict());script_model.eval();scripted=torch.jit.script(script_model)
        ckpt_path=os.path.join(ckpt_dir,f"ckpt_epoch_{epoch:04d}.pt");scripted.save(ckpt_path)
        if first_mtime is None: first_mtime=os.path.getmtime(ckpt_path)
        last_mtime=os.path.getmtime(ckpt_path)
    wall=time.time()-start_wall
    span=last_mtime-first_mtime if first_mtime else wall
    t_star = span * hit_epoch/(hit_epoch-1) if hit_epoch and hit_epoch>1 else None
    print(f"DONE depth{depth} widen{widen} epochs{epochs} lr{lr_max} wall {wall:.1f}s span {span:.1f}s hit {hit_epoch} t_star {t_star} acc {last_acc:.4f}")
    return hit_epoch, wall, span, t_star, last_acc

# sweep
configs=[
 (16,8,35,1.0),
 (16,10,35,1.0),
 (16,8,30,1.2),
 (22,8,35,1.0),
]
for depth,widen,epochs,lr in configs:
    ckpt=f"/workspace/scratch_ckpt_{depth}_{widen}_{epochs}"
    hit,wall,span,t_star,acc = run(depth,widen,epochs,lr,ckpt)
    print(f"RESULT {depth}-{widen} e{epochs} lr{lr} hit{hit} t_star{t_star} acc{acc}")

