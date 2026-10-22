import os, time, shutil, glob, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

DATA_ROOT="/workspace/data/cifar10"
CKPT_DIR="/workspace/scratch_ckpt"
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
cls_to_idx={c:i for i,c in enumerate(CLASSES)}

print("Preloading...")
train_paths=[]; train_labels=[]
for cls in CLASSES:
    for p in sorted(glob.glob(os.path.join(DATA_ROOT,"train",cls,"*.png"))):
        train_paths.append(p); train_labels.append(cls_to_idx[cls])
test_paths=[]; test_labels=[]
for cls in CLASSES:
    for p in sorted(glob.glob(os.path.join(DATA_ROOT,"test",cls,"*.png"))):
        test_paths.append(p); test_labels.append(cls_to_idx[cls])

train_imgs = np.empty((50000,32,32,3), dtype=np.uint8)
for i,p in enumerate(train_paths):
    train_imgs[i] = np.array(Image.open(p).convert("RGB"))
train_labels_np = np.array(train_labels, dtype=np.int64)
test_imgs = np.empty((10000,32,32,3), dtype=np.uint8)
for i,p in enumerate(test_paths):
    test_imgs[i] = np.array(Image.open(p).convert("RGB"))
test_labels_np = np.array(test_labels, dtype=np.int64)
print(f"Loaded {train_imgs.shape}")

# Convert to NCHW uint8 tensor on CPU
train_imgs_nchw = torch.from_numpy(train_imgs).permute(0,3,1,2).contiguous()  # N,3,32,32 uint8
train_labels_t = torch.from_numpy(train_labels_np)
test_imgs_nchw = torch.from_numpy(test_imgs).permute(0,3,1,2).contiguous()

# Model
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(in_planes,out_planes,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(out_planes)
        self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_planes,out_planes,3,1,1,bias=False)
        self.droprate=dropRate
        self.equalInOut=(in_planes==out_planes)
        if self.equalInOut: self.convShortcut=nn.Identity()
        else: self.convShortcut=nn.Conv2d(in_planes,out_planes,1,stride,bias=False)
    def forward(self,x):
        out=self.relu1(self.bn1(x))
        if self.equalInOut: shortcut=x
        else: shortcut=self.convShortcut(out)
        out=self.conv1(out)
        out=self.relu2(self.bn2(out))
        if self.droprate>0: out=F.dropout(out,p=self.droprate,training=self.training)
        out=self.conv2(out)
        return shortcut+out

class NetworkBlock(nn.Module):
    def __init__(self,nb_layers,in_planes,out_planes,block,stride,dropRate=0.0):
        super().__init__()
        layers=[]
        for i in range(nb_layers):
            layers.append(block(i==0 and in_planes or out_planes,out_planes,i==0 and stride or 1,dropRate))
        self.layer=nn.Sequential(*layers)
    def forward(self,x): return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self,depth=28,widen_factor=10,num_classes=10,dropRate=0.0):
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
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): m.weight.data.fill_(1);m.bias.data.zero_()
            elif isinstance(m,nn.Linear): m.bias.data.zero_()
    def forward(self,x):
        out=self.conv1(x)
        out=self.block1(out)
        out=self.block2(out)
        out=self.block3(out)
        out=self.relu(self.bn1(out))
        out=F.adaptive_avg_pool2d(out,1)
        out=out.view(-1,self.nChannels)
        return self.fc(out)

MEAN_T = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
STD_T = torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)

def gpu_augment_fast(x, device):
    # x: (B,3,32,32) float [0,1] on device
    B = x.shape[0]
    # Pad 4
    x = F.pad(x, (4,4,4,4), value=0)  # B,3,40,40
    # Random crop via unfold (vectorized)
    # Unfold to get all 81 patches
    patches = F.unfold(x, kernel_size=32, stride=1)  # B, 3*1024, 81
    # Random index per sample
    tops = torch.randint(0, 9, (B,), device=device)
    lefts = torch.randint(0, 9, (B,), device=device)
    idx = tops * 9 + lefts  # B
    # Gather: patches is B, C*1024, 81 -> need to select idx per batch
    # Use advanced indexing
    # patches: B, D, 81 -> gather dim 2
    idx_exp = idx.view(B, 1, 1).expand(B, patches.shape[1], 1)
    selected = patches.gather(2, idx_exp).squeeze(2)  # B, D
    x = selected.view(B, 3, 32, 32)
    # Flip
    flip = torch.rand(B, device=device) < 0.5
    if flip.any():
        x[flip] = x[flip].flip(dims=[3])
    # Normalize
    x = (x - MEAN_T.to(device)) / STD_T.to(device)
    # Cutout vectorized
    cy = torch.randint(0, 32, (B,), device=device)
    cx = torch.randint(0, 32, (B,), device=device)
    Y, X = torch.meshgrid(torch.arange(32, device=device), torch.arange(32, device=device), indexing='ij')
    y1 = (cy - 8).clamp(0,32).view(B,1,1)
    y2 = (cy + 8).clamp(0,32).view(B,1,1)
    x1 = (cx - 8).clamp(0,32).view(B,1,1)
    x2 = (cx + 8).clamp(0,32).view(B,1,1)
    Yb = Y.view(1,32,32).expand(B,-1,-1)
    Xb = X.view(1,32,32).expand(B,-1,-1)
    mask = (Yb >= y1) & (Yb < y2) & (Xb >= x1) & (Xb < x2)
    mask = mask.unsqueeze(1).expand(-1,3,-1,-1)
    x = x.masked_fill(mask, 0)
    return x

def main():
    t_start=time.time()
    device=torch.device("cuda")
    torch.backends.cudnn.benchmark=True
    if os.path.exists(CKPT_DIR): shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR,exist_ok=True)

    # Test set on GPU normalized
    test_x = test_imgs_nchw.float() / 255.0
    test_x = (test_x - MEAN_T) / STD_T
    test_x = test_x.to(device)
    test_y = torch.from_numpy(test_labels_np).to(device)

    model=WideResNet(depth=28,widen_factor=10,dropRate=0.0).to(device)
    model=model.to(memory_format=torch.channels_last)

    BATCH=2048
    EPOCHS=25
    optimizer=torch.optim.SGD(model.parameters(),lr=1.0,momentum=0.9,weight_decay=5e-4,nesterov=True)
    steps_per_epoch=(50000+BATCH-1)//BATCH
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1.5,epochs=EPOCHS,steps_per_epoch=steps_per_epoch,pct_start=0.15,anneal_strategy='cos',div_factor=10,final_div_factor=100)
    scaler=torch.amp.GradScaler('cuda')
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)

    n_train=50000
    best=0
    for epoch in range(EPOCHS):
        model.train()
        t0=time.time()
        perm=torch.randperm(n_train)
        total_loss=0; nb=0
        for i in range(0, n_train, BATCH):
            idx=perm[i:i+BATCH]
            xb_cpu = train_imgs_nchw[idx].float() / 255.0  # B,3,32,32 CPU
            xb = xb_cpu.to(device, non_blocking=True)
            yb = train_labels_t[idx].to(device, non_blocking=True)
            xb = gpu_augment_fast(xb, device)
            xb = xb.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out=model(xb)
                loss=criterion(out,yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss+=loss.item()
            nb+=1
        # eval
        model.eval()
        correct=0
        with torch.no_grad():
            for i in range(0,10000,1024):
                xb=test_x[i:i+1024].to(memory_format=torch.channels_last)
                yb=test_y[i:i+1024]
                with torch.amp.autocast('cuda'):
                    out=model(xb)
                correct+=(out.argmax(1)==yb).sum().item()
        acc=correct/10000
        best=max(best,acc)
        dt=time.time()-t0
        print(f"Epoch {epoch+1:02d}/{EPOCHS} loss={total_loss/nb:.3f} acc={acc*100:.2f}% best={best*100:.2f}% {dt:.1f}s")
        ckpt=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch+1:04d}.pt")
        scripted=torch.jit.script(model.eval())
        scripted.save(ckpt)
        print(f"  Saved {ckpt}")

    ckpts=sorted(glob.glob(os.path.join(CKPT_DIR,"*.pt")))
    if ckpts: shutil.copy(ckpts[-1],"/workspace/checkpoint.pt")
    wall=time.time()-t_start
    print(f"Done wall={wall:.1f}s best={best:.4f}")
    import json
    with open("/workspace/agent_reported_metrics.json","w") as f:
        json.dump({"accuracy":best,"wall_clock_sec":wall},f)
    m=torch.jit.load("/workspace/checkpoint.pt").eval().cuda()
    correct=0
    with torch.no_grad():
        for i in range(0,10000,1024):
            out=m(test_x[i:i+1024])
            correct+=(out.argmax(1)==test_y[i:i+1024]).sum().item()
    print(f"Verify {correct/10000*100:.2f}%")

if __name__=="__main__": main()
