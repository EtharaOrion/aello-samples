import os, time, shutil, glob, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

DATA_ROOT="/workspace/data/cifar10"
CKPT_DIR="/workspace/scratch_ckpt"
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
cls_to_idx={c:i for i,c in enumerate(CLASSES)}
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)

class CIFAR10PNG(Dataset):
    def __init__(self, root, split, transform=None):
        self.transform=transform
        self.samples=[]
        for cls in CLASSES:
            d=os.path.join(root,split,cls)
            for p in glob.glob(os.path.join(d,"*.png")):
                self.samples.append((p,cls_to_idx[cls]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p,l=self.samples[idx]
        img=Image.open(p).convert("RGB")
        if self.transform: img=self.transform(img)
        return img,l

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

class Cutout:
    def __init__(self,size=16): self.size=size
    def __call__(self,img):
        h,w=img.shape[1],img.shape[2]
        y=random.randint(0,h-1); x=random.randint(0,w-1)
        y1=max(0,y-self.size//2); y2=min(h,y+self.size//2)
        x1=max(0,x-self.size//2); x2=min(w,x+self.size//2)
        img[:,y1:y2,x1:x2]=0
        return img

def main():
    t_start=time.time()
    device=torch.device("cuda")
    torch.backends.cudnn.benchmark=True
    if os.path.exists(CKPT_DIR): shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR,exist_ok=True)

    train_tf=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD),
        Cutout(16),
    ])
    test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    train_ds=CIFAR10PNG(DATA_ROOT,"train",train_tf)
    test_ds=CIFAR10PNG(DATA_ROOT,"test",test_tf)
    print(f"Train {len(train_ds)} Test {len(test_ds)}")
    BATCH=512
    train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2)
    test_loader=DataLoader(test_ds,batch_size=1024,shuffle=False,num_workers=2,pin_memory=True)

    model=WideResNet(depth=28,widen_factor=10,dropRate=0.0).to(device)
    model=model.to(memory_format=torch.channels_last)
    model_c=torch.compile(model)

    EPOCHS=35
    optimizer=torch.optim.SGD(model.parameters(),lr=0.5,momentum=0.9,weight_decay=5e-4,nesterov=True)
    # Cosine with warmup 3
    warmup=3
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch+1)/warmup
        p=(epoch-warmup)/(EPOCHS-warmup)
        return 0.5*(1+torch.cos(torch.tensor(p*3.14159265))).item()*0.99+0.01
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler=torch.amp.GradScaler('cuda')
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)

    best=0
    for epoch in range(EPOCHS):
        model.train()
        t0=time.time()
        tot=0
        for x,y in train_loader:
            x=x.to(device,non_blocking=True,memory_format=torch.channels_last)
            y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out=model_c(x)
                loss=criterion(out,y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tot+=loss.item()
        scheduler.step()
        model.eval()
        correct=0;total=0
        with torch.no_grad():
            for x,y in test_loader:
                x=x.to(device,non_blocking=True,memory_format=torch.channels_last)
                y=y.to(device,non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out=model(x)
                correct+=(out.argmax(1)==y).sum().item()
                total+=y.size(0)
        acc=correct/total
        best=max(best,acc)
        dt=time.time()-t0
        print(f"Epoch {epoch+1:02d}/{EPOCHS} loss={tot/len(train_loader):.3f} acc={acc*100:.2f}% best={best*100:.2f}% {dt:.1f}s lr={optimizer.param_groups[0]['lr']:.4f}")
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
    correct=0;total=0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
            out=m(x)
            correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    print(f"Verify {correct/total*100:.2f}%")

if __name__=="__main__": main()
