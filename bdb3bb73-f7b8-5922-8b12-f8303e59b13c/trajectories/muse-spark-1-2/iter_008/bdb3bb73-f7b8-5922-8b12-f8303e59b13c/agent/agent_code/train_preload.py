import os, time, shutil, random, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
CKPT_DIR="/workspace/scratch_ckpt"
EPOCHS=35
BATCH_TRAIN=1024
BATCH_TEST=512
NUM_WORKERS=0  # preload so 0 workers is fine, or 2
LR=0.6
WD=5e-4
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)
device=torch.device("cuda")

# Preload
print("Preloading train...")
classes=sorted(os.listdir(DATA_TRAIN))
class_to_idx={c:i for i,c in enumerate(classes)}
train_paths=[]
train_labels=[]
for c in classes:
    idx=class_to_idx[c]
    for p in glob.glob(os.path.join(DATA_TRAIN,c,"*.png")):
        train_paths.append(p)
        train_labels.append(idx)
print(f"train {len(train_paths)}")

# Load all images into RAM as uint8 tensors
import numpy as np
from torchvision.transforms.functional import to_tensor

# Use PIL preload to tensor
train_images=[]
for p in train_paths:
    img=Image.open(p).convert("RGB")
    # store as PIL for aug? Let's store as tensor uint8 3x32x32
    t=torch.from_numpy(np.array(img)).permute(2,0,1)  # HWC to CHW
    train_images.append(t)
train_images=torch.stack(train_images)  # N,3,32,32 uint8
train_labels=torch.tensor(train_labels, dtype=torch.long)
print(f"train_images {train_images.shape} {train_images.dtype}")

print("Preloading test...")
test_paths=[]
test_labels=[]
for c in classes:
    idx=class_to_idx[c]
    for p in glob.glob(os.path.join(DATA_TEST,c,"*.png")):
        test_paths.append(p)
        test_labels.append(idx)
test_images=[]
for p in test_paths:
    img=Image.open(p).convert("RGB")
    t=torch.from_numpy(np.array(img)).permute(2,0,1)
    test_images.append(t)
test_images=torch.stack(test_images)
test_labels=torch.tensor(test_labels, dtype=torch.long)
print(f"test_images {test_images.shape}")

# Dataset with aug on the fly (CPU tensor ops)
# Use torchvision transforms that work on tensors
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ConvertImageDtype(torch.float32),  # uint8 -> float [0,1]
    transforms.Normalize(MEAN, STD),
    # RandomErasing after normalize
])

# For tensor input, RandomCrop expects PIL or tensor; with padding_mode reflect it works on tensor
# But AutoAugment expects PIL or tensor? In torchvision 0.19 it supports tensor
# We'll test

test_tf = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(MEAN, STD),
])

class PreloadDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images=images
        self.labels=labels
        self.transform=transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img=self.images[idx]  # uint8 3x32x32
        # transform expects tensor image; need to ensure dtype uint8 for aug then convert
        if self.transform:
            img=self.transform(img)
        return img, self.labels[idx]

train_ds=PreloadDataset(train_images, train_labels, transform=train_tf)
test_ds=PreloadDataset(test_images, test_labels, transform=test_tf)

# Need to handle RandomErasing separately because it expects float tensor after normalize
# Add it manually in dataset or via transform after normalize
# Our train_tf already includes ConvertImageDtype and Normalize, but RandomErasing should be after
# Let's add RandomErasing as last step via transform
# Actually we omitted it; add via extra
# We'll keep as is and add erasing in training loop or via transform

# Recreate with erasing
train_tf2 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.TrivialAugmentWide(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02,0.33), ratio=(0.3,3.3)),
])
train_ds.transform=train_tf2

train_loader=DataLoader(train_ds,batch_size=BATCH_TRAIN,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
test_loader=DataLoader(test_ds,batch_size=BATCH_TEST,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True)

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

model=WRN(depth=28,widen=10).to(device)
model=model.to(memory_format=torch.channels_last)
try:
    model=torch.compile(model)
    print("compiled")
except Exception as e:
    print(f"compile fail {e}")

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
        scaler.step(optimizer)
        scaler.update()
        tot_loss+=loss.item()
    acc=evaluate(model,test_loader)
    print(f"Epoch {epoch:02d} loss {tot_loss/len(train_loader):.3f} acc {acc*100:.2f}% lr {optimizer.param_groups[0]['lr']:.4f} time {time.time()-start:.1f}s")
    to_script=model._orig_mod if hasattr(model,'_orig_mod') else model
    to_script.eval()
    scripted=torch.jit.script(to_script)
    ckpt=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt)
    to_script.train()
    model.train()
    if first is None: first=time.time()
    last=time.time()
    if acc>=0.945 and epoch>=25:
        print("early stop")
        break

ckpts=sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_epoch_")])
print(f"done {len(ckpts)}")
if ckpts:
    last_ckpt=os.path.join(CKPT_DIR,ckpts[-1])
    m_final=torch.jit.load(last_ckpt,map_location=device)
    m_final.eval()
    acc_final=evaluate(m_final,test_loader)
    print(f"final acc {acc_final*100:.2f} wall {last-start:.1f} span {last-first:.1f}")
    with open("/tmp/preload_metrics.json","w") as f:
        json.dump({"accuracy":float(acc_final),"wall_clock_sec":float(last-start),"span":float(last-first),"epochs":len(ckpts)},f)
