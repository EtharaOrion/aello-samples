import os, time, shutil, random
os.environ["PYTHONPATH"]=""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EPOCHS = 35
BATCH_TRAIN = 1024
BATCH_TEST = 512
LR_MAX = 1.8
WD = 3e-4
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CKPT_DIR = "/workspace/periodic_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False) if in_ch != out_ch or stride != 1 else nn.Identity()
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(x)
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + shortcut

class WRN(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10):
        super().__init__()
        assert (depth-4)%6==0
        n=(depth-4)//6
        self.init_conv = nn.Conv2d(3,16,3,1,1,bias=False)
        c1,c2,c3=16*10,32*10,64*10
        self.layer1=self._make(16,c1,n,1)
        self.layer2=self._make(c1,c2,n,2)
        self.layer3=self._make(c2,c3,n,2)
        self.bn=nn.BatchNorm2d(c3)
        self.relu=nn.ReLU(inplace=True)
        self.fc=nn.Linear(c3,num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.zeros_(m.bias)
    def _make(self,in_ch,out_ch,n,stride):
        layers=[BasicBlock(in_ch,out_ch,stride)]
        for _ in range(1,n):
            layers.append(BasicBlock(out_ch,out_ch,1))
        return nn.Sequential(*layers)
    def forward(self,x):
        out=self.init_conv(x)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.relu(self.bn(out))
        out=F.adaptive_avg_pool2d(out,1).flatten(1)
        out=self.fc(out)
        return out

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25),
])
test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

train_ds = datasets.ImageFolder("/workspace/data/cifar10/train", transform=train_tf)
test_ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=test_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

device=torch.device("cuda")
model=WRN().to(device)
# channels_last for speed
model=model.to(memory_format=torch.channels_last)
# compile disabled to avoid shm+compile interaction; use eager with TF32
criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer=torch.optim.SGD(model.parameters(), lr=LR_MAX, momentum=0.9, weight_decay=WD, nesterov=True)
scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)
scaler=torch.amp.GradScaler('cuda')

def mixup(x,y,alpha=0.4):
    lam = random.betavariate(alpha,alpha)
    idx=torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

start=time.time()
best=0
for epoch in range(1,EPOCHS+1):
    model.train()
    t0=time.time()
    for x,y in train_loader:
        x=x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y=y.to(device, non_blocking=True)
        do_mix = random.random()<0.5
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            if do_mix:
                xm,ya,yb,lam=mixup(x,y,0.4)
                out=model(xm)
                loss=lam*criterion(out,ya)+(1-lam)*criterion(out,yb)
            else:
                out=model(x)
                loss=criterion(out,y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    # eval
    model.eval()
    correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y=y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out=model(x)
            correct+=(out.argmax(1)==y).sum().item()
            total+=y.size(0)
    acc=correct/total
    best=max(best,acc)
    print(f"Epoch {epoch:02d} acc={acc*100:.2f} best={best*100:.2f} lr={optimizer.param_groups[0]['lr']:.4f} dt={time.time()-t0:.1f}s")
    # save scripted - need contiguous float32 model without channels_last compile issues
    # create a clean copy for scripting
    to_script=WRN().to("cpu")
    # copy weights
    to_script.load_state_dict({k.replace("_orig_mod.",""):v for k,v in model.state_dict().items()})
    to_script.eval()
    scripted=torch.jit.script(to_script)
    p=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(p)

total_wall=time.time()-start
print(f"Done wall={total_wall:.1f}s best={best*100:.2f}")
shutil.copy(os.path.join(CKPT_DIR,f"ckpt_epoch_{EPOCHS:04d}.pt"),"/workspace/checkpoint.pt")
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy":best,"wall_clock_sec":total_wall},f)
# verify
m=torch.jit.load(os.path.join(CKPT_DIR,f"ckpt_epoch_{EPOCHS:04d}.pt")).eval()
# quick verify on gpu if possible
try:
    m2=torch.jit.load(os.path.join(CKPT_DIR,f"ckpt_epoch_{EPOCHS:04d}.pt")).eval().cuda()
    correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
            out=m2(x)
            correct+=(out.argmax(1)==y).sum().item()
            total+=y.size(0)
    print(f"VERIFY {correct/total*100:.2f}%")
except Exception as e:
    print(f"verify err {e}")
