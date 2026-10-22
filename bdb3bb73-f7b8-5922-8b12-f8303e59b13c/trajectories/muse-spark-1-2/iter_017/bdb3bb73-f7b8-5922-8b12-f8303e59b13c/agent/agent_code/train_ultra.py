import os, time, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/scratch_ckpt"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
BATCH = 1024
EPOCHS = 30
LR = 1.0
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

class PreActBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.has_sc = stride != 1 or in_ch != out_ch
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False) if self.has_sc else nn.Identity()
    def forward(self, x):
        out = F.relu(self.bn1(x))
        sc = self.shortcut(out) if self.has_sc else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += sc
        return out

class CifarResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make(64, 64, 2, 1)
        self.layer2 = self._make(64, 128, 2, 2)
        self.layer3 = self._make(128, 256, 2, 2)
        self.layer4 = self._make(256, 512, 2, 2)
        self.bn = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self._init()
    def _make(self, in_ch, out_ch, n, stride):
        layers = [PreActBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            layers.append(PreActBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_loaders():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
    test_ds = datasets.ImageFolder(DATA_TEST, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader

def evaluate(model, loader, device):
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y=y.to(device, non_blocking=True)
            out=model(x)
            pred=out.argmax(1)
            correct+=(pred==y).sum().item()
            total+=y.size(0)
    return correct/total

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        os.remove(os.path.join(CKPT_DIR,f))
    device=torch.device("cuda")
    train_loader, test_loader = get_loaders()
    print(f"Steps {len(train_loader)} batch {BATCH}")
    model = CifarResNet().to(device).to(memory_format=torch.channels_last)
    # no compile for now to avoid autotune overhead
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
    warmup=2
    def lr_lambda(epoch):
        e=epoch+1
        if e<=warmup:
            return e/warmup
        prog=(e-warmup)/(EPOCHS-warmup)
        return 0.5*(1+np.cos(prog*np.pi))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler=torch.amp.GradScaler('cuda')
    start_wall=time.time()
    first=None; last=None; best=0; crossing=None
    for epoch in range(1, EPOCHS+1):
        model.train()
        t0=time.time()
        for x,y in train_loader:
            x=x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y=y.to(device, non_blocking=True)
            if random.random()<0.5:
                lam=np.random.beta(1.0,1.0)
                idx=torch.randperm(x.size(0), device=device)
                y_a,y_b=y,y[idx]
                W=32;H=32
                cut_rat=np.sqrt(1.-lam)
                cut_w=int(W*cut_rat); cut_h=int(H*cut_rat)
                cx=np.random.randint(W); cy=np.random.randint(H)
                bbx1=np.clip(cx-cut_w//2,0,W); bby1=np.clip(cy-cut_h//2,0,H)
                bbx2=np.clip(cx+cut_w//2,0,W); bby2=np.clip(cy+cut_h//2,0,H)
                if bbx2>bbx1 and bby2>bby1:
                    x[:,:,bby1:bby2,bbx1:bbx2]=x[idx,:,bby1:bby2,bbx1:bbx2]
                    lam=1-((bbx2-bbx1)*(bby2-bby1)/(W*H))
                else:
                    lam=1.0
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out=model(x)
                    loss=lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out=model(x)
                    loss=criterion(out,y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
        scheduler.step()
        cpu_model=CifarResNet()
        cpu_model.load_state_dict(model.state_dict())
        cpu_model.eval()
        scripted=torch.jit.script(cpu_model)
        ckpt=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(ckpt)
        if first is None: first=time.time()
        last=time.time()
        acc=evaluate(model,test_loader,device)
        if acc>best: best=acc
        if crossing is None and acc>=0.94: crossing=epoch
        print(f"Epoch {epoch:02d}/{EPOCHS} acc {acc*100:.2f} best {best*100:.2f} lr {optimizer.param_groups[0]['lr']:.4f} time {time.time()-t0:.2f}s crossing {crossing}")
        if epoch>=20 and best>=0.94 and crossing is not None and epoch>=crossing+2:
            print("Early stop")
            break
    total=last-start_wall
    span=last-first if first else total
    print(f"Done best {best*100:.2f} crossing {crossing} total {total:.1f} span {span:.1f}")
    with open("/tmp/metrics.json","w") as f:
        json.dump({"accuracy":best,"wall_clock_sec":total,"crossing":crossing},f)

if __name__=="__main__":
    main()
