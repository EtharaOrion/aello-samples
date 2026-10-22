import os, time, shutil, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 50
BATCH_TRAIN = 512
BATCH_TEST = 1024
WD = 5e-4
LR_MAX = 1.2

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Residual(128),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Residual(256),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Residual(512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)
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
        x = F.relu(self.bn0(self.conv0(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        fp=os.path.join(CKPT_DIR,f)
        if os.path.isfile(fp):
            os.remove(fp)
    if os.path.exists(FINAL_CKPT):
        os.remove(FINAL_CKPT)
    device=torch.device("cuda")
    print(f"Device {torch.cuda.get_device_name(0)}")

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
    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False)

    model = Net().to(device).to(memory_format=torch.channels_last)
    # try torch.compile for speed
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception as e:
        print(f"compile failed {e}")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR_MAX, momentum=0.9, weight_decay=WD, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_wall=time.time()
    for epoch in range(1, EPOCHS+1):
        model.train()
        t0=time.time()
        for x,y in train_loader:
            x=x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y=y.to(device, non_blocking=True)
            r=random.random()
            if r < 0.3:
                lam = random.betavariate(1.0,1.0)
                idx=torch.randperm(x.size(0), device=device)
                W,H=x.size(3),x.size(2)
                cut_rat=(1-lam)**0.5
                cut_w=int(W*cut_rat); cut_h=int(H*cut_rat)
                cx=random.randint(0,W-1); cy=random.randint(0,H-1)
                bbx1=max(cx-cut_w//2,0); bby1=max(cy-cut_h//2,0)
                bbx2=min(cx+cut_w//2,W); bby2=min(cy+cut_h//2,H)
                lam=1-((bbx2-bbx1)*(bby2-bby1)/(W*H))
                x[:, :, bby1:bby2, bbx1:bbx2]=x[idx, :, bby1:bby2, bbx1:bbx2]
                y_a, y_b = y, y[idx]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out=model(x)
                    loss=lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            elif r < 0.4:
                lam=random.betavariate(0.2,0.2)
                idx=torch.randperm(x.size(0), device=device)
                mixed_x=lam*x+(1-lam)*x[idx]
                y_a, y_b = y, y[idx]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out=model(mixed_x)
                    loss=lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out=model(x)
                    loss=criterion(out,y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        model.eval()
        correct=0; total=0
        with torch.no_grad():
            for x,y in test_loader:
                x=x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y=y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out=model(x)
                correct+=(out.argmax(1)==y).sum().item()
                total+=y.size(0)
        acc=correct/total
        print(f"Epoch {epoch:03d} acc={acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.5f} time={time.time()-t0:.2f}s")
        # save scripted - need to handle compiled model: unwrap
        to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        # ensure eval
        to_save.eval()
        scripted=torch.jit.script(to_save)
        ckpt_path=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(ckpt_path)
        # verify
        try:
            m2=torch.jit.load(ckpt_path, map_location=device)
            m2.eval()
            with torch.no_grad():
                xb,_=next(iter(test_loader))
                xb=xb.to(device).to(memory_format=torch.channels_last)
                _=m2(xb[:2])
        except Exception as e:
            print(f"verify fail {e}")
        # if compiled, need to keep model as compiled for training
        # re-wrap? model is already compiled, to_save is orig, but model still compiled
        # ensure model.train() for next epoch
        model.train()

    last=os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
    if os.path.exists(last):
        shutil.copy(last, FINAL_CKPT)
    total_wall=time.time()-start_wall
    # final acc
    to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    to_save.eval()
    correct=0; total=0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y=y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out=to_save(x)
            correct+=(out.argmax(1)==y).sum().item()
            total+=y.size(0)
    acc=correct/total
    print(f"Final acc {acc*100:.2f}% wall {total_wall:.2f}s")
    with open("/workspace/agent_reported_metrics.json","w") as f:
        json.dump({"accuracy":acc,"wall_clock_sec":total_wall},f)

if __name__=="__main__":
    main()
