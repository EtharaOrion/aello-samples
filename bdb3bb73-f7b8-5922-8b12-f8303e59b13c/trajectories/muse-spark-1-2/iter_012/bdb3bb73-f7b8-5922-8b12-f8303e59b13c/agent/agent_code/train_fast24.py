import os, time, shutil, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

CKPT_DIR="/workspace/scratch_ckpt"
EPOCHS=24
BATCH=2048
LR=1.5
WD=5e-4

for f in os.listdir(CKPT_DIR):
    p=os.path.join(CKPT_DIR,f)
    if os.path.isfile(p): os.remove(p)

torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
device=torch.device("cuda")
print(f"torch {torch.__version__}")

train_ds = datasets.ImageFolder("/workspace/data/cifar10/train", transform=transforms.ToTensor())
test_ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=transforms.ToTensor())
train_images = torch.stack([img for img,_ in train_ds])
train_labels = torch.tensor([lbl for _,lbl in train_ds], dtype=torch.long)
test_images = torch.stack([img for img,_ in test_ds])
test_labels = torch.tensor([lbl for _,lbl in test_ds], dtype=torch.long)
print(f"train {train_images.shape}")

mean = torch.tensor([0.4914,0.4822,0.4465], device=device).view(1,3,1,1)
std = torch.tensor([0.2470,0.2435,0.2616], device=device).view(1,3,1,1)
train_images_gpu = train_images.to(device)
train_labels_gpu = train_labels.to(device)
test_images_gpu = (test_images.to(device) - mean) / std
test_labels_gpu = test_labels.to(device)

class BlockEqual(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.bn1=nn.BatchNorm2d(planes)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
    def forward(self, x):
        out=self.relu1(self.bn1(x))
        out=self.relu2(self.bn2(self.conv1(out)))
        out=self.conv2(out)
        return x+out

class BlockDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2=nn.BatchNorm2d(out_planes)
        self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.shortcut=nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False)
    def forward(self, x):
        x1=self.relu1(self.bn1(x))
        out=self.relu2(self.bn2(self.conv1(x1)))
        out=self.conv2(out)
        return self.shortcut(x1)+out

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10):
        super().__init__()
        nChannels=[16,16*widen,32*widen,64*widen]
        assert (depth-4)%6==0
        n=(depth-4)//6
        self.conv1=nn.Conv2d(3,nChannels[0],3,1,1,bias=False)
        layers1=[]
        for i in range(n):
            if i==0: layers1.append(BlockDown(nChannels[0], nChannels[1], 1))
            else: layers1.append(BlockEqual(nChannels[1]))
        self.block1=nn.Sequential(*layers1)
        layers2=[]
        for i in range(n):
            if i==0: layers2.append(BlockDown(nChannels[1], nChannels[2], 2))
            else: layers2.append(BlockEqual(nChannels[2]))
        self.block2=nn.Sequential(*layers2)
        layers3=[]
        for i in range(n):
            if i==0: layers3.append(BlockDown(nChannels[2], nChannels[3], 2))
            else: layers3.append(BlockEqual(nChannels[3]))
        self.block3=nn.Sequential(*layers3)
        self.bn1=nn.BatchNorm2d(nChannels[3])
        self.relu=nn.ReLU(inplace=True)
        self.fc=nn.Linear(nChannels[3], num_classes)
        self.nChannels=nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out=self.conv1(x)
        out=self.block1(out)
        out=self.block2(out)
        out=self.block3(out)
        out=self.relu(self.bn1(out))
        out=F.adaptive_avg_pool2d(out,1)
        out=out.view(-1,self.nChannels)
        return self.fc(out)

model=WideResNet(depth=28,widen=10).to(device)
model=model.to(memory_format=torch.channels_last)
print(f"Params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
torch.jit.script(model)
print("script ok")

criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer=torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler=torch.amp.GradScaler('cuda')

def gpu_augment(x):
    B=x.size(0)
    x_pad = F.pad(x, (4,4,4,4), mode='reflect')
    patches = F.unfold(x_pad, kernel_size=32, stride=1)
    idx = torch.randint(0,81,(B,), device=x.device)
    idx_exp = idx.view(B,1,1).expand(B,3072,1)
    cropped = torch.gather(patches, 2, idx_exp).view(B,3,32,32)
    flip = torch.rand(B, device=x.device) < 0.5
    if flip.any():
        flipped = torch.flip(cropped, dims=[3])
        mask = flip.view(B,1,1,1)
        cropped = torch.where(mask, flipped, cropped)
    cropped = (cropped - mean) / std
    return cropped

best_acc=0
start_time=time.time()
last_ckpt=None
num_train=train_images_gpu.size(0)

for epoch in range(1, EPOCHS+1):
    model.train()
    if epoch <=5:
        lr=LR*epoch/5
        for pg in optimizer.param_groups:
            pg['lr']=lr
    else:
        if epoch==6:
            for pg in optimizer.param_groups:
                pg['lr']=LR
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-5)
        else:
            scheduler.step()

    perm=torch.randperm(num_train, device=device)
    running=0
    batches=0
    for i in range(0, num_train, BATCH):
        idx=perm[i:i+BATCH]
        x=train_images_gpu[idx]
        y=train_labels_gpu[idx]
        x=gpu_augment(x)
        x=x.to(memory_format=torch.channels_last)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out=model(x)
            loss=criterion(out, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running+=loss.item()
        batches+=1

    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, test_images_gpu.size(0), BATCH):
                x=test_images_gpu[i:i+BATCH].to(memory_format=torch.channels_last)
                y=test_labels_gpu[i:i+BATCH]
                out=model(x)
                _, pred=out.max(1)
                total+=y.size(0)
                correct+=pred.eq(y).sum().item()
    acc=correct/total
    best_acc=max(best_acc, acc)
    print(f"Epoch {epoch}/{EPOCHS} loss {running/batches:.3f} acc {acc*100:.2f} best {best_acc*100:.2f} lr {optimizer.param_groups[0]['lr']:.4f} time {time.time()-start_time:.1f}s")

    model.eval()
    scripted=torch.jit.script(model)
    ckpt_path=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    model.train()
    last_ckpt=time.time()

total_wall=last_ckpt-start_time
print(f"Done best {best_acc*100:.2f} wall {total_wall:.1f}s")
with open("/workspace/scratch_metrics.json","w") as f:
    json.dump({"accuracy": best_acc, "wall_clock_sec": total_wall}, f)
