import os, time, shutil, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/scratch_ckpt"
FINAL_CKPT = "/workspace/scratch_ckpt_final.pt"
EPOCHS = 30
BATCH = 1024
LR = 0.2
WD = 5e-4

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(x)
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + shortcut

class WRN(nn.Module):
    def __init__(self, depth=28, widen=10):
        super().__init__()
        n = (depth - 4)//6
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        def make(in_ch, out_ch, n, stride):
            layers = [BasicBlock(in_ch, out_ch, stride)]
            for _ in range(1, n):
                layers.append(BasicBlock(out_ch, out_ch, 1))
            return nn.Sequential(*layers)
        c1, c2, c3 = 16*widen, 32*widen, 64*widen
        self.s1 = make(16, c1, n, 1)
        self.s2 = make(c1, c2, n, 2)
        self.s3 = make(c2, c3, n, 2)
        self.bn = nn.BatchNorm2d(c3)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(c3, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.init_conv(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02,0.33), ratio=(0.3,3.3)),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder(DATA_TRAIN, transform=train_tf)
test_ds = datasets.ImageFolder(DATA_TEST, transform=test_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train {len(train_ds)} Test {len(test_ds)}")

model = WRN(28,10).to(device).to(memory_format=torch.channels_last)
print(f"Params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler('cuda')

def rand_bbox(size, lam):
    W = size[-1]; H = size[-2]
    cut_rat = (1 - lam) ** 0.5
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = random.randint(0, W-1); cy = random.randint(0, H-1)
    x1 = max(cx - cut_w//2, 0); y1 = max(cy - cut_h//2, 0)
    x2 = min(cx + cut_w//2, W); y2 = min(cy + cut_h//2, H)
    return x1, y1, x2, y2

if os.path.exists(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
os.makedirs(CKPT_DIR, exist_ok=True)

start_wall = time.time()
first_ckpt = None
last_ckpt = None

for epoch in range(1, EPOCHS+1):
    model.train()
    if epoch <= 5:
        lr = LR * epoch / 5
    else:
        prog = (epoch-5)/(EPOCHS-5)
        lr = 1e-4 + 0.5*(LR-1e-4)*(1+torch.cos(torch.tensor(prog*3.14159265)).item())
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    for x, y in train_loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        r = random.random()
        lam = 1.0
        y_a = y_b = y
        if r < 0.5:
            lam0 = random.betavariate(1,1)
            x1,y1,x2,y2 = rand_bbox(x.size(), lam0)
            idx = torch.randperm(x.size(0), device=device)
            x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
            lam = 1 - ((x2-x1)*(y2-y1)/(x.size(-1)*x.size(-2)))
            y_a, y_b = y, y[idx]
        elif r < 0.7:
            lam = random.betavariate(0.2,0.2)
            idx = torch.randperm(x.size(0), device=device)
            x = lam*x + (1-lam)*x[idx]
            y_a, y_b = y, y[idx]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            if lam != 1.0:
                loss = lam*criterion(out, y_a) + (1-lam)*criterion(out, y_b)
            else:
                loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    elapsed=time.time()-start_wall
    print(f"Epoch {epoch:02d} acc={acc*100:.2f}% lr={lr:.4f} time={elapsed:.1f}s")

    ckpt_path=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    m_cpu=model.to("cpu")
    m_cpu.eval()
    torch.jit.script(m_cpu).save(ckpt_path)
    model.to(device).to(memory_format=torch.channels_last)
    if first_ckpt is None:
        first_ckpt=time.time()
    last_ckpt=time.time()

    if acc>=0.94:
        print(f"Hit 94 at epoch {epoch}")
        break

ckpts=sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_epoch_")])
if ckpts:
    shutil.copy(os.path.join(CKPT_DIR, ckpts[-1]), FINAL_CKPT)
    print(f"Copied {ckpts[-1]}")

total_wall = last_ckpt - start_wall if last_ckpt else time.time()-start_wall
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
print(f"Done acc {acc:.4f} wall {total_wall:.1f}s")
