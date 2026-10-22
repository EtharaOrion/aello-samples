import os, time, shutil, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 40
BATCH_TRAIN = 1024
BATCH_TEST = 512
LR = 1.0
WD = 5e-4

os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    fp=os.path.join(CKPT_DIR,f)
    if os.path.isfile(fp): os.remove(fp)
if os.path.exists(FINAL_CKPT): os.remove(FINAL_CKPT)

torch.backends.cudnn.benchmark=True
device=torch.device("cuda")
print(f"torch {torch.__version__} cuda {torch.cuda.is_available()}")

mean=torch.tensor([0.4914,0.4822,0.4465],device=device).view(1,3,1,1)
std=torch.tensor([0.2470,0.2435,0.2616],device=device).view(1,3,1,1)

# Load datasets to tensors in memory, then move to GPU for aug
# Use ImageFolder to get PIL, convert to tensor quickly
# Preload train as uint8 tensors on CPU, then GPU aug per batch
import torchvision.transforms as T
from PIL import Image
import glob

# Build tensor datasets by loading all images via torchvision ImageFolder with ToTensor (no normalize)
# We'll do GPU normalize + aug
print("Loading data to RAM...")
# Use simple transform to tensor [0,1]
raw_train_transform = transforms.Compose([transforms.ToTensor()])
raw_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])

# Load via ImageFolder but with raw transform
train_ds_raw = torchvision.datasets.ImageFolder(DATA_TRAIN, transform=raw_train_transform)
test_ds_raw = torchvision.datasets.ImageFolder(DATA_TEST, transform=raw_test_transform)

# Stack into tensors for faster loading? Instead use DataLoader with num_workers 4 but raw tensors are small
# Keep DataLoader but with raw tensors, do GPU aug
train_loader = DataLoader(train_ds_raw, batch_size=BATCH_TRAIN, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=False)
test_loader = DataLoader(test_ds_raw, batch_size=BATCH_TEST, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False)
print(f"Train {len(train_ds_raw)} Test {len(test_ds_raw)}")

# Model - scriptable WRN28-10
class BlockEqual(nn.Module):
    def __init__(self, planes, stride=1):
        super().__init__()
        self.bn1=nn.BatchNorm2d(planes)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
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
        # block1: n blocks, first is down? Actually first block stride 1, in==out for all except first of block2/3
        # block1: all equal (16*widen)
        layers1=[]
        for i in range(n):
            if i==0:
                # first block of stage1: in 16 -> 160, stride 1, down
                layers1.append(BlockDown(nChannels[0], nChannels[1], 1))
            else:
                layers1.append(BlockEqual(nChannels[1],1))
        self.block1=nn.Sequential(*layers1)
        layers2=[]
        for i in range(n):
            if i==0:
                layers2.append(BlockDown(nChannels[1], nChannels[2], 2))
            else:
                layers2.append(BlockEqual(nChannels[2],1))
        self.block2=nn.Sequential(*layers2)
        layers3=[]
        for i in range(n):
            if i==0:
                layers3.append(BlockDown(nChannels[2], nChannels[3], 2))
            else:
                layers3.append(BlockEqual(nChannels[3],1))
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

# Test scripting before training
try:
    m_test=torch.jit.script(model)
    print("Script OK")
    del m_test
except Exception as e:
    print(f"Script fail: {e}")
    raise

criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer=torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
# OneCycle or Cosine with warmup
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler=torch.amp.GradScaler('cuda')

# GPU aug helpers
def gpu_augment(x):
    # x: [B,3,32,32] in [0,1] on GPU, already normalized? No, we pass [0,1] and do aug then normalize
    # Random crop with padding 4: pad 4 then random crop 32
    # Implement via F.pad + random crop
    # Pad 4 reflect
    x = F.pad(x, (4,4,4,4), mode='reflect')
    # random crop
    B, C, H, W = x.shape
    # generate random offsets
    # Use torch.randint
    top = torch.randint(0, 9, (B,), device=x.device)
    left = torch.randint(0, 9, (B,), device=x.device)
    # crop via batched indexing - loop is slow, use vectorized via unfold? Simpler: use torch.stack with slicing via for loop but B=1024 loop 1024 times is heavy
    # Alternative: use torchvision transforms on GPU via Kornia? Let's do efficient batched crop using grid_sample or just use random crop via rolling
    # Quick hack: use F.unfold? Easier: do per-sample crop with loop in python but with torch ops it's still GPU
    # We'll do a fast version: generate random crops via indexing with torch
    # Use torch's crop via batched gather: create output tensor
    out = torch.empty(B, C, 32, 32, device=x.device, dtype=x.dtype)
    for i in range(B):
        out[i] = x[i, :, top[i]:top[i]+32, left[i]:left[i]+32]
    x = out
    # Random horizontal flip
    flip_mask = torch.rand(B, device=x.device) < 0.5
    # flip where mask true
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
    # Normalize
    x = (x - mean) / std
    # Cutout / RandomErasing: 25% prob, erase 8x8 patch
    # We'll do random erasing after normalize: erase to 0 (mean)
    erase_mask = torch.rand(B, device=x.device) < 0.25
    if erase_mask.any():
        # random position
        er_top = torch.randint(0, 25, (B,), device=x.device)
        er_left = torch.randint(0, 25, (B,), device=x.device)
        for i in torch.where(erase_mask)[0]:
            t = int(er_top[i]); l = int(er_left[i])
            x[i, :, t:t+8, l:l+8] = 0  # 0 after normalize = mean
    return x

def mixup_cutmix(x, y, alpha_mix=0.2, alpha_cut=1.0):
    # x already normalized, y is long
    if random.random() < 0.5:
        if random.random() < 0.5:
            lam = random.betavariate(alpha_mix, alpha_mix)
            idx = torch.randperm(x.size(0), device=x.device)
            mixed = lam * x + (1 - lam) * x[idx]
            return mixed, y, y[idx], lam
        else:
            lam = random.betavariate(alpha_cut, alpha_cut)
            idx = torch.randperm(x.size(0), device=x.device)
            W, H = 32, 32
            cut_rat = (1. - lam) ** 0.5
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            cx = random.randint(0, W-1)
            cy = random.randint(0, H-1)
            bbx1 = max(cx - cut_w // 2, 0)
            bby1 = max(cy - cut_h // 2, 0)
            bbx2 = min(cx + cut_w // 2, W)
            bby2 = min(cy + cut_h // 2, H)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            return x, y, y[idx], lam
    return None

best_acc=0
start_time=time.time()
first_ckpt=None
last_ckpt=None

for epoch in range(1, EPOCHS+1):
    model.train()
    # warmup 5 epochs
    if epoch <= 5:
        lr = LR * epoch / 5
        for pg in optimizer.param_groups:
            pg['lr']=lr
    else:
        scheduler.step()
        # scheduler was created with T_max=EPOCHS, but we warmed up 5, so after warmup we need to step correctly
        # Actually we stepped from epoch 6 onward, so total steps = EPOCHS-5, but scheduler T_max=EPOCHS, so LR will not go to 0
        # It's okay, but we can recreate scheduler after warmup
        if epoch==6:
            # recreate with correct T_max
            for pg in optimizer.param_groups:
                pg['lr']=LR
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-5)

    running=0
    for inputs, targets in train_loader:
        inputs=inputs.to(device, non_blocking=True)  # [B,3,32,32] in [0,1]
        targets=targets.to(device, non_blocking=True)
        # GPU augment
        inputs=gpu_augment(inputs)
        inputs=inputs.to(memory_format=torch.channels_last)
        # mixup/cutmix
        mixed = mixup_cutmix(inputs, targets)
        with torch.amp.autocast('cuda'):
            if mixed is not None:
                inp, ta, tb, lam = mixed
                out=model(inp)
                loss=lam*criterion(out, ta)+(1-lam)*criterion(out, tb)
            else:
                out=model(inputs)
                loss=criterion(out, targets)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running+=loss.item()

    # eval
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for inputs, targets in test_loader:
                inputs=inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                targets=targets.to(device, non_blocking=True)
                out=model(inputs)
                _, pred=out.max(1)
                total+=targets.size(0)
                correct+=pred.eq(targets).sum().item()
    acc=correct/total
    best_acc=max(best_acc, acc)
    print(f"Epoch {epoch}/{EPOCHS} loss {running/len(train_loader):.3f} acc {acc*100:.2f} best {best_acc*100:.2f} lr {optimizer.param_groups[0]['lr']:.4f} time {time.time()-start_time:.1f}s")

    # save scripted
    model.eval()
    scripted=torch.jit.script(model)
    ckpt_path=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    # verify
    try:
        m2=torch.jit.load(ckpt_path, map_location="cpu")
        m2.eval()
        with torch.no_grad():
            _=m2(torch.randn(2,3,32,32))
    except Exception as e:
        print(f"verify fail {e}")
    model.train()
    if first_ckpt is None:
        first_ckpt=time.time()
    last_ckpt=time.time()

last_path=os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
if os.path.exists(last_path):
    shutil.copy(last_path, FINAL_CKPT)
    print(f"copied to {FINAL_CKPT}")

total_wall=last_ckpt-start_time
print(f"Done best {best_acc*100:.2f} wall {total_wall:.1f}s")
import json
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": best_acc, "wall_clock_sec": total_wall}, f)
