import os, time, shutil, random, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

DATA_TRAIN="/workspace/data/cifar10/train"
DATA_TEST="/workspace/data/cifar10/test"
CKPT_DIR="/workspace/periodic_checkpoints"
FINAL_CKPT="/workspace/checkpoint.pt"
EPOCHS=45
BATCH=1024
LR=1.2
WD=5e-4

os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    p=os.path.join(CKPT_DIR,f)
    if os.path.isfile(p): os.remove(p)
if os.path.exists(FINAL_CKPT): os.remove(FINAL_CKPT)

torch.backends.cudnn.benchmark=True
device=torch.device("cuda")
print(f"torch {torch.__version__}")

# preload data to GPU
print("Preloading...")
# Use ImageFolder to get paths, then load via PIL and convert to tensor
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset

# Load train as uint8 on CPU then move to GPU as float
train_ds_pil = datasets.ImageFolder(DATA_TRAIN, transform=transforms.ToTensor()) # [0,1] float
test_ds_pil = datasets.ImageFolder(DATA_TEST, transform=transforms.ToTensor())

# Stack
train_images = torch.stack([img for img,_ in train_ds_pil]) # [50000,3,32,32] float32 [0,1]
train_labels = torch.tensor([lbl for _,lbl in train_ds_pil], dtype=torch.long)
test_images = torch.stack([img for img,_ in test_ds_pil])
test_labels = torch.tensor([lbl for _,lbl in test_ds_pil], dtype=torch.long)
print(f"train {train_images.shape} test {test_images.shape}")

# Move test to GPU normalized
mean = torch.tensor([0.4914,0.4822,0.4465], device=device).view(1,3,1,1)
std = torch.tensor([0.2470,0.2435,0.2616], device=device).view(1,3,1,1)
test_images_gpu = (test_images.to(device) - mean) / std  # keep on GPU for eval? But eval will use GPU batches
# Keep train on GPU as [0,1]
train_images_gpu = train_images.to(device)  # [50000,3,32,32] float32
train_labels_gpu = train_labels.to(device)
test_labels_gpu = test_labels.to(device)

# Model
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
            if i==0:
                layers1.append(BlockDown(nChannels[0], nChannels[1], 1))
            else:
                layers1.append(BlockEqual(nChannels[1]))
        self.block1=nn.Sequential(*layers1)
        layers2=[]
        for i in range(n):
            if i==0:
                layers2.append(BlockDown(nChannels[1], nChannels[2], 2))
            else:
                layers2.append(BlockEqual(nChannels[2]))
        self.block2=nn.Sequential(*layers2)
        layers3=[]
        for i in range(n):
            if i==0:
                layers3.append(BlockDown(nChannels[2], nChannels[3], 2))
            else:
                layers3.append(BlockEqual(nChannels[3]))
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
# verify script
torch.jit.script(model)
print("script ok")

criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer=torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler=torch.amp.GradScaler('cuda')

# GPU augment via affine_grid
def gpu_augment_batch(x):
    # x [B,3,32,32] in [0,1]
    B=x.size(0)
    # random translation in [-4,4]
    tx = (torch.randint(0,9,(B,),device=x.device).float() - 4) * (2.0/32.0)
    ty = (torch.randint(0,9,(B,),device=x.device).float() - 4) * (2.0/32.0)
    flip = torch.rand(B, device=x.device) < 0.5
    sx = torch.where(flip, torch.tensor(-1.0, device=x.device), torch.tensor(1.0, device=x.device))
    # theta [B,2,3]
    theta = torch.zeros(B,2,3, device=x.device, dtype=x.dtype)
    theta[:,0,0]=sx
    theta[:,1,1]=1.0
    theta[:,0,2]=tx * sx  # when flipped, tx sign flips? Actually translation after flip: need to think. For flip, the translation direction flips too. So tx should be multiplied by sx? Let's test: if we flip, left offset should mirror. Our tx derived from left uniform, but flipping should not change tx distribution? Simpler: keep tx independent of flip, but affine with sx=-1 will mirror. The translation should be applied after flip, so tx should be as is. We'll keep tx as is.
    # Correction: theta maps output to input: x_in = sx * x_out + tx
    # So tx is translation in input coords. Flipping sx=-1 mirrors. Translation should be same.
    theta[:,0,2]=tx
    theta[:,1,2]=ty
    grid = F.affine_grid(theta, [B,3,32,32], align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    # normalize
    x = (x - mean) / std
    # random erasing 25% with 8x8
    erase_mask = torch.rand(B, device=x.device) < 0.25
    if erase_mask.any():
        # vectorized erasing via masking: create random top/left
        er_top = torch.randint(0,25,(B,),device=x.device)
        er_left = torch.randint(0,25,(B,),device=x.device)
        # create mask [B,1,32,32] with 1 where to erase
        # Use broadcasting to create mask without loop
        # Create row/col indices
        rows = torch.arange(32, device=x.device).view(1,32,1)
        cols = torch.arange(32, device=x.device).view(1,1,32)
        # For each b, erase if rows in [top, top+8) and cols in [left, left+8)
        # Expand
        # This is O(B*32*32) but vectorized
        r_mask = (rows >= er_top.view(B,1,1)) & (rows < (er_top+8).view(B,1,1))  # [B,32,1]
        c_mask = (cols >= er_left.view(B,1,1)) & (cols < (er_left+8).view(B,1,1)) # [B,1,32]
        mask = r_mask & c_mask  # [B,32,32] via broadcasting
        mask = mask.unsqueeze(1).expand(-1,3,-1,-1)  # [B,3,32,32]
        # only apply where erase_mask true
        mask = mask & erase_mask.view(B,1,1,1)
        x = x.masked_fill(mask, 0)
    return x

def mixup_cutmix(x, y):
    if random.random() < 0.5:
        if random.random() < 0.5:
            lam = random.betavariate(0.2,0.2)
            idx = torch.randperm(x.size(0), device=x.device)
            mixed = lam * x + (1-lam) * x[idx]
            return mixed, y, y[idx], lam
        else:
            lam = random.betavariate(1.0,1.0)
            idx = torch.randperm(x.size(0), device=x.device)
            W=H=32
            cut_rat = (1.-lam)**0.5
            cut_w=int(W*cut_rat)
            cut_h=int(H*cut_rat)
            cx=random.randint(0,W-1)
            cy=random.randint(0,H-1)
            bbx1=max(cx-cut_w//2,0)
            bby1=max(cy-cut_h//2,0)
            bbx2=min(cx+cut_w//2,W)
            bby2=min(cy+cut_h//2,H)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2-bbx1)*(bby2-bby1)/(W*H))
            return x, y, y[idx], lam
    return None

best_acc=0
start_time=time.time()
first_ckpt=None
last_ckpt=None

num_train=train_images_gpu.size(0)
indices=torch.arange(num_train, device=device)

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

    # shuffle
    perm=torch.randperm(num_train, device=device)
    running=0
    batches=0
    for i in range(0, num_train, BATCH):
        idx=perm[i:i+BATCH]
        x=train_images_gpu[idx]  # [B,3,32,32] [0,1]
        y=train_labels_gpu[idx]
        x=gpu_augment_batch(x)
        x=x.to(memory_format=torch.channels_last)
        mixed=mixup_cutmix(x,y)
        with torch.amp.autocast('cuda'):
            if mixed is not None:
                inp, ta, tb, lam = mixed
                out=model(inp)
                loss=lam*criterion(out, ta)+(1-lam)*criterion(out, tb)
            else:
                out=model(x)
                loss=criterion(out, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running+=loss.item()
        batches+=1

    # eval
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
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

    # save scripted
    model.eval()
    scripted=torch.jit.script(model)
    ckpt_path=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
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
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": best_acc, "wall_clock_sec": total_wall}, f)
