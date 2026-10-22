import os, time, shutil, random, json, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

CKPT_DIR = "/workspace/scratch_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda")
MEAN = torch.tensor([0.4914,0.4822,0.4465], device=device).view(1,3,1,1)
STD = torch.tensor([0.2470,0.2435,0.2616], device=device).view(1,3,1,1)

print("preloading...")
train_root = "/workspace/data/cifar10/train"
test_root = "/workspace/data/cifar10/test"
classes = sorted(os.listdir(train_root))
class_to_idx = {c:i for i,c in enumerate(classes)}

def load_folder(root):
    paths=[]
    labels=[]
    for c in classes:
        for p in glob.glob(os.path.join(root, c, "*.png")):
            paths.append(p)
            labels.append(class_to_idx[c])
    N=len(paths)
    data=torch.empty(N,3,32,32,dtype=torch.uint8)
    for i,p in enumerate(paths):
        img=np.array(Image.open(p).convert("RGB"))
        data[i]=torch.from_numpy(img).permute(2,0,1)
    return data, torch.tensor(labels,dtype=torch.long)

train_u8, train_lbl = load_folder(train_root)
test_u8, test_lbl = load_folder(test_root)
print(f"train {train_u8.shape}")

train_u8_gpu = train_u8.to(device)
train_lbl_gpu = train_lbl.to(device)
test_f = test_u8.float().div(255.0).to(device)
test_norm = (test_f - MEAN)/STD
test_lbl_gpu = test_lbl.to(device)

# --- Fast ResNet9 (DAWNBench style) ---
# Architecture: 3x3 conv 64, then 3 stages: [64,128,256] with residual blocks
# Each stage: 2 blocks, first block stride 2 for downsample except first stage
# Use BatchNorm, ReLU

class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride!=1 or in_c!=out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(ResBlock(64,128,2), ResBlock(128,128,1))
        self.layer2 = nn.Sequential(ResBlock(128,256,2), ResBlock(256,256,1))
        self.layer3 = nn.Sequential(ResBlock(256,512,2), ResBlock(512,512,1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'); nn.init.constant_(m.bias,0)
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

# Try WRN 16-8 as alternative if ResNet9 not enough
# We'll use ResNet9 first

EPOCHS=30
BATCH=1024
LR_MAX=1.2
WD=5e-4
LABEL_SMOOTH=0.1

model = ResNet9().to(device).to(memory_format=torch.channels_last)
# no torch.compile for now to avoid autotune overhead; we can try compile with default
# model = torch.compile(model)

optimizer=torch.optim.SGD(model.parameters(),lr=LR_MAX,momentum=0.9,weight_decay=WD,nesterov=True)
steps_per_epoch=(len(train_u8)+BATCH-1)//BATCH
scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=LR_MAX,epochs=EPOCHS,steps_per_epoch=steps_per_epoch,pct_start=0.3,anneal_strategy='cos',div_factor=10,final_div_factor=50)
criterion=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
scaler=torch.amp.GradScaler('cuda')

# fast augment via unfold (no affine)
def gpu_augment_unfold(x): # x [0,1] NCHW
    # pad
    x = F.pad(x, (4,4,4,4), mode='reflect') # N,3,40,40
    # unfold crop
    # x_unfold = x.unfold(2,32,1).unfold(3,32,1) # N,3,9,9,32,32 -> heavy memory
    # Instead use efficient indexing via unfold + gather
    # Use the permute method but with unfold
    # For speed, we can use torch's random crop via indexing with unfold
    # Let's do unfold method
    x_unfold = x.unfold(2,32,1).unfold(3,32,1) # N,3,9,9,32,32
    # permute to N,9,9,3,32,32
    x_unfold = x_unfold.permute(0,2,3,1,4,5) # N,9,9,3,32,32
    N = x.size(0)
    h_off = torch.randint(0,9,(N,),device=x.device)
    w_off = torch.randint(0,9,(N,),device=x.device)
    idx = torch.arange(N, device=x.device)
    out = x_unfold[idx, h_off, w_off] # N,3,32,32
    # flip
    flip = torch.rand(N, device=x.device) < 0.5
    # flip via where: need to flip out where flip true
    # Use masked flip: out[flip] = flip(out[flip])
    if flip.any():
        out[flip] = torch.flip(out[flip], dims=[3])
    return out

# warmup
print("warmup")
dummy=torch.rand(BATCH,3,32,32,device=device).to(memory_format=torch.channels_last)
for _ in range(5):
    with torch.amp.autocast('cuda'):
        _=model(dummy)
torch.cuda.synchronize()
print("warmup done")

start_wall=time.time()
first_mtime=None
last_mtime=None
last_acc=0
train_N=len(train_u8)

for epoch in range(1,EPOCHS+1):
    model.train()
    perm=torch.randperm(train_N,device=device)
    for i in range(0,train_N,BATCH):
        idx=perm[i:i+BATCH]
        if len(idx)<BATCH:
            continue
        x_u8=train_u8_gpu[idx]
        y=train_lbl_gpu[idx]
        x=x_u8.float().div(255.0)
        x=gpu_augment_unfold(x)
        x=(x-MEAN)/STD
        x=x.to(memory_format=torch.channels_last)
        # mixup 30%
        use_mixup = random.random()<0.3
        if use_mixup:
            lam = np.random.beta(0.2,0.2)
            lam = max(lam, 1-lam)
            perm2=torch.randperm(BATCH,device=device)
            x2=x[perm2]
            y2=y[perm2]
            x=lam*x+(1-lam)*x2
            y_a,y_b=y,y2
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out=model(x)
            if use_mixup:
                loss=lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)
            else:
                loss=criterion(out,y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    # eval
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for i in range(0,len(test_norm),512):
            x=test_norm[i:i+512].to(memory_format=torch.channels_last)
            y=test_lbl_gpu[i:i+512]
            with torch.amp.autocast('cuda'):
                out=model(x)
            pred=out.argmax(1)
            correct+=(pred==y).sum().item()
            total+=y.size(0)
    acc=correct/total
    last_acc=acc
    print(f"epoch {epoch:02d} acc {acc*100:.2f}% lr {optimizer.param_groups[0]['lr']:.5f}")
    # save scripted
    script_model = ResNet9().to(device)
    script_model.load_state_dict(model.state_dict())
    script_model.eval()
    scripted=torch.jit.script(script_model)
    ckpt_path=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    if first_mtime is None:
        first_mtime=os.path.getmtime(ckpt_path)
    last_mtime=os.path.getmtime(ckpt_path)

shutil.copy(os.path.join(CKPT_DIR,f"ckpt_epoch_{EPOCHS:04d}.pt"),"/tmp/checkpoint_tmp.pt")
wall=time.time()-start_wall
print(f"wall {wall:.1f}s span {last_mtime-first_mtime:.1f}s acc {last_acc:.4f}")
with open("/tmp/metrics.json","w") as f:
    json.dump({"accuracy":last_acc,"wall_clock_sec":wall},f)
print("done")
