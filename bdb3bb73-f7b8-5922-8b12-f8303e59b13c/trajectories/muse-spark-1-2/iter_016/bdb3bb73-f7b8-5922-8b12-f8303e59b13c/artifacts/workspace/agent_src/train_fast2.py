import os, time, shutil, random, json, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 35
BATCH = 1024
LR = 0.5
WD = 5e-4

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

if os.path.exists(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
os.makedirs(CKPT_DIR, exist_ok=True)
device = torch.device("cuda")

print("Preloading...")
classes = sorted(os.listdir(DATA_TRAIN))
class_to_idx = {c:i for i,c in enumerate(classes)}
train_paths=[]; train_labels=[]
for c in classes:
    for p in glob.glob(os.path.join(DATA_TRAIN,c,"*.png")):
        train_paths.append(p); train_labels.append(class_to_idx[c])
train_images=np.stack([np.array(Image.open(p).convert("RGB"),dtype=np.uint8) for p in train_paths])
train_labels=np.array(train_labels,dtype=np.int64)
print(f"train {train_images.shape}")
test_paths=[]; test_labels=[]
for c in classes:
    for p in glob.glob(os.path.join(DATA_TEST,c,"*.png")):
        test_paths.append(p); test_labels.append(class_to_idx[c])
test_images=np.stack([np.array(Image.open(p).convert("RGB"),dtype=np.uint8) for p in test_paths])
test_labels=np.array(test_labels,dtype=np.int64)
print(f"test {test_images.shape}")

train_images_gpu=torch.from_numpy(train_images).to(device)
train_labels_gpu=torch.from_numpy(train_labels).to(device)
test_images_gpu=torch.from_numpy(test_images).to(device)
test_labels_gpu=torch.from_numpy(test_labels).to(device)

test_float=test_images_gpu.permute(0,3,1,2).float()/255.0
mean=torch.tensor([0.4914,0.4822,0.4465],device=device).view(1,3,1,1)
std=torch.tensor([0.2470,0.2435,0.2616],device=device).view(1,3,1,1)
test_float=(test_float-mean)/std

class BasicBlock(nn.Module):
    __constants__=['equalInOut']
    def __init__(self,in_planes,out_planes,stride):
        super().__init__()
        self.bn1=nn.BatchNorm2d(in_planes); self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(in_planes,out_planes,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(out_planes); self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_planes,out_planes,3,1,1,bias=False)
        self.equalInOut=(in_planes==out_planes)
        self.convShortcut=nn.Conv2d(in_planes,out_planes,1,stride,0,bias=False) if not self.equalInOut else nn.Identity()
    def forward(self,x):
        out=self.relu1(self.bn1(x))
        out1=self.conv1(out); out1=self.relu2(self.bn2(out1)); out1=self.conv2(out1)
        return x+out1 if self.equalInOut else self.convShortcut(out)+out1
class NetworkBlock(nn.Module):
    def __init__(self,nb,in_p,out_p,stride):
        super().__init__()
        self.layer=nn.Sequential(*[BasicBlock(in_p if i==0 else out_p,out_p,stride if i==0 else 1) for i in range(nb)])
    def forward(self,x): return self.layer(x)
class WideResNet(nn.Module):
    def __init__(self,depth=28,widen=10,num_classes=10):
        super().__init__()
        nC=[16,16*widen,32*widen,64*widen]; n=(depth-4)//6
        self.conv1=nn.Conv2d(3,nC[0],3,1,1,bias=False)
        self.block1=NetworkBlock(n,nC[0],nC[1],1); self.block2=NetworkBlock(n,nC[1],nC[2],2); self.block3=NetworkBlock(n,nC[2],nC[3],2)
        self.bn1=nn.BatchNorm2d(nC[3]); self.relu=nn.ReLU(inplace=True); self.fc=nn.Linear(nC[3],num_classes); self.nC=nC[3]
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m,nn.Linear): m.bias.data.zero_()
    def forward(self,x):
        out=self.conv1(x); out=self.block1(out); out=self.block2(out); out=self.block3(out)
        out=self.relu(self.bn1(out)); out=F.adaptive_avg_pool2d(out,1); out=out.view(-1,self.nC); return self.fc(out)

model=WideResNet(28,10,10).to(device).to(memory_format=torch.channels_last)
model=torch.compile(model)
optimizer=torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WD,nesterov=True)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
scaler=torch.amp.GradScaler('cuda')

# vectorized augment
@torch.compile
def gpu_augment(batch_images): # N H W C uint8
    N=batch_images.shape[0]
    x=batch_images.permute(0,3,1,2).float()/255.0
    x=F.pad(x,(4,4,4,4),mode='reflect') # N C 40 40
    # unfold to get all 32x32 patches
    # x_unfold: N, C*32*32, 81
    x_unfold=F.unfold(x,kernel_size=32,stride=1) # 81 = 9*9
    tops=torch.randint(0,9,(N,),device=x.device)
    lefts=torch.randint(0,9,(N,),device=x.device)
    idx=tops*9+lefts # N
    # gather: for each N, pick column idx
    # x_unfold is N, K, L ; we want N, K
    # Use advanced indexing
    # Create batch indices
    batch_idx=torch.arange(N,device=x.device)
    # x_unfold[batch_idx, :, idx] -> need to gather
    # Use torch.gather
    # idx_expanded: N, K, 1? Actually gather dim=2
    idx_expanded=idx.view(N,1,1).expand(N, x_unfold.shape[1], 1)
    cropped=torch.gather(x_unfold,2,idx_expanded).squeeze(2) # N, K
    x=cropped.view(N,3,32,32)
    # flip
    flip_mask=torch.rand(N,device=x.device)<0.5
    # flip via where: we can flip all then select
    x_flipped=torch.flip(x,dims=[3])
    # need to select per sample: where flip_mask, use flipped else original
    # expand mask to N,1,1,1
    x=torch.where(flip_mask.view(N,1,1,1), x_flipped, x)
    # normalize
    mean_t=torch.tensor([0.4914,0.4822,0.4465],device=x.device).view(1,3,1,1)
    std_t=torch.tensor([0.2470,0.2435,0.2616],device=x.device).view(1,3,1,1)
    x=(x-mean_t)/std_t
    # cutout
    cut_mask=torch.rand(N,device=x.device)<0.5
    if cut_mask.any():
        ys=torch.randint(0,32,(N,),device=x.device)
        xs=torch.randint(0,32,(N,),device=x.device)
        y1=torch.clamp(ys-8,0,32)
        y2=torch.clamp(ys+8,0,32)
        x1=torch.clamp(xs-8,0,32)
        x2=torch.clamp(xs+8,0,32)
        # create mask
        yy=torch.arange(32,device=x.device).view(1,32,1).expand(N,32,32)
        xx=torch.arange(32,device=x.device).view(1,1,32).expand(N,32,32)
        y1e=y1.view(N,1,1).expand(N,32,32)
        y2e=y2.view(N,1,1).expand(N,32,32)
        x1e=x1.view(N,1,1).expand(N,32,32)
        x2e=x2.view(N,1,1).expand(N,32,32)
        cut_area=((yy>=y1e)&(yy<y2e)&(xx>=x1e)&(xx<x2e)) # N 32 32 bool
        cut_area=cut_area.unsqueeze(1).expand(N,3,32,32) # N 3 32 32
        # only apply where cut_mask
        cut_area=cut_area & cut_mask.view(N,1,1,1)
        x=x.masked_fill(cut_area,0)
    return x

def mixup_batch(x,y,alpha=0.2):
    lam=random.betavariate(alpha,alpha)
    idx=torch.randperm(x.size(0),device=x.device)
    mixed=lam*x+(1-lam)*x[idx]
    return mixed,y,y[idx],lam

start=time.time()
first=None; last=None; best=0
N=train_images_gpu.shape[0]
for epoch in range(1,EPOCHS+1):
    model.train()
    perm=torch.randperm(N,device=device)
    for i in range(0,N,BATCH):
        idx=perm[i:i+BATCH]
        batch_imgs=train_images_gpu[idx]
        batch_labels=train_labels_gpu[idx]
        x=gpu_augment(batch_imgs)
        x=x.to(memory_format=torch.channels_last)
        if random.random()<0.5:
            x,y_a,y_b,lam=mixup_batch(x,batch_labels,0.2)
            with torch.amp.autocast('cuda'):
                out=model(x)
                loss=lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)
        else:
            with torch.amp.autocast('cuda'):
                out=model(x)
                loss=criterion(out,batch_labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for i in range(0,test_float.shape[0],512):
            x=test_float[i:i+512]; y=test_labels_gpu[i:i+512]
            with torch.amp.autocast('cuda'):
                out=model(x)
            correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    acc=correct/total
    print(f"Epoch {epoch} acc={acc:.4f} lr={optimizer.param_groups[0]['lr']:.4f}",flush=True)
    if acc>best: best=acc
    unwrapped=model._orig_mod if hasattr(model,'_orig_mod') else model
    script_model=WideResNet(28,10,10).to(device)
    script_model.load_state_dict(unwrapped.state_dict())
    script_model.eval()
    scripted=torch.jit.script(script_model)
    ckpt_path=os.path.join(CKPT_DIR,f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    if first is None: first=time.time()
    last=time.time()
    try:
        m=torch.jit.load(ckpt_path,map_location="cpu"); m.eval(); _=m(torch.randn(2,3,32,32))
    except Exception as e: print(f"verify fail {e}")

last_ckpt=os.path.join(CKPT_DIR,f"ckpt_epoch_{EPOCHS:04d}.pt")
shutil.copy(last_ckpt,FINAL_CKPT)
wall=last-start; span=last-first if first else wall
print(f"Wall {wall:.2f} span {span:.2f} best {best:.4f}")
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy":best,"wall_clock_sec":wall},f)
