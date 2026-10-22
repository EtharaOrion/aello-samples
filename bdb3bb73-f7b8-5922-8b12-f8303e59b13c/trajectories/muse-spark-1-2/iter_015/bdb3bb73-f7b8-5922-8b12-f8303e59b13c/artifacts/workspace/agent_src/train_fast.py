import os, time, shutil, random, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import torchvision.transforms as T
import numpy as np

DATA_ROOT = "/workspace/data/cifar10"
CKPT_DIR = "/workspace/scratch_ckpt"
FINAL_CKPT = "/workspace/checkpoint.pt"
GRADED_DIR = "/workspace/periodic_checkpoints"
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CLS_TO_IDX = {c:i for i,c in enumerate(CLASSES)}
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

# --- Preload dataset into RAM as tensors ---
def preload(root, split):
    base = os.path.join(root, split)
    imgs = []
    labels = []
    for cls in CLASSES:
        d = os.path.join(base, cls)
        for fname in sorted(os.listdir(d)):
            path = os.path.join(d, fname)
            img = Image.open(path).convert("RGB")
            arr = np.array(img)  # HWC uint8
            imgs.append(arr)
            labels.append(CLS_TO_IDX[cls])
    imgs = np.stack(imgs)  # N H W C
    imgs = torch.from_numpy(imgs).permute(0,3,1,2).contiguous()  # N C H W uint8
    labels = torch.tensor(labels, dtype=torch.long)
    # shuffle
    perm = torch.randperm(len(labels))
    return imgs[perm], labels[perm]

# --- Model ---
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.shortcut = nn.Identity() if in_planes==out_planes and stride==1 else nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.dropout(self.conv2(self.relu2(self.bn2(out))))
        out += self.shortcut(x)
        return out

class NetworkBlock(nn.Module):
    def __init__(self, n, in_planes, out_planes, stride, dropout):
        super().__init__()
        layers=[]
        for i in range(n):
            s=stride if i==0 else 1
            inp=in_planes if i==0 else out_planes
            layers.append(BasicBlock(inp,out_planes,s,dropout))
        self.layer=nn.Sequential(*layers)
    def forward(self,x): return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth-4)%6==0
        n=(depth-4)//6; k=widen
        self.conv1=nn.Conv2d(3,16,3,1,1,bias=False)
        self.block1=NetworkBlock(n,16,16*k,1,dropout)
        self.block2=NetworkBlock(n,16*k,32*k,2,dropout)
        self.block3=NetworkBlock(n,32*k,64*k,2,dropout)
        self.bn=nn.BatchNorm2d(64*k)
        self.relu=nn.ReLU(inplace=True)
        self.fc=nn.Linear(64*k,num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m,nn.Linear): m.bias.data.zero_()
    def forward(self,x):
        out=self.conv1(x)
        out=self.block1(out)
        out=self.block2(out)
        out=self.block3(out)
        out=self.relu(self.bn(out))
        out=F.adaptive_avg_pool2d(out,1)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

# GPU augmentation helpers
def gpu_augment(x, mean, std):
    # x: [B,3,32,32] float in [0,1] on GPU
    # RandomCrop 32 with padding 4
    # pad 4 on each side -> 40x40 then random crop 32
    B = x.size(0)
    # pad
    x = F.pad(x, (4,4,4,4), mode='reflect')  # [B,3,40,40]
    # random crop offsets
    h_off = torch.randint(0, 9, (B,), device=x.device)
    w_off = torch.randint(0, 9, (B,), device=x.device)
    # Use batched crop via loop? Do vectorized via grid? Simple loop is okay for B=1024 but may be slow.
    # Instead use unfold trick or just do per-sample via indexing - we can do a fast batched crop using torch
    # We'll do a loop in python but with GPU tensors it's still okay (1024 iterations per batch is heavy)
    # Alternative: use torch.stack with list comprehension - still python loop
    # Let's implement efficient batched crop using advanced indexing
    # Create output tensor
    out = torch.empty(B, 3, 32, 32, device=x.device, dtype=x.dtype)
    for i in range(B):
        out[i] = x[i, :, h_off[i]:h_off[i]+32, w_off[i]:w_off[i]+32]
    x = out
    # Random horizontal flip
    flip_mask = torch.rand(B, device=x.device) < 0.5
    # flip where mask true
    if flip_mask.any():
        x[flip_mask] = x[flip_mask].flip(dims=[3])
    # Normalize
    x = (x - mean.to(x.device)) / std.to(x.device)
    return x

def rand_bbox(size, lam):
    W=size[3]; H=size[2]
    cut_rat=(1-lam)**0.5
    cut_w=int(W*cut_rat); cut_h=int(H*cut_rat)
    cx=np.random.randint(W); cy=np.random.randint(H)
    bbx1=np.clip(cx-cut_w//2,0,W); bby1=np.clip(cy-cut_h//2,0,H)
    bbx2=np.clip(cx+cut_w//2,0,W); bby2=np.clip(cy+cut_h//2,0,H)
    return bbx1,bby1,bbx2,bby2

def main():
    t0=time.time()
    device=torch.device("cuda")
    print(f"Device {torch.cuda.get_device_name(0)}")

    print("Preloading data...")
    train_imgs, train_labels = preload(DATA_ROOT, "train")
    test_imgs, test_labels = preload(DATA_ROOT, "test")
    print(f"Train {train_imgs.shape} Test {test_imgs.shape} preload done in {time.time()-t0:.1f}s")

    # Keep train_imgs on CPU as uint8, will move batch to GPU
    # Create dataset that just indexes
    # Use DataLoader with batch 1024, num_workers 0 (since data is in RAM, no need for workers)
    # Actually with num_workers 0, we avoid overhead. We'll use num_workers 0 and pin_memory False, just slice tensors.

    BATCH=1024
    EPOCHS=40

    # Model
    model=WideResNet(depth=28,widen=10,dropout=0.0).to(device).to(memory_format=torch.channels_last)
    # Try torch.compile for speed
    try:
        model=torch.compile(model, mode="max-autotune")
        print("torch.compile enabled")
    except Exception as e:
        print(f"compile failed {e}")

    optimizer=torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # OneCycleLR: max_lr 0.5, pct_start 0.3, anneal cosine
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.5, epochs=EPOCHS, steps_per_epoch=(50000+BATCH-1)//BATCH, pct_start=0.15, anneal_strategy='cos', div_factor=10, final_div_factor=100)

    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler=torch.amp.GradScaler('cuda')

    # Prepare test loader: we can just evaluate in batches on GPU without dataloader
    # For training, we will manually batch

    if os.path.exists(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR, exist_ok=True)

    mean_gpu=MEAN.cuda()
    std_gpu=STD.cuda()

    # For fast batching, create indices
    N=len(train_labels)
    steps_per_epoch=(N+BATCH-1)//BATCH

    best=0
    for epoch in range(1, EPOCHS+1):
        model.train()
        t_epoch=time.time()
        # shuffle indices each epoch
        perm=torch.randperm(N)
        # we will iterate batches
        for i in range(0, N, BATCH):
            idx=perm[i:i+BATCH]
            x=train_imgs[idx].to(device, non_blocking=True).float().div(255.0)  # [B,3,32,32] float [0,1]
            y=train_labels[idx].to(device, non_blocking=True)

            # GPU augment
            x=gpu_augment(x, mean_gpu, std_gpu)

            # Mixup/CutMix 50% chance
            if random.random()<0.5:
                if random.random()<0.5:
                    # mixup
                    lam=np.random.beta(0.2,0.2)
                    perm2=torch.randperm(x.size(0), device=device)
                    x_mixed=lam*x + (1-lam)*x[perm2]
                    y_a=y; y_b=y[perm2]
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        out=model(x_mixed.to(memory_format=torch.channels_last))
                        loss=lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)
                else:
                    # cutmix
                    lam=np.random.beta(1.0,1.0)
                    perm2=torch.randperm(x.size(0), device=device)
                    bbx1,bby1,bbx2,bby2=rand_bbox(x.size(), lam)
                    x[:, :, bby1:bby2, bbx1:bbx2]=x[perm2, :, bby1:bby2, bbx1:bbx2]
                    lam2=1-((bbx2-bbx1)*(bby2-bby1)/(32*32))
                    y_a=y; y_b=y[perm2]
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        out=model(x.to(memory_format=torch.channels_last))
                        loss=lam2*criterion(out,y_a)+(1-lam2)*criterion(out,y_b)
            else:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out=model(x.to(memory_format=torch.channels_last))
                    loss=criterion(out,y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Eval
        model.eval()
        correct=0; total=0
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for i in range(0, len(test_labels), 1024):
                    x=test_imgs[i:i+1024].to(device).float().div(255.0)
                    x=(x - mean_gpu)/std_gpu
                    x=x.to(memory_format=torch.channels_last)
                    y=test_labels[i:i+1024].to(device)
                    out=model(x)
                    pred=out.argmax(1)
                    correct+=(pred==y).sum().item()
                    total+=y.size(0)
        acc=correct/total
        best=max(best,acc)
        dt=time.time()-t_epoch
        tot=time.time()-t0
        print(f"Epoch {epoch:02d}/{EPOCHS} acc={acc*100:.2f}% best={best*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} epoch_time={dt:.2f}s total={tot:.1f}s")

        # Save scripted checkpoint - need to unwrap compiled model
        # Get underlying model
        raw_model=model._orig_mod if hasattr(model,'_orig_mod') else model
        raw_model.eval()
        try:
            scripted=torch.jit.script(raw_model)
            path=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
            scripted.save(path)
            print(f"  saved {path}")
        except Exception as e:
            print(f"  script failed {e}")
            # try trace
            try:
                ex=torch.randn(1,3,32,32, device=device).to(memory_format=torch.channels_last)
                # need to handle channels_last
                traced=torch.jit.trace(raw_model, ex)
                path=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
                traced.save(path)
            except Exception as e2:
                print(f"  trace also failed {e2}")
        raw_model.train()
        # also need to set model back to train (compiled wrapper)
        model.train()

        if time.time()-t0>1700:
            break

    # Find best epoch and copy last to graded? Actually we need to copy scratch to graded if better
    # For now just report
    # Verify last
    last=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    if os.path.exists(last):
        try:
            m=torch.jit.load(last, map_location="cuda")
            m.eval()
            with torch.no_grad():
                x=torch.randn(2,3,32,32, device="cuda")
                y=m(x)
                print(f"Verify OK {y.shape}")
        except Exception as e:
            print(f"Verify fail {e}")
    wall=time.time()-t0
    print(f"Done wall={wall:.1f}s best={best*100:.2f}%")
    # Write metrics for scratch
    with open("/tmp/scratch_metrics.json","w") as f:
        json.dump({"accuracy":best,"wall_clock_sec":wall},f)

if __name__=="__main__":
    main()
