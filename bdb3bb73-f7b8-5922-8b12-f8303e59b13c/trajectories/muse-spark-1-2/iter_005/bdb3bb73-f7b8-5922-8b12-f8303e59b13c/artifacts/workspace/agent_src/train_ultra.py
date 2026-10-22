import os, time, shutil, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/scratch_ckpt"
EPOCHS = 30
BATCH = 1024
LR = 2.0
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Preload data to RAM as uint8
def preload(folder):
    # folder has subdirs 0-9 named by class
    classes = sorted(os.listdir(folder))
    class_to_idx = {c:i for i,c in enumerate(classes)}
    all_imgs = []
    all_labels = []
    for c in classes:
        d = os.path.join(folder, c)
        for fname in sorted(os.listdir(d)):
            path = os.path.join(d, fname)
            img = Image.open(path).convert('RGB')
            arr = np.array(img, dtype=np.uint8)  # HWC
            all_imgs.append(arr)
            all_labels.append(class_to_idx[c])
    all_imgs = np.stack(all_imgs)  # N H W C
    all_labels = np.array(all_labels, dtype=np.int64)
    # convert to NCHW uint8 tensor
    t = torch.from_numpy(all_imgs).permute(0,3,1,2).contiguous()  # N C H W
    l = torch.from_numpy(all_labels)
    return t, l

print("Preloading...")
train_imgs, train_labels = preload(DATA_TRAIN)
test_imgs, test_labels = preload(DATA_TEST)
print(f"Train {train_imgs.shape} Test {test_imgs.shape}")

# Move test to GPU normalized once
device = torch.device("cuda")
mean_gpu = MEAN.to(device)
std_gpu = STD.to(device)

# Model
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if len(self.shortcut) > 0 else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class CifarResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen=4):
        super().__init__()
        n = (depth - 4)//6
        k = widen
        self.in_ch = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(16*k, n, stride=1)
        self.layer2 = self._make_layer(32*k, n, stride=2)
        self.layer3 = self._make_layer(64*k, n, stride=2)
        self.bn = nn.BatchNorm2d(64*k)
        self.fc = nn.Linear(64*k, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, out_ch, blocks, stride):
        layers=[]
        for i in range(blocks):
            s = stride if i==0 else 1
            layers.append(BasicBlock(self.in_ch, out_ch, s))
            self.in_ch = out_ch
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# GPU augmentation functions
def gpu_augment(x):  # x is float in [0,1] normalized? We'll do augment before normalize
    # x: N C H W float [0,1] on GPU
    # RandomCrop with padding 4
    # pad 4 on each side -> 40x40, then random crop 32
    # Use F.pad
    x = F.pad(x, (4,4,4,4), mode='reflect')  # N C 40 40
    # random crop
    N = x.size(0)
    # generate random offsets
    # Use torch.randint
    h_off = torch.randint(0, 9, (N,), device=x.device)
    w_off = torch.randint(0, 9, (N,), device=x.device)
    # crop via batched indexing - loop is slow, use unfold trick or just use torch.stack with slicing?
    # For speed, we can use a vectorized crop via grid_sample or via manual
    # Simple: use for loop over batch but N=1024, loop 1024 times per batch is heavy
    # Alternative: use torch's random crop via torchvision on GPU? Let's implement via batched gather
    # Use unfold: x is N C 40 40, we want N C 32 32
    # We can use F.unfold? Easier to just use loop with small N? 1024 loops of slicing is okay in Python? Might be slow.
    # Let's try to use torch.stack with list comprehension - still Python loop
    # Alternative: use Kornia style: generate crops via indexing with advanced indexing
    # We'll do: create output tensor and fill via batched operation using torch.arange
    # Simpler: use random horizontal flip + crop via affine grid
    # For now, implement loop in Python but with torch ops - should be okay for 50 steps per epoch (50*1024 loops = 51200 slices) maybe okay
    # Let's try to vectorize with torch's `torch.nn.functional.unfold` not needed
    # We'll do a fast path: use `torch.randint` and then use `torch.stack` with slicing via Python loop but compiled?
    # Let's just do loop
    crops = []
    for i in range(N):
        crops.append(x[i:i+1, :, h_off[i]:h_off[i]+32, w_off[i]:w_off[i]+32])
    x = torch.cat(crops, dim=0)
    # Random horizontal flip
    flip_mask = torch.rand(N, device=x.device) < 0.5
    # flip where mask true
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
    return x

# Alternative faster: use torchvision transforms on CPU preloaded? But we want GPU
# We'll keep gpu_augment as above but we can also try to use CutMix/Mixup after

def rand_bbox(size, lam):
    W = size[-1]; H = size[-2]
    cut_rat = (1. - lam) ** 0.5
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = random.randint(0, W-1); cy = random.randint(0, H-1)
    bbx1 = max(0, cx - cut_w // 2); bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2); bby2 = min(H, cy + cut_h // 2)
    return bbx1, bby1, bbx2, bby2

def mixup_cutmix(batch_x, batch_y, alpha=1.0):
    if random.random() > 0.5:
        lam = random.betavariate(alpha, alpha) if alpha>0 else 1.0
        idx = torch.randperm(batch_x.size(0), device=batch_x.device)
        mixed_x = lam * batch_x + (1 - lam) * batch_x[idx]
        y_a, y_b = batch_y, batch_y[idx]
        return mixed_x, y_a, y_b, lam
    else:
        lam = random.betavariate(alpha, alpha) if alpha>0 else 1.0
        idx = torch.randperm(batch_x.size(0), device=batch_x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(batch_x.size(), lam)
        mixed_x = batch_x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = batch_x[idx, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size(-1) * batch_x.size(-2)))
        y_a, y_b = batch_y, batch_y[idx]
        return mixed_x, y_a, y_b, lam

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        if f.startswith("ckpt_epoch_"):
            os.remove(os.path.join(CKPT_DIR, f))
    print(f"BATCH {BATCH} LR {LR} EPOCHS {EPOCHS}")

    model = CifarResNet(depth=28, widen=4).to(device)
    model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
    warmup_epochs = 2
    def lr_lambda(epoch):
        e = epoch + 1
        if e <= warmup_epochs:
            return e / warmup_epochs
        else:
            progress = (e - warmup_epochs) / (EPOCHS - warmup_epochs)
            return 0.5 * (1 + math.cos(progress * math.pi))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    # Prepare test loader GPU normalized
    test_imgs_gpu = test_imgs.to(device, non_blocking=True).float().div(255.0)
    test_imgs_gpu = (test_imgs_gpu - mean_gpu) / std_gpu
    test_imgs_gpu = test_imgs_gpu.to(memory_format=torch.channels_last)
    test_labels_gpu = test_labels.to(device)

    # Train indices
    N = train_imgs.size(0)
    train_imgs_gpu_cache = train_imgs  # keep on CPU, move per batch

    start_wall = time.time()
    first_ckpt_time = None
    last_ckpt_time = None
    best_acc = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        t0 = time.time()
        # shuffle
        perm = torch.randperm(N)
        # iterate batches
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            x = train_imgs[idx].to(device, non_blocking=True).float().div(255.0)  # N C H W [0,1]
            y = train_labels[idx].to(device, non_blocking=True)
            # GPU augment
            x = gpu_augment(x)
            # normalize
            x = (x - mean_gpu) / std_gpu
            x = x.to(memory_format=torch.channels_last)
            if random.random() < 0.5:
                x, y_a, y_b, lam = mixup_cutmix(x, y, alpha=1.0)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = criterion(out, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, test_imgs_gpu.size(0), 512):
                x = test_imgs_gpu[i:i+512]
                y = test_labels_gpu[i:i+512]
                with torch.amp.autocast('cuda'):
                    out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct/total
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch:03d}/{EPOCHS} acc={acc*100:.2f}% best={best_acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} time={time.time()-t0:.1f}s")
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted = torch.jit.script(model.eval())
        scripted.save(ckpt_path)
        if first_ckpt_time is None:
            first_ckpt_time = time.time()
        last_ckpt_time = time.time()

    total_wall = last_ckpt_time - start_wall if last_ckpt_time else time.time() - start_wall
    print(f"Done best {best_acc*100:.2f}% total_wall {total_wall:.1f}s")
    import json
    with open("/tmp/ultra_metrics.json","w") as f:
        json.dump({"accuracy": best_acc, "wall_clock_sec": total_wall}, f)

if __name__ == "__main__":
    main()
