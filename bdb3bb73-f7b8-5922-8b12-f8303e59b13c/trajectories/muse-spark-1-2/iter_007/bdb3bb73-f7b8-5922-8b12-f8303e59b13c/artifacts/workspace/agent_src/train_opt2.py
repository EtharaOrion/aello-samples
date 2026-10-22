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
EPOCHS = 35
BATCH = 2048
BATCH_EVAL = 1024
LR = 1.5
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1
WARMUP = 2
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CifarResNet(nn.Module):
    def __init__(self, widen=4, num_classes=10):
        super().__init__()
        c1 = 16*widen; c2=32*widen; c3=64*widen
        self.conv1 = nn.Conv2d(3, c1, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.layer1 = self._make(c1,c1,4,1)
        self.layer2 = self._make(c1,c2,4,2)
        self.layer3 = self._make(c2,c3,4,2)
        self.bn_final = nn.BatchNorm2d(c3)
        self.fc = nn.Linear(c3, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias,0)
    def _make(self, in_ch, out_ch, n, stride):
        layers=[Block(in_ch,out_ch,stride)]
        for _ in range(1,n):
            layers.append(Block(out_ch,out_ch,1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out,1)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out

def preload_train():
    print("Preloading train...")
    # Use ImageFolder to get paths and labels, then load via PIL and convert to tensor
    # Faster: walk directory
    classes = sorted(os.listdir(DATA_TRAIN))
    class_to_idx = {c:i for i,c in enumerate(classes)}
    paths = []
    labels = []
    for c in classes:
        d = os.path.join(DATA_TRAIN, c)
        for f in os.listdir(d):
            paths.append(os.path.join(d,f))
            labels.append(class_to_idx[c])
    # load all images
    N = len(paths)
    data = torch.empty(N,3,32,32, dtype=torch.uint8)
    # use torchvision to load quickly with workers? simple loop
    import torchvision.io as io
    # Use PIL for now, but try to parallelize
    from concurrent.futures import ThreadPoolExecutor
    def load_one(i):
        p = paths[i]
        img = Image.open(p).convert('RGB')
        arr = np.array(img, dtype=np.uint8) # HWC
        t = torch.from_numpy(arr).permute(2,0,1)
        return t
    # sequential for simplicity but with 50k it's okay ~ few seconds
    for i, p in enumerate(paths):
        img = Image.open(p).convert('RGB')
        arr = np.array(img, dtype=np.uint8)
        data[i] = torch.from_numpy(arr).permute(2,0,1)
        if i%10000==0:
            print(f"  {i}/{N}")
    labels = torch.tensor(labels, dtype=torch.long)
    print(f"Preloaded {N} images")
    return data, labels

def preload_test():
    print("Preloading test...")
    classes = sorted(os.listdir(DATA_TEST))
    class_to_idx = {c:i for i,c in enumerate(classes)}
    paths=[]; labels=[]
    for c in classes:
        d=os.path.join(DATA_TEST,c)
        for f in os.listdir(d):
            paths.append(os.path.join(d,f))
            labels.append(class_to_idx[c])
    N=len(paths)
    data=torch.empty(N,3,32,32,dtype=torch.uint8)
    for i,p in enumerate(paths):
        img=Image.open(p).convert('RGB')
        arr=np.array(img,dtype=np.uint8)
        data[i]=torch.from_numpy(arr).permute(2,0,1)
    labels=torch.tensor(labels,dtype=torch.long)
    return data, labels

# GPU augmentation helpers
def gpu_augment(x, mean, std):
    # x: uint8 [B,3,32,32] on GPU, values 0-255
    # RandomCrop with padding 4, RandomFlip
    B = x.size(0)
    # pad 4 on each side -> 40x40
    x = F.pad(x.float(), (4,4,4,4), mode='reflect')  # [B,3,40,40]
    # random crop 32x32
    # generate random offsets
    # Use torch.randint
    h_off = torch.randint(0, 9, (B,), device=x.device)
    w_off = torch.randint(0, 9, (B,), device=x.device)
    # crop via loop? Use vectorized gather: we can use unfold or just loop per batch is okay for 2048? Might be slow.
    # Instead use batched indexing via torch.stack
    # For speed, use simple method: generate crops via F.unfold? Let's do loop with torch but compiled?
    # We'll do efficient batched crop using torch's grid? Simpler: use for loop in python but with GPU tensors it's still okay for 2048? 2048 iterations per batch would be heavy.
    # Alternative: use torchvision transforms on GPU via Kornia? Let's implement vectorized crop using torch.gather
    # We can create a batch of crops by using advanced indexing: x is [B,3,40,40], we want [B,3,32,32] where each B has different offset.
    # We can do: create indices and use torch's unfold trick: use torch.nn.functional.unfold? Not.
    # Simpler: use random crop via rolling? Let's just use a fast approximation: use same crop for whole batch (less augmentation but faster)
    # For speed, use same random crop per batch (not per sample) - still gives augmentation but less diverse. Let's do per-batch crop for speed.
    # Actually we can do per-sample with vectorized operation using torch's `torch.stack` and indexing via `torch.arange`
    # Let's implement per-sample crop using `torch` vectorization with `F.pad` already done, then use `torch` to gather.
    # Approach: create output tensor and fill via loop over B but using GPU slicing which is still Python loop 2048 times -> ~2048 kernel launches per batch -> slow.
    # Better: use `torch` to do batched crop via `torch.nn.functional.affine_grid`? Could use `grid_sample`.
    # Simplest fast: use per-batch single crop (one random offset for whole batch)
    ho = int(h_off[0].item()); wo = int(w_off[0].item())
    x = x[:, :, ho:ho+32, wo:wo+32]
    # random flip
    flip_mask = torch.rand(B, device=x.device) < 0.5
    # flip where mask true
    # Use torch.flip on selected indices
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
    # to float 0-1, normalize
    x = x / 255.0
    x = (x - mean.to(x.device)) / std.to(x.device)
    # RandomErasing 25% prob per sample, scale 0.02-0.1
    # Implement simple erasing: for 25% of samples, erase random rectangle
    erase_mask = torch.rand(B, device=x.device) < 0.25
    if erase_mask.any():
        for idx in torch.where(erase_mask)[0]:
            # random area
            area = 32*32
            target_area = random.uniform(0.02, 0.1) * area
            aspect = random.uniform(0.3, 3.3)
            h = int(round(math.sqrt(target_area * aspect)))
            w = int(round(math.sqrt(target_area / aspect)))
            if h < 32 and w < 32:
                y1 = random.randint(0, 32 - h)
                x1 = random.randint(0, 32 - w)
                x[idx, :, y1:y1+h, x1:x1+w] = 0  # after normalize, 0 corresponds to mean
    return x

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        if f.startswith("ckpt_epoch_"):
            os.remove(os.path.join(CKPT_DIR,f))
    train_data, train_labels = preload_train()
    test_data, test_labels = preload_test()
    # move test to GPU for fast eval? Keep on CPU and move per batch
    # For training, keep train_data on CPU (uint8) and move per batch to GPU
    # Use pinned memory for faster transfer
    train_data = train_data.pin_memory() if torch.cuda.is_available() else train_data
    # Create model
    model = CifarResNet(widen=4).to(device)
    model = model.to(memory_format=torch.channels_last)
    # try torch.compile
    try:
        model = torch.compile(model, mode="max-autotune")
        print("Compiled model")
    except Exception as e:
        print(f"Compile failed: {e}")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True, fused=True if hasattr(torch.optim.SGD, 'fused') else False)
    def lr_lambda(epoch):
        e=epoch+1
        if e<=WARMUP:
            return e/WARMUP
        else:
            prog=(e-WARMUP)/(EPOCHS-WARMUP)
            return 0.5*(1+math.cos(math.pi*prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    mean = MEAN; std = STD
    start_wall=time.time()
    print(f"Start training epochs={EPOCHS} batch={BATCH} lr={LR}")

    # For fast shuffling, create indices
    N = train_data.size(0)
    for epoch in range(1, EPOCHS+1):
        model.train()
        # shuffle
        perm = torch.randperm(N)
        # iterate batches
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            if idx.size(0) != BATCH:
                continue  # drop_last
            x_cpu = train_data[idx]  # [B,3,32,32] uint8
            y = train_labels[idx].to(device, non_blocking=True)
            x = x_cpu.to(device, non_blocking=True)  # uint8 on GPU
            # augment on GPU
            x = gpu_augment(x, mean, std)
            # mixup/cutmix 80% prob
            if random.random() < 0.8:
                r = random.random()
                if r < 0.5:
                    lam = random.betavariate(0.2,0.2)
                    perm2 = torch.randperm(BATCH, device=device)
                    mixed_x = lam * x + (1-lam) * x[perm2]
                    y_a, y_b = y, y[perm2]
                    lam_a = lam
                else:
                    lam = random.betavariate(1.0,1.0)
                    perm2 = torch.randperm(BATCH, device=device)
                    # cutmix bbox
                    W=32; H=32
                    cut_rat = (1.-lam)**0.5
                    cut_w = int(W*cut_rat); cut_h = int(H*cut_rat)
                    cx = random.randint(0,W-1); cy = random.randint(0,H-1)
                    bbx1 = max(0, cx - cut_w//2); bby1 = max(0, cy - cut_h//2)
                    bbx2 = min(W, cx + cut_w//2); bby2 = min(H, cy + cut_h//2)
                    mixed_x = x.clone()
                    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[perm2, :, bby1:bby2, bbx1:bbx2]
                    lam = 1 - ((bbx2-bbx1)*(bby2-bby1)/(W*H))
                    y_a, y_b = y, y[perm2]
                    lam_a = lam
                with torch.amp.autocast('cuda'):
                    logits = model(mixed_x)
                    loss = lam_a * criterion(logits, y_a) + (1-lam_a) * criterion(logits, y_b)
            else:
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # eval
        model.eval()
        correct=0; total=0
        with torch.no_grad():
            for i in range(0, test_data.size(0), BATCH_EVAL):
                x = test_data[i:i+BATCH_EVAL].float() / 255.0
                x = (x - MEAN.squeeze(-1).squeeze(-1).view(1,3,1,1)) / STD.squeeze(-1).squeeze(-1).view(1,3,1,1)
                # Actually MEAN is [1,3,1,1], need to handle
                # Use precomputed mean/std tensors
                # Let's just do manual
                # x already on CPU, move to GPU
                y = test_labels[i:i+BATCH_EVAL].to(device)
                x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                pred = out.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        acc = correct/total
        elapsed = time.time()-start_wall
        print(f"Epoch {epoch:03d}/{EPOCHS} acc={acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f} time={elapsed:.1f}s")
        # save scripted checkpoint - need to unwrap compiled model
        # Get original model for scripting
        to_script = model._orig_mod if hasattr(model, '_orig_mod') else model
        # Ensure eval mode
        to_script.eval()
        scripted = torch.jit.script(to_script)
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(ckpt_path)
        # back to train
        to_script.train()
        model.train()

    total_time=time.time()-start_wall
    print(f"Done total_time={total_time:.1f}s")
    import json
    with open("/tmp/fast2_metrics.json","w") as f:
        json.dump({"accuracy":acc, "wall_clock_sec":total_time},f)

if __name__=="__main__":
    main()
