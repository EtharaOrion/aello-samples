import os, time, shutil, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import pathlib

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 50
BATCH_TRAIN = 2048
BATCH_TEST = 1024
WD = 5e-4
LR_MAX = 1.5

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

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

def load_train_to_ram():
    # Load all train images as uint8 tensors (N,3,32,32)
    p = pathlib.Path(DATA_TRAIN)
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    class_to_idx = {c:i for i,c in enumerate(classes)}
    images = []
    labels = []
    for c in classes:
        idx = class_to_idx[c]
        for img_path in (p/c).glob("*.png"):
            img = Image.open(img_path).convert("RGB")
            # to tensor uint8
            import numpy as np
            arr = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).numpy().copy() if False else __import__('numpy').array(img)).permute(2,0,1) if False else None
            # simpler: use torchvision ToTensor then *255
            # Use PIL -> numpy
            import numpy as np
            arr = np.array(img)  # HWC uint8
            t = torch.from_numpy(arr).permute(2,0,1)  # 3,32,32
            images.append(t)
            labels.append(idx)
    images = torch.stack(images)  # N,3,32,32 uint8
    labels = torch.tensor(labels, dtype=torch.long)
    # shuffle once
    perm = torch.randperm(len(labels))
    images = images[perm]
    labels = labels[perm]
    print(f"Loaded train {images.shape} {labels.shape}")
    return images, labels

def load_test_to_ram():
    p = pathlib.Path(DATA_TEST)
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    class_to_idx = {c:i for i,c in enumerate(classes)}
    images = []
    labels = []
    for c in classes:
        idx = class_to_idx[c]
        for img_path in (p/c).glob("*.png"):
            img = Image.open(img_path).convert("RGB")
            import numpy as np
            arr = np.array(img)
            t = torch.from_numpy(arr).permute(2,0,1)
            images.append(t)
            labels.append(idx)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    print(f"Loaded test {images.shape}")
    return images, labels

def gpu_augment(x, mean, std):
    # x: (B,3,32,32) float in [0,1] on GPU, already padded? We'll do random crop and flip on GPU
    B = x.size(0)
    # Random crop with padding 4: pad 4 on each side -> 40x40 then crop 32
    # Pad using F.pad
    x = F.pad(x, (4,4,4,4), mode='reflect')  # B,3,40,40
    # random crop offsets
    # generate random offsets per sample
    # Use torch.randint
    h_off = torch.randint(0, 9, (B,), device=x.device)
    w_off = torch.randint(0, 9, (B,), device=x.device)
    # Need to crop each sample differently - do via loop or vectorized?
    # Vectorized via grid? Simpler: use torch.stack with slicing via advanced indexing
    # We can use F.unfold trick or just loop in python (B=2048 loop is heavy)
    # Alternative: use batched crop via torch.nn.functional with affine? Simpler to use random offsets shared? But we want per-sample.
    # We can implement via torch.gather or use Kornia style: create batch of crops via indexing
    # Fastest: use torch.randint and then use a kernel: we can do via `torch.stack([x[i,:,h_off[i]:h_off[i]+32, w_off[i]:w_off[i]+32] for i in range(B)])` - this is python loop but on GPU tensors, still maybe okay for B=2048? 2048 iterations per batch * 25 batches = 50k iterations per epoch - too slow.
    # Instead, use vectorized approach: use `torch.nn.functional.unfold`? Or use `torch.roll`?
    # Simpler: use random crop via `torchvision.transforms.functional` on GPU? Not.
    # Alternative: do augmentation on CPU but with preloaded tensors and num_workers=0, using torch operations on CPU then transfer to GPU - still fast.
    # For now, implement efficient batched crop using `torch.arange` and advanced indexing
    # Create indices
    # x is B,3,40,40
    # We want B,3,32,32
    # Use `torch.stack` with list comprehension but with CUDA graphs maybe okay? Let's try to use `torch.vmap`?
    # Another trick: use `F.pad` + `torch.randint` to generate random shifts and use `torch.roll`? Not.
    # Let's implement using `torch.nn.functional` with `grid_sample` - generate affine grids for crop
    # Simpler: just do random flip and random erasing, skip random crop for speed? But crop is important.
    # We can do crop by using `torch.cuda` kernel via `torch.index_select`?
    # Let's try to implement batched crop using `torch.gather`:
    # Create base indices
    device = x.device
    # Create meshgrid for 32x32
    # Instead, use `torch.nn.Unfold` to extract all 9x9=81 possible crops? Not.
    # Easiest: use `torch.stack` loop but with `torch.jit`? Let's benchmark: loop 2048 times per batch might be ~5ms? Could be okay.
    # We'll try loop with `torch.stack` but using `torch` operations that are GPU-accelerated per slice (still GPU kernel per slice)
    # Alternative: use `kornia.augmentation.RandomCrop` which is GPU-optimized.
    # Let's try to use `torchvision.transforms.v2` on GPU tensors - it supports GPU tensors for some ops.
    # For now, implement simple loop but with `torch` no-grad and using `torch` slicing which is fast (view).
    crops = []
    for i in range(B):
        crops.append(x[i:i+1, :, h_off[i]:h_off[i]+32, w_off[i]:w_off[i]+32])
    x = torch.cat(crops, dim=0)
    # Random horizontal flip
    flip_mask = torch.rand(B, device=device) < 0.5
    # flip where mask true
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
    # Normalize
    x = (x - mean) / std
    # Random erasing p=0.25
    # For speed, skip erasing or do simple
    # Do erasing: for 25% of samples, erase random rectangle
    erase_mask = torch.rand(B, device=device) < 0.25
    if erase_mask.any():
        # for each erased sample, pick random area
        # Use vectorized: generate random coords
        # We'll do loop for erased samples (approx 512 per batch)
        idxs = torch.where(erase_mask)[0]
        for i in idxs:
            h = random.randint(4, 8)
            w = random.randint(4, 8)
            y1 = random.randint(0, 32 - h)
            x1 = random.randint(0, 32 - w)
            x[i, :, y1:y1+h, x1:x1+w] = 0  # after normalize, 0 corresponds to mean
    return x

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

    train_images, train_labels = load_train_to_ram()  # uint8 CPU
    test_images, test_labels = load_test_to_ram()
    # Move test to GPU for fast eval? Keep on CPU and transfer per batch
    # Pre-normalize test? We'll do on GPU per eval

    # Move mean/std to GPU
    mean_gpu = MEAN.to(device)
    std_gpu = STD.to(device)
    mean_cpu = MEAN.squeeze(0).squeeze(-1).squeeze(-1)  # 3
    std_cpu = STD.squeeze(0).squeeze(-1).squeeze(-1)

    model = Net().to(device).to(memory_format=torch.channels_last)
    # try compile
    # model = torch.compile(model)  # compile may break jit script saving; skip for now

    optimizer = torch.optim.SGD(model.parameters(), lr=LR_MAX, momentum=0.9, weight_decay=WD, nesterov=True)
    steps_per_epoch = (len(train_labels) + BATCH_TRAIN - 1)//BATCH_TRAIN
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Pin memory for faster transfer? We'll keep train_images on CPU and copy per batch to GPU
    # Use non_blocking transfer

    start_wall=time.time()
    for epoch in range(1, EPOCHS+1):
        model.train()
        t0=time.time()
        # shuffle indices each epoch
        perm = torch.randperm(len(train_labels))
        train_images_shuffled = train_images[perm]
        train_labels_shuffled = train_labels[perm]
        # iterate batches
        for i in range(0, len(train_labels), BATCH_TRAIN):
            x_cpu = train_images_shuffled[i:i+BATCH_TRAIN]  # uint8
            y_cpu = train_labels_shuffled[i:i+BATCH_TRAIN]
            # to GPU float [0,1]
            x = x_cpu.to(device, non_blocking=True).float() / 255.0
            y = y_cpu.to(device, non_blocking=True)
            # GPU augment
            x = gpu_augment(x, mean_gpu, std_gpu)

            # CutMix/MixUp
            r = random.random()
            if r < 0.30:
                lam = random.betavariate(1.0,1.0)
                idx = torch.randperm(x.size(0), device=device)
                W,H = 32,32
                cut_rat = (1-lam)**0.5
                cut_w = int(W*cut_rat); cut_h = int(H*cut_rat)
                cx = random.randint(0,W-1); cy = random.randint(0,H-1)
                bbx1 = max(cx-cut_w//2,0); bby1 = max(cy-cut_h//2,0)
                bbx2 = min(cx+cut_w//2,W); bby2 = min(cy+cut_h//2,H)
                lam = 1 - ((bbx2-bbx1)*(bby2-bby1)/(W*H))
                x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
                y_a, y_b = y, y[idx]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            elif r < 0.40:
                lam = random.betavariate(0.2,0.2)
                idx = torch.randperm(x.size(0), device=device)
                mixed_x = lam*x + (1-lam)*x[idx]
                y_a, y_b = y, y[idx]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(mixed_x)
                    loss = lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = criterion(out,y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        # eval
        model.eval()
        correct=0; total=0
        with torch.no_grad():
            for i in range(0, len(test_labels), BATCH_TEST):
                x_cpu = test_images[i:i+BATCH_TEST]
                y_cpu = test_labels[i:i+BATCH_TEST]
                x = x_cpu.to(device, non_blocking=True).float() / 255.0
                y = y_cpu.to(device, non_blocking=True)
                # normalize
                x = (x - mean_gpu) / std_gpu
                x = x.to(memory_format=torch.channels_last)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                correct += (out.argmax(1)==y).sum().item()
                total += y.size(0)
        acc = correct/total
        print(f"Epoch {epoch:03d} acc={acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.5f} time={time.time()-t0:.2f}s")
        ckpt_path=os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted=torch.jit.script(model.eval())
        scripted.save(ckpt_path)
        try:
            m2=torch.jit.load(ckpt_path, map_location=device)
            m2.eval()
            with torch.no_grad():
                xb = test_images[:2].to(device).float()/255.0
                xb = (xb - mean_gpu)/std_gpu
                xb = xb.to(memory_format=torch.channels_last)
                _=m2(xb)
        except Exception as e:
            print(f"verify fail {e}")
        model.train()

    last=os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
    if os.path.exists(last):
        shutil.copy(last, FINAL_CKPT)
    total_wall=time.time()-start_wall
    # final acc
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for i in range(0, len(test_labels), BATCH_TEST):
            x_cpu = test_images[i:i+BATCH_TEST]
            y_cpu = test_labels[i:i+BATCH_TEST]
            x = x_cpu.to(device, non_blocking=True).float()/255.0
            y = y_cpu.to(device, non_blocking=True)
            x = (x - mean_gpu)/std_gpu
            x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda'):
                out=model(x)
            correct+=(out.argmax(1)==y).sum().item()
            total+=y.size(0)
    acc=correct/total
    print(f"Final acc {acc*100:.2f}% wall {total_wall:.2f}s")
    with open("/workspace/agent_reported_metrics.json","w") as f:
        json.dump({"accuracy":acc,"wall_clock_sec":total_wall},f)

if __name__=="__main__":
    main()
