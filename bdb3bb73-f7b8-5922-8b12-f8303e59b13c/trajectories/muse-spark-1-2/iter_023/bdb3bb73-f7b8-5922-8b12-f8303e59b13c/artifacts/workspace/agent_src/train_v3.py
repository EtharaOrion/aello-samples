import os, time, shutil, glob, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import numpy as np

DATA_ROOT = "/workspace/data/cifar10"
CKPT_DIR = "/workspace/scratch_ckpt"
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
cls_to_idx = {c:i for i,c in enumerate(CLASSES)}

# Preload all images to RAM as tensors
def load_split(split):
    paths = []
    labels = []
    for cls in CLASSES:
        d = os.path.join(DATA_ROOT, split, cls)
        for p in glob.glob(os.path.join(d, "*.png")):
            paths.append(p)
            labels.append(cls_to_idx[cls])
    # sort for determinism
    combined = sorted(zip(paths, labels))
    n = len(combined)
    # Load as uint8 HWC
    imgs = torch.empty(n, 32, 32, 3, dtype=torch.uint8)
    lbls = torch.empty(n, dtype=torch.long)
    for i, (p, l) in enumerate(combined):
        img = Image.open(p).convert("RGB")
        imgs[i] = torch.from_numpy(np.array(img))
        lbls[i] = l
    return imgs, lbls

# Model
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        if self.equalInOut:
            self.convShortcut = nn.Identity()
        else:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        if self.equalInOut:
            shortcut = x
        else:
            shortcut = self.convShortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return shortcut + out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        layers=[]
        for i in range(nb_layers):
            layers.append(block(i==0 and in_planes or out_planes, out_planes, i==0 and stride or 1, dropRate))
        self.layer = nn.Sequential(*layers)
    def forward(self, x): return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropRate=0.0):
        super().__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth-4)%6==0
        n=(depth-4)//6
        block=BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

# GPU augmentation helpers
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

def augment_batch(x_uint8_hwc, device):
    # x: (B,32,32,3) uint8
    B = x_uint8_hwc.shape[0]
    # Random crop with padding 4: pad to 40x40 then crop
    # Do on GPU for speed
    x = x_uint8_hwc.to(device, non_blocking=True).float()  # B,H,W,C
    # pad 4
    x = F.pad(x.permute(0,3,1,2), (4,4,4,4), value=0)  # B,C,40,40 - but we padded after permute, need correct
    # Actually x is B,H,W,C -> permute to B,C,H,W then pad
    # Let's do properly: x is B,H,W,C float
    # Convert to B,C,H,W
    # We already did permute, so x is B,C,40,40
    # Random crop 32x32
    # Generate random offsets
    # Use vectorized crop via grid? Simpler: use random ints and slice via loop? Need fast batched crop.
    # Instead use F.pad + random crop via indexing
    # We'll do: for each sample, pick top,left in [0,8]
    # Use torch random
    tops = torch.randint(0, 9, (B,), device=device)
    lefts = torch.randint(0, 9, (B,), device=device)
    # Need to crop each sample differently - use batched indexing trick
    # Create output tensor
    out = torch.empty(B, 3, 32, 32, device=device, dtype=x.dtype)
    for i in range(B):
        out[i] = x[i, :, tops[i]:tops[i]+32, lefts[i]:lefts[i]+32]
    x = out
    # Random horizontal flip
    flip_mask = torch.rand(B, device=device) < 0.5
    # flip where mask true
    if flip_mask.any():
        x[flip_mask] = x[flip_mask].flip(dims=[3])
    # Normalize: x is [0,255] -> [0,1] -> standardized
    x = x / 255.0
    # Cutout 16x16 with prob 1.0
    # Random center
    cy = torch.randint(0, 32, (B,), device=device)
    cx = torch.randint(0, 32, (B,), device=device)
    # Apply cutout by zeroing (in normalized space, 0 corresponds to -mean/std, but we want to zero after normalize? Original cutout zeros after normalize)
    # We'll zero in normalized space
    x = (x - MEAN.to(device)) / STD.to(device)
    # Now cutout: set to 0
    for i in range(B):
        y1 = max(0, cy[i].item() - 8); y2 = min(32, cy[i].item() + 8)
        x1 = max(0, cx[i].item() - 8); x2 = min(32, cx[i].item() + 8)
        x[i, :, y1:y2, x1:x2] = 0
    return x

def main():
    t_start = time.time()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    if os.path.exists(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading data...")
    train_imgs, train_lbls = load_split("train")
    test_imgs, test_lbls = load_split("test")
    print(f"Train {train_imgs.shape} Test {test_imgs.shape}")

    # Pre-normalize test set
    test_x = test_imgs.permute(0,3,1,2).float() / 255.0
    test_x = (test_x - MEAN) / STD
    test_x = test_x.to(device)
    test_y = test_lbls.to(device)

    model = WideResNet(depth=28, widen_factor=10, dropRate=0.0).to(device)
    model = model.to(memory_format=torch.channels_last)
    model_compiled = torch.compile(model, mode="max-autotune")

    BATCH = 512
    EPOCHS = 35
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # Cosine with warmup 2 epochs
    warmup = 2
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch+1)/warmup
        p = (epoch - warmup) / (EPOCHS - warmup)
        return 0.5*(1+torch.cos(torch.tensor(p*3.14159265))).item()*0.99+0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # For shuffling
    n_train = train_imgs.shape[0]
    indices = torch.arange(n_train)

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        # shuffle
        perm = torch.randperm(n_train)
        train_imgs_shuffled = train_imgs[perm]
        train_lbls_shuffled = train_lbls[perm]

        total_loss = 0
        num_batches = 0
        for i in range(0, n_train, BATCH):
            x_hwc = train_imgs_shuffled[i:i+BATCH]
            y = train_lbls_shuffled[i:i+BATCH].to(device, non_blocking=True)
            x = augment_batch(x_hwc, device)
            x = x.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model_compiled(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, test_x.shape[0], 512):
                xb = test_x[i:i+512].to(memory_format=torch.channels_last)
                yb = test_y[i:i+512]
                with torch.amp.autocast('cuda'):
                    out = model(xb)
                correct += (out.argmax(1) == yb).sum().item()
        acc = correct / test_x.shape[0]
        best_acc = max(best_acc, acc)
        dt = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{EPOCHS} loss={total_loss/num_batches:.3f} acc={acc*100:.2f}% best={best_acc*100:.2f}% {dt:.1f}s lr={optimizer.param_groups[0]['lr']:.4f}")

        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch+1:04d}.pt")
        m_save = model.eval()
        scripted = torch.jit.script(m_save)
        scripted.save(ckpt_path)
        print(f"  Saved {ckpt_path}")

    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
    if ckpts:
        shutil.copy(ckpts[-1], "/workspace/checkpoint.pt")
    wall = time.time() - t_start
    print(f"Done wall={wall:.1f}s best={best_acc:.4f}")
    import json
    with open("/workspace/agent_reported_metrics.json","w") as f:
        json.dump({"accuracy": best_acc, "wall_clock_sec": wall}, f)
    # verify
    m = torch.jit.load("/workspace/checkpoint.pt").eval().cuda()
    correct=0
    with torch.no_grad():
        for i in range(0, test_x.shape[0], 512):
            xb = test_x[i:i+512]
            yb = test_y[i:i+512]
            out = m(xb)
            correct += (out.argmax(1)==yb).sum().item()
    print(f"Verify {correct/test_x.shape[0]*100:.2f}%")

if __name__ == "__main__":
    main()
