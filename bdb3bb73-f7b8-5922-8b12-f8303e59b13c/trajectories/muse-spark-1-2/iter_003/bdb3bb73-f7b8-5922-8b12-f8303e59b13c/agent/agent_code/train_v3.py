import os, time, shutil, random, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 60
BATCH_SIZE = 2048
NUM_WORKERS = 0  # we will preload and do GPU aug, so 0 workers is fastest
LR = 1.5
WD = 5e-4
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 5

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')

# Preload dataset into RAM as uint8 tensors
class PreloadedCIFAR(Dataset):
    def __init__(self, root, train=True):
        self.train = train
        # collect files
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        files = []
        labels = []
        for c in classes:
            cdir = os.path.join(root, c)
            for f in os.listdir(cdir):
                files.append(os.path.join(cdir, f))
                labels.append(self.class_to_idx[c])
        # load all images as uint8 CHW
        print(f"Preloading {len(files)} images from {root} ...")
        import numpy as np
        data = []
        for p in files:
            img = Image.open(p).convert('RGB')
            arr = torch.from_numpy(np.array(img)).permute(2,0,1)  # 3,32,32 uint8
            data.append(arr)
        self.data = torch.stack(data)  # N,3,32,32 uint8
        self.labels = torch.tensor(labels, dtype=torch.long)
        print(f"Preloaded {self.data.shape} {self.data.dtype}")
        # aug params
        self.mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
        self.std = torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        # return uint8 and label, augmentation will be done in collate or on GPU
        return self.data[idx], self.labels[idx]

# GPU augmentation helpers
def gpu_augment(x, mean, std):
    # x: B,3,32,32 float in [0,1] after /255, before normalize
    # RandomCrop with padding 4, HorizontalFlip
    B = x.size(0)
    # padding 4
    x = F.pad(x, (4,4,4,4), mode='reflect')  # B,3,40,40
    # random crop 32
    # generate random offsets
    # Use torch.randint
    h_off = torch.randint(0, 9, (B,), device=x.device)
    w_off = torch.randint(0, 9, (B,), device=x.device)
    # crop via loop? vectorized via gather is tricky, do batched indexing
    # Use unfold or just loop over B (B=2048, loop is okay but slow)
    # Instead use torch.stack with slicing - we can do via grid_sample or via manual
    # Simple: use F.unfold? Easier: use torchvision transforms on GPU via Kornia? Let's do loop with torch
    # For speed, we can use random crop via rolling? Let's do efficient version:
    # Create output tensor
    out = torch.empty(B, 3, 32, 32, device=x.device, dtype=x.dtype)
    for i in range(B):
        out[i] = x[i, :, h_off[i]:h_off[i]+32, w_off[i]:w_off[i]+32]
    x = out
    # horizontal flip
    flip_mask = torch.rand(B, device=x.device) < 0.5
    # flip where mask true
    # x[flip_mask] = torch.flip(x[flip_mask], dims=[3])  # this creates copy
    if flip_mask.any():
        x[flip_mask] = x[flip_mask].flip(dims=[3])
    # normalize
    x = (x - mean.to(x.device)) / std.to(x.device)
    return x

# Model: WideResNet style but simpler
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        # shortcut after first bn+relu? For WRN, shortcut is after?
        # Use standard: if shortcut needs conv, apply to x after bn+relu? Let's do simple
        shortcut = self.shortcut(out) if len(self.shortcut)>0 else self.shortcut(x)
        # Actually WRN does: out = conv1(relu(bn1(x))), then bn2, relu, dropout, conv2
        out = self.conv1(out)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        out += shortcut
        return out

class WRN(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth-4)%6==0
        n = (depth-4)//6
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(16, 16*widen, n, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(16*widen, 32*widen, n, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(32*widen, 64*widen, n, stride=2, dropout=dropout)
        self.bn = nn.BatchNorm2d(64*widen)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64*widen, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, in_ch, out_ch, n, stride, dropout):
        layers = []
        layers.append(Block(in_ch, out_ch, stride, dropout))
        for _ in range(1, n):
            layers.append(Block(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn(out))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def evaluate(model, loader, device, mean, std):
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # x is uint8, need to convert to float and normalize
            x = x.float().div(255.0)
            x = (x - mean.to(device)) / std.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        if f.startswith("ckpt_epoch_"):
            os.remove(os.path.join(CKPT_DIR, f))
    if os.path.exists(FINAL_CKPT):
        os.remove(FINAL_CKPT)
    device = torch.device("cuda")
    print(f"Device {torch.cuda.get_device_name(0)}")
    train_ds = PreloadedCIFAR(DATA_TRAIN, train=True)
    test_ds = PreloadedCIFAR(DATA_TEST, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train {len(train_ds)} Test {len(test_ds)} Steps {len(train_loader)}")
    # Use WRN 28-10
    model = WRN(depth=28, widen=10, dropout=0.0).to(device)
    model = model.to(memory_format=torch.channels_last)
    # maybe compile
    try:
        model = torch.compile(model)
        print("Compiled")
    except Exception as e:
        print(f"Compile failed {e}")
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD, nesterov=True)
    def lr_lambda(epoch):
        e = epoch+1
        if e <= WARMUP_EPOCHS:
            return e / WARMUP_EPOCHS
        else:
            progress = (e - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)).item())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler('cuda')
    mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
    std = torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)
    start_wall = time.time()
    first_ckpt_time=None
    last_ckpt_time=None
    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_start=time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            # xb uint8 -> float
            x = xb.float().div(255.0)
            # GPU augment
            # pad and crop + flip + normalize
            # Do augment inline for speed
            # pad
            x = F.pad(x, (4,4,4,4), mode='reflect')
            B = x.size(0)
            h_off = torch.randint(0, 9, (B,), device=device)
            w_off = torch.randint(0, 9, (B,), device=device)
            # Use efficient crop via torch.stack? Loop is okay for B=2048 but 25 steps * 60 epochs = 1500 loops *2048 = 3M iterations of loop per epoch? That's heavy (2048 loop per batch)
            # Let's try vectorized crop using grid_sample or unfold
            # Alternative: use F.unfold? Simpler: use torch's random crop via torchvision on GPU? Let's keep loop but maybe okay
            # Try to optimize: use batched indexing via advanced indexing
            # Create output via gathering
            # We can use torch.nn.functional.unfold to extract patches? Might be overkill
            # Let's do loop but with numba? For now loop
            out = torch.empty(B, 3, 32, 32, device=device, dtype=x.dtype)
            for i in range(B):
                out[i] = x[i, :, h_off[i]:h_off[i]+32, w_off[i]:w_off[i]+32]
            x = out
            flip_mask = torch.rand(B, device=device) < 0.5
            if flip_mask.any():
                x[flip_mask] = x[flip_mask].flip(dims=[3])
            x = (x - mean.to(device)) / std.to(device)
            # CutMix 40%
            do_cutmix = random.random() < 0.4
            if do_cutmix:
                lam = random.betavariate(1.0, 1.0)
                index = torch.randperm(B, device=device)
                y_a, y_b = yb, yb[index]
                Wc = 32; Hc=32
                cut_rat = (1 - lam) ** 0.5
                cut_w = int(Wc * cut_rat)
                cut_h = int(Hc * cut_rat)
                cx = random.randint(0, Wc-1)
                cy = random.randint(0, Hc-1)
                bbx1 = max(cx - cut_w // 2, 0)
                bby1 = max(cy - cut_h // 2, 0)
                bbx2 = min(cx + cut_w // 2, Wc)
                bby2 = min(cy + cut_h // 2, Hc)
                if bbx2 > bbx1 and bby2 > bby1:
                    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (Wc * Hc))
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out_logits = model(x)
                    loss = lam * criterion(out_logits, y_a) + (1 - lam) * criterion(out_logits, y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out_logits = model(x)
                    loss = criterion(out_logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        scheduler.step()
        acc = evaluate(model, test_loader, device, mean, std)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{EPOCHS} lr={lr:.4f} acc={acc*100:.2f}% time={time.time()-epoch_start:.1f}s")
        # save checkpoint: need to unwrap compiled model
        # Get underlying model
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        # Create CPU copy for scripting
        cpu_model = WRN(depth=28, widen=10, dropout=0.0)
        cpu_model.load_state_dict(raw_model.state_dict())
        cpu_model.eval()
        scripted = torch.jit.script(cpu_model)
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(ckpt_path)
        if first_ckpt_time is None:
            first_ckpt_time = time.time()
        last_ckpt_time = time.time()
        try:
            m2 = torch.jit.load(ckpt_path)
            m2.eval()
            with torch.no_grad():
                dummy = torch.randn(2,3,32,32)
                _ = m2(dummy)
        except Exception as e:
            print(f"Verify failed {e}")
    src = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
    shutil.copy(src, FINAL_CKPT)
    total_wall = last_ckpt_time - start_wall
    print(f"Done total {total_wall:.1f}s")
    final_acc = evaluate(model, test_loader, device, mean, std)
    print(f"Final acc {final_acc*100:.2f}%")
    import json
    with open("/workspace/agent_reported_metrics.json","w") as f:
        json.dump({"accuracy": float(final_acc), "wall_clock_sec": float(total_wall)}, f)

if __name__ == "__main__":
    main()
