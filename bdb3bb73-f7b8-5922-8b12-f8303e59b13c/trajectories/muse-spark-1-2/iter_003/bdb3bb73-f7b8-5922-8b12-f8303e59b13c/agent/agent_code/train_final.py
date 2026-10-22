import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"
EPOCHS = 35
BATCH_SIZE = 2048
LR = 2.0
WD = 5e-4
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 3

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

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
        return F.relu(out)

class FastResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 3, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        layers.append(Block(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(Block(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def preload(root):
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
    class_to_idx = {c:i for i,c in enumerate(classes)}
    files = []
    labels = []
    for c in classes:
        cdir = os.path.join(root, c)
        for f in sorted(os.listdir(cdir)):
            files.append(os.path.join(cdir, f))
            labels.append(class_to_idx[c])
    data = []
    for p in files:
        img = Image.open(p).convert('RGB')
        arr = np.array(img)
        data.append(torch.from_numpy(arr).permute(2,0,1))
    data = torch.stack(data)
    labels = torch.tensor(labels, dtype=torch.long)
    return data, labels

def evaluate(model, data, labels, device, mean, std, batch_size=512):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            x = data[i:i+batch_size].to(device, non_blocking=True).float().div(255.0)
            y = labels[i:i+batch_size].to(device, non_blocking=True)
            x = (x - mean) / std
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
    train_data, train_labels = preload(DATA_TRAIN)
    test_data, test_labels = preload(DATA_TEST)
    print(f"Train {train_data.shape} Test {test_data.shape}")
    mean = torch.tensor([0.4914,0.4822,0.4465], device=device).view(1,3,1,1)
    std = torch.tensor([0.2470,0.2435,0.2616], device=device).view(1,3,1,1)
    train_data_gpu = train_data.to(device)
    train_labels_gpu = train_labels.to(device)
    model = FastResNet().to(device)
    model = model.to(memory_format=torch.channels_last)
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
    N = len(train_labels)
    start_wall = time.time()
    first_ckpt_time=None
    last_ckpt_time=None
    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_start=time.time()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            x = train_data_gpu[idx].float().div(255.0)
            y = train_labels_gpu[idx]
            B = x.size(0)
            x_padded = F.pad(x, (4,4,4,4), mode='reflect')
            patches = F.unfold(x_padded, kernel_size=32)
            h_off = torch.randint(0, 9, (B,), device=device)
            w_off = torch.randint(0, 9, (B,), device=device)
            sel_idx = h_off*9 + w_off
            selected = patches.gather(2, sel_idx.view(B,1,1).expand(B, patches.size(1), 1)).squeeze(2)
            x = selected.view(B,3,32,32)
            flip_mask = torch.rand(B, device=device) < 0.5
            if flip_mask.any():
                x[flip_mask] = x[flip_mask].flip(dims=[3])
            x = (x - mean) / std
            do_cutmix = random.random() < 0.5
            if do_cutmix:
                lam = random.betavariate(1.0, 1.0)
                perm2 = torch.randperm(B, device=device)
                y_a, y_b = y, y[perm2]
                Wc=32; Hc=32
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
                    x[:, :, bby1:bby2, bbx1:bbx2] = x[perm2, :, bby1:bby2, bbx1:bbx2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (Wc * Hc))
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        scheduler.step()
        acc = evaluate(model, test_data, test_labels, device, mean, std)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{EPOCHS} lr={lr:.4f} acc={acc*100:.2f}% time={time.time()-epoch_start:.1f}s")
        cpu_model = FastResNet()
        cpu_model.load_state_dict(model.state_dict())
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
    final_acc = evaluate(model, test_data, test_labels, device, mean, std)
    print(f"Final acc {final_acc*100:.2f}%")
    import json
    with open("/workspace/agent_reported_metrics.json","w") as f:
        json.dump({"accuracy": float(final_acc), "wall_clock_sec": float(total_wall)}, f)

if __name__ == "__main__":
    main()
