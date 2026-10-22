import os, time, shutil, glob, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/scratch_ckpt"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

BATCH = 1024
EPOCHS = 40
LR = 1.2
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 64, 2, 1)
        self.layer2 = self._make(64, 128, 2, 2)
        self.layer3 = self._make(128, 256, 2, 2)
        self.layer4 = self._make(256, 512, 2, 2)
        self.fc = nn.Linear(512, num_classes)
        self._init()
    def _make(self, in_ch, out_ch, n, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    def _init(self):
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def load_folder(folder):
    classes = sorted(os.listdir(folder))
    class_to_idx = {c:i for i,c in enumerate(classes)}
    imgs = []
    labels = []
    for c in classes:
        idx = class_to_idx[c]
        for f in glob.glob(os.path.join(folder, c, "*.png")):
            img = Image.open(f).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            imgs.append(arr)
            labels.append(idx)
    data = np.stack(imgs)  # N H W C
    labels = np.array(labels, dtype=np.int64)
    data = torch.from_numpy(data).permute(0,3,1,2).contiguous()  # N C H W uint8
    labels = torch.from_numpy(labels)
    return data, labels

def gpu_augment_fast(x, mean, std):
    # x: B C 32 32 float [0,1]
    B = x.size(0)
    # pad 4 reflect
    x = F.pad(x, (4,4,4,4), mode='reflect')  # B C 40 40
    # random crop via unfold+gather
    # unfold
    # x_unfold: B, C*1024, 81
    x_unfold = F.unfold(x, kernel_size=32, stride=1)  # B, C*1024, 81
    h_off = torch.randint(0, 9, (B,), device=x.device)
    w_off = torch.randint(0, 9, (B,), device=x.device)
    idx = h_off * 9 + w_off  # B
    # gather
    C = x.size(1)
    # x_unfold shape B, C*1024, 81
    idx_exp = idx.view(B, 1, 1).expand(B, C*32*32, 1)
    x_crop_flat = torch.gather(x_unfold, 2, idx_exp)  # B, C*1024, 1
    x = x_crop_flat.view(B, C, 32, 32)
    # flip
    flip = torch.rand(B, device=x.device) < 0.5
    if flip.any():
        x[flip] = torch.flip(x[flip], dims=[3])
    # normalize
    x = (x - mean) / std
    return x

def evaluate(model, test_data, test_labels, device, batch=512):
    model.eval()
    correct = 0
    total = 0
    mean = MEAN.to(device)
    std = STD.to(device)
    with torch.no_grad():
        for i in range(0, len(test_data), batch):
            x = test_data[i:i+batch].to(device, non_blocking=True).float() / 255.0
            y = test_labels[i:i+batch].to(device, non_blocking=True)
            x = (x - mean) / std
            x = x.to(memory_format=torch.channels_last)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct/total

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        os.remove(os.path.join(CKPT_DIR, f))
    device = torch.device("cuda")
    print("Loading data...")
    train_data, train_labels = load_folder(DATA_TRAIN)
    test_data, test_labels = load_folder(DATA_TEST)
    print(f"Train {train_data.shape} Test {test_data.shape}")
    # keep train_data on CPU, will move per batch; but we can also keep on GPU for speed
    # Let's keep on GPU as uint8 to avoid H2D copy per batch? 150MB fits.
    # Move to GPU once
    train_data_gpu = train_data.to(device)  # N C H W uint8 on GPU
    train_labels_gpu = train_labels.to(device)
    test_data_gpu = test_data  # keep CPU for eval, but eval will move
    # Actually keep test on CPU too

    model = ResNet18().to(device)
    model = model.to(memory_format=torch.channels_last)
    # compile for speed
    model_compiled = torch.compile(model, mode="max-autotune")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
    warmup = 5
    def lr_lambda(epoch):
        e = epoch+1
        if e <= warmup:
            return e / warmup
        prog = (e - warmup) / (EPOCHS - warmup)
        return 0.5 * (1 + np.cos(prog * np.pi))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')
    mean = MEAN.to(device)
    std = STD.to(device)

    start_wall = time.time()
    first_ckpt = None
    last_ckpt = None
    best = 0
    crossing = None

    N = len(train_data_gpu)
    indices = torch.arange(N, device=device)

    for epoch in range(1, EPOCHS+1):
        model.train()
        # shuffle indices on GPU
        perm = torch.randperm(N, device=device)
        # we will iterate over perm in chunks
        epoch_start = time.time()
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            if len(idx) < BATCH:
                continue
            x = train_data_gpu[idx].float() / 255.0  # B C H W
            y = train_labels_gpu[idx]
            # augment
            x = gpu_augment_fast(x, mean, std)
            # cutmix 50%
            do_cutmix = random.random() < 0.5
            if do_cutmix:
                lam = np.random.beta(1.0, 1.0)
                perm2 = torch.randperm(x.size(0), device=device)
                y_a, y_b = y, y[perm2]
                W = 32; H = 32
                cut_rat = np.sqrt(1. - lam)
                cut_w = int(W * cut_rat)
                cut_h = int(H * cut_rat)
                cx = np.random.randint(W)
                cy = np.random.randint(H)
                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)
                if bbx2 > bbx1 and bby2 > bby1:
                    x[:, :, bby1:bby2, bbx1:bbx2] = x[perm2, :, bby1:bby2, bbx1:bbx2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W*H))
                else:
                    lam = 1.0
                x = x.to(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model_compiled(x)
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                x = x.to(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model_compiled(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        scheduler.step()
        # save ckpt - need to use non-compiled model state
        cpu_model = ResNet18()
        cpu_model.load_state_dict(model.state_dict())
        cpu_model.eval()
        scripted = torch.jit.script(cpu_model)
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted.save(ckpt_path)
        if first_ckpt is None:
            first_ckpt = time.time()
        last_ckpt = time.time()
        acc = evaluate(model, test_data, test_labels, device)
        if acc > best:
            best = acc
        if crossing is None and acc >= 0.94:
            crossing = epoch
        print(f"Epoch {epoch:02d}/{EPOCHS} acc {acc*100:.2f} best {best*100:.2f} lr {optimizer.param_groups[0]['lr']:.4f} time {time.time()-epoch_start:.2f}s crossing {crossing}")
        if epoch >= 30 and best >= 0.94 and crossing is not None and epoch >= crossing+2:
            print("Early stop")
            break

    total_wall = last_ckpt - start_wall
    span = last_ckpt - first_ckpt if first_ckpt else total_wall
    print(f"Done best {best*100:.2f} crossing {crossing} total {total_wall:.1f} span {span:.1f}")
    with open("/tmp/metrics.json","w") as f:
        json.dump({"accuracy":best, "wall_clock_sec":total_wall, "crossing":crossing}, f)

if __name__ == "__main__":
    main()
