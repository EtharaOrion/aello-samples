import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/scratch_ckpt"
EPOCHS = 20
BATCH_TRAIN = 4096
BATCH_TEST = 1024
LR_MAX = 2.0
WD = 5e-4
LS = 0.1
device = torch.device("cuda")

class Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 128, 2, 1)
        self.layer2 = self._make(128, 256, 2, 2)
        self.layer3 = self._make(256, 512, 2, 2)
        self.fc = nn.Linear(512, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make(self, in_c, out_c, n, stride):
        layers = [Block(in_c, out_c, stride)]
        for _ in range(1, n):
            layers.append(Block(out_c, out_c, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def preload():
    print("Preloading...", flush=True)
    classes = sorted(os.listdir(DATA_TRAIN))
    c2i = {c:i for i,c in enumerate(classes)}
    train_imgs = []
    train_labels = []
    for cls in classes:
        d = os.path.join(DATA_TRAIN, cls)
        for fname in sorted(os.listdir(d)):
            train_imgs.append(np.array(Image.open(os.path.join(d, fname)).convert('RGB')))
            train_labels.append(c2i[cls])
    train_imgs = np.stack(train_imgs)
    train_labels = np.array(train_labels)
    test_imgs = []
    test_labels = []
    for cls in classes:
        d = os.path.join(DATA_TEST, cls)
        for fname in sorted(os.listdir(d)):
            test_imgs.append(np.array(Image.open(os.path.join(d, fname)).convert('RGB')))
            test_labels.append(c2i[cls])
    test_imgs = np.stack(test_imgs)
    test_labels = np.array(test_labels)
    print(f"Train {train_imgs.shape} Test {test_imgs.shape}", flush=True)
    train_t = torch.from_numpy(train_imgs).permute(0,3,1,2).contiguous()
    test_t = torch.from_numpy(test_imgs).permute(0,3,1,2).float() / 255.0
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)
    test_t = (test_t - mean) / std
    return train_t, torch.from_numpy(train_labels).long(), test_t, torch.from_numpy(test_labels).long()

def augment_gpu(imgs_uint8):
    B = imgs_uint8.size(0)
    x = imgs_uint8.float() / 255.0
    x = F.pad(x, (4,4,4,4), mode='reflect')
    patches = F.unfold(x, kernel_size=32, stride=1)
    h_off = torch.randint(0, 9, (B,), device=x.device)
    w_off = torch.randint(0, 9, (B,), device=x.device)
    idx = h_off * 9 + w_off
    idx_exp = idx.view(B, 1, 1).expand(B, patches.size(1), 1)
    selected = patches.gather(2, idx_exp).squeeze(2)
    x = selected.view(B, 3, 32, 32)
    flip = torch.rand(B, device=x.device) < 0.5
    if flip.any():
        flipped = torch.flip(x, dims=[3])
        mask = flip.view(B,1,1,1)
        x = torch.where(mask, flipped, x)
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1,3,1,1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=x.device).view(1,3,1,1)
    x = (x - mean) / std
    erase = torch.rand(B, device=x.device) < 0.25
    if erase.any():
        erase_idx = torch.where(erase)[0]
        for i in erase_idx:
            y1 = random.randint(0, 24)
            x1 = random.randint(0, 24)
            x[i, :, y1:y1+8, x1:x1+8] = 0
    return x

def main():
    start_wall = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        fp = os.path.join(CKPT_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)
    train_imgs, train_labels, test_imgs, test_labels = preload()
    train_g = train_imgs.cuda()
    train_labels_g = train_labels.cuda()
    test_g = test_imgs.cuda()
    test_labels_g = test_labels.cuda()
    model = Net().cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_MAX, momentum=0.9, weight_decay=WD, nesterov=True)
    steps_per_epoch = (len(train_g) + BATCH_TRAIN - 1) // BATCH_TRAIN
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=LS)
    best_acc = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        perm = torch.randperm(len(train_g), device=train_g.device)
        train_g_p = train_g[perm]
        train_labels_p = train_labels_g[perm]
        for i in range(0, len(train_g), BATCH_TRAIN):
            b_imgs = train_g_p[i:i+BATCH_TRAIN]
            b_labels = train_labels_p[i:i+BATCH_TRAIN]
            b_imgs = augment_gpu(b_imgs)
            b_imgs = b_imgs.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(b_imgs)
                loss = criterion(out, b_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(test_g), BATCH_TEST):
                inp = test_g[i:i+BATCH_TEST].to(memory_format=torch.channels_last)
                tgt = test_labels_g[i:i+BATCH_TEST]
                with torch.amp.autocast('cuda'):
                    out = model(inp)
                _, pred = out.max(1)
                correct += pred.eq(tgt).sum().item()
                total += tgt.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch}/{EPOCHS} acc={acc*100:.2f}% best={best_acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.4f}", flush=True)
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted = torch.jit.script(model.eval())
        scripted.save(ckpt_path)
        model.train()
    end_wall = time.time()
    wall = end_wall - start_wall
    print(f"Done best {best_acc*100:.2f}% wall {wall:.1f}s", flush=True)
    import json
    with open("/tmp/scratch_metrics.json", "w") as f:
        json.dump({"accuracy": best_acc, "wall_clock_sec": wall}, f)

if __name__ == "__main__":
    main()
