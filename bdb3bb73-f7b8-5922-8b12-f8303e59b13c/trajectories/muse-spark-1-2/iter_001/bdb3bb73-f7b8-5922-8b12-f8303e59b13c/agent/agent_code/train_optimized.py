import os, time, shutil, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
CKPT_DIR = "/workspace/periodic_checkpoints"
FINAL_CKPT = "/workspace/checkpoint.pt"

EPOCHS = 40
BATCH_TRAIN = 2048
BATCH_TEST = 512
LR_MAX = 2.0
WD = 5e-4
LS = 0.1

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

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
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
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

def preload_data():
    print("Preloading data...", flush=True)
    # Load train
    train_imgs = []
    train_labels = []
    classes = sorted(os.listdir(DATA_TRAIN))
    class_to_idx = {c:i for i,c in enumerate(classes)}
    for cls in classes:
        d = os.path.join(DATA_TRAIN, cls)
        for fname in sorted(os.listdir(d)):
            path = os.path.join(d, fname)
            img = Image.open(path).convert('RGB')
            arr = np.array(img)  # HWC 0-255
            train_imgs.append(arr)
            train_labels.append(class_to_idx[cls])
    train_imgs = np.stack(train_imgs)  # N H W C
    train_labels = np.array(train_labels)
    print(f"Train: {train_imgs.shape} {train_labels.shape}", flush=True)
    # Load test
    test_imgs = []
    test_labels = []
    for cls in classes:
        d = os.path.join(DATA_TEST, cls)
        for fname in sorted(os.listdir(d)):
            path = os.path.join(d, fname)
            img = Image.open(path).convert('RGB')
            arr = np.array(img)
            test_imgs.append(arr)
            test_labels.append(class_to_idx[cls])
    test_imgs = np.stack(test_imgs)
    test_labels = np.array(test_labels)
    print(f"Test: {test_imgs.shape} {test_labels.shape}", flush=True)
    # Convert to tensors, keep on CPU as uint8 for fast augmentation
    train_imgs_t = torch.from_numpy(train_imgs).permute(0,3,1,2)  # N C H W uint8
    test_imgs_t = torch.from_numpy(test_imgs).permute(0,3,1,2).float() / 255.0
    # Normalize test
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)
    test_imgs_t = (test_imgs_t - mean) / std
    train_labels_t = torch.from_numpy(train_labels).long()
    test_labels_t = torch.from_numpy(test_labels).long()
    return train_imgs_t, train_labels_t, test_imgs_t, test_labels_t

def augment_batch(imgs_uint8, device):
    # imgs_uint8: B C H W uint8 on CPU or GPU
    # Do augmentation on GPU for speed
    B = imgs_uint8.size(0)
    # Convert to float 0-1
    x = imgs_uint8.float() / 255.0
    # Random crop with padding 4: pad then crop
    # Pad 4 on each side -> 40x40, then random crop 32x32
    x = F.pad(x, (4,4,4,4), mode='reflect')
    # Random crop offsets
    h_off = torch.randint(0, 9, (B,), device=x.device)
    w_off = torch.randint(0, 9, (B,), device=x.device)
    # Use grid sampling or manual slicing - manual is faster with loop? Use batched indexing
    # Create cropped batch
    cropped = torch.empty(B, 3, 32, 32, device=x.device, dtype=x.dtype)
    for i in range(B):
        cropped[i] = x[i, :, h_off[i]:h_off[i]+32, w_off[i]:w_off[i]+32]
    x = cropped
    # Random horizontal flip
    flip_mask = torch.rand(B, device=x.device) < 0.5
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
    # Normalize
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1,3,1,1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=x.device).view(1,3,1,1)
    x = (x - mean) / std
    # Random erasing: 25% chance per image, erase random rectangle
    erase_mask = torch.rand(B, device=x.device) < 0.25
    for i in range(B):
        if erase_mask[i]:
            h = random.randint(4, 10)
            w = random.randint(4, 10)
            y1 = random.randint(0, 32 - h)
            x1 = random.randint(0, 32 - w)
            x[i, :, y1:y1+h, x1:x1+w] = 0  # after normalize, 0 is mean
    return x

def main():
    start_wall = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        fp = os.path.join(CKPT_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    train_imgs, train_labels, test_imgs, test_labels = preload_data()
    # Move test to GPU once
    test_imgs_gpu = test_imgs.to(device)
    test_labels_gpu = test_labels.to(device)
    # Keep train on CPU as uint8, will move batch to GPU and augment there
    train_labels_gpu_cache = train_labels  # keep on CPU, move per batch

    model = Net().to(device)
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(), lr=LR_MAX, momentum=0.9, weight_decay=WD, nesterov=True)
    # OneCycle
    steps_per_epoch = (len(train_imgs) + BATCH_TRAIN - 1) // BATCH_TRAIN
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100
    )
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(label_smoothing=LS)

    best_acc = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        # Shuffle indices
        perm = torch.randperm(len(train_imgs))
        train_imgs_shuffled = train_imgs[perm]
        train_labels_shuffled = train_labels[perm]
        for i in range(0, len(train_imgs), BATCH_TRAIN):
            batch_imgs = train_imgs_shuffled[i:i+BATCH_TRAIN].to(device, non_blocking=True)
            batch_labels = train_labels_shuffled[i:i+BATCH_TRAIN].to(device, non_blocking=True)
            # Augment on GPU
            batch_imgs = augment_batch(batch_imgs, device)
            batch_imgs = batch_imgs.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(test_imgs_gpu), BATCH_TEST):
                inputs = test_imgs_gpu[i:i+BATCH_TEST].to(memory_format=torch.channels_last)
                targets = test_labels_gpu[i:i+BATCH_TEST]
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                _, pred = outputs.max(1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch}/{EPOCHS} acc={acc*100:.2f}% best={best_acc*100:.2f}% lr={optimizer.param_groups[0]['lr']:.5f}", flush=True)

        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
        scripted = torch.jit.script(model.eval())
        scripted.save(ckpt_path)
        model.train()

    last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
    shutil.copy(last_ckpt, FINAL_CKPT)
    end_wall = time.time()
    wall_clock = end_wall - start_wall
    print(f"Done. Best {best_acc*100:.2f}% wall {wall_clock:.1f}s", flush=True)
    import json
    with open("/workspace/agent_reported_metrics.json", "w") as f:
        json.dump({"accuracy": best_acc, "wall_clock_sec": wall_clock}, f)

if __name__ == "__main__":
    main()
