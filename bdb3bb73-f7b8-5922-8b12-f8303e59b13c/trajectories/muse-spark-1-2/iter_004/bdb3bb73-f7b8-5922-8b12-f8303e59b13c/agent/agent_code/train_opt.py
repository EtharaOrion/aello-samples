import os, time, shutil, random, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

CKPT_DIR = "/workspace/scratch_ckpt"
FINAL_CKPT = "/workspace/scratch_final.pt"
DATA_TRAIN = "/workspace/data/cifar10/train"
DATA_TEST = "/workspace/data/cifar10/test"
BATCH = 2048
EPOCHS = 40
LR = 1.2
WD = 5e-4
MOM = 0.9
LABEL_SMOOTH = 0.1

torch.backends.cudnn.benchmark = True
device = torch.device("cuda")

# --- Model ---
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)
        else:
            self.convShortcut = nn.Identity()
    def forward(self, x):
        if self.equalInOut:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
            out = self.conv2(out)
            return x + out
        else:
            out = self.relu1(self.bn1(x))
            shortcut = self.convShortcut(out)
            out = self.conv1(out)
            out = self.relu2(self.bn2(out))
            out = self.conv2(out)
            return shortcut + out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride):
        super().__init__()
        layers=[]
        for i in range(nb_layers):
            layers.append(BasicBlock(i==0 and in_planes or out_planes, out_planes, i==0 and stride or 1))
        self.layer = nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, num_classes=10):
        super().__init__()
        nChannels = [16, 16*widen, 32*widen, 64*widen]
        assert (depth-4)%6==0
        n = (depth-4)//6
        self.conv1 = nn.Conv2d(3, nChannels[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], 2)
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

# --- Preload data to RAM as uint8 tensors ---
def load_folder(root):
    # root has subfolders 0-9 or class names, ImageFolder sorts alphabetically
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
    class_to_idx = {c:i for i,c in enumerate(classes)}
    print(f"Classes: {classes} -> {class_to_idx}")
    images=[]
    labels=[]
    for cls in classes:
        folder = os.path.join(root, cls)
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        for f in files:
            img = Image.open(f).convert("RGB")
            arr = np.array(img, dtype=np.uint8)  # HWC
            images.append(arr)
            labels.append(class_to_idx[cls])
    images = np.stack(images)  # N H W C
    labels = np.array(labels, dtype=np.int64)
    print(f"Loaded {root}: {images.shape} {labels.shape}")
    return torch.from_numpy(images), torch.from_numpy(labels)

print("Loading data...")
train_images, train_labels = load_folder(DATA_TRAIN)
test_images, test_labels = load_folder(DATA_TEST)
# train_images: N H W C uint8, test same
# Move test to GPU pre-normalized for fast eval
mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1,3,1,1)

# Pre-normalize test set on GPU
test_images_gpu = test_images.permute(0,3,1,2).float().div(255.0).to(device)
test_images_gpu = (test_images_gpu - mean) / std
test_labels_gpu = test_labels.to(device)

# For training, keep on CPU as uint8, will batch and augment on GPU
train_images_cpu = train_images  # N H W C uint8
train_labels_cpu = train_labels

# Create indices dataset for fast batching
N = train_images_cpu.shape[0]

model = WideResNet(depth=28, widen=10).to(device)
model = model.to(memory_format=torch.channels_last)
# compile for speed
try:
    model = torch.compile(model, mode="max-autotune")
    print("compile ok")
except Exception as e:
    print(f"compile fail {e}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
warmup_epochs = 3
scaler = torch.amp.GradScaler('cuda')

# GPU augmentation helpers
def augment_batch(x):  # x: B C H W float in [0,1] before normalize, on GPU
    B, C, H, W = x.shape
    x_padded = F.pad(x, (4,4,4,4), mode='reflect')  # B C 40 40
    h_off = torch.randint(0, 9, (B,), device=x.device)
    w_off = torch.randint(0, 9, (B,), device=x.device)
    # vectorized crop via unfold
    # x_padded: B C 40 40 -> unfold to B C 9 9 32 32
    x_unfold = x_padded.unfold(2, 32, 1).unfold(3, 32, 1)  # B C 9 9 32 32
    # index
    b_idx = torch.arange(B, device=x.device)
    # x_unfold[b_idx, :, h_off, w_off] -> B C 32 32
    # Need to handle C dimension: use advanced indexing
    # x_unfold shape B C 9 9 32 32, we want B C 32 32
    # Use: x_unfold[b_idx[:,None], torch.arange(C)[None,:], h_off[:,None], w_off[:,None]]
    # But that broadcasts to B C 32 32? Let's do simpler: gather via indexing per B
    # Use torch's indexing: x_unfold[torch.arange(B), :, h_off, w_off] works if we use proper broadcasting
    # Actually x_unfold[ b_idx, :, h_off, w_off ] should work: first dim B, second :, third h_off (B), fourth w_off (B)
    # PyTorch advanced indexing: x_unfold[b_idx, :, h_off, w_off] -> need to check
    # We tested: x_unfold[arange(B), :, h_off, w_off] gives B C 32 32? Let's use that
    # For safety, use loop-free gather via indexing
    # Create indices
    # Use: x_unfold[b_idx[:,None], torch.arange(C, device=x.device)[None,:], h_off[:,None], w_off[:,None]] -> but that is complex
    # Simpler: use the tested working form
    x = x_unfold[torch.arange(B, device=x.device), :, h_off, w_off]  # B C 32 32
    # Random horizontal flip
    flip_mask = torch.rand(B, device=x.device) < 0.5
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
    return x

# For CutMix/Mixup
def rand_bbox(W, H, lam):
    cut_rat = (1. - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W-1)
    cy = random.randint(0, H-1)
    bbx1 = max(0, cx - cut_w//2)
    bby1 = max(0, cy - cut_h//2)
    bbx2 = min(W, cx + cut_w//2)
    bby2 = min(H, cy + cut_h//2)
    return bbx1, bby1, bbx2, bby2

def evaluate():
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for i in range(0, test_images_gpu.shape[0], 512):
            x = test_images_gpu[i:i+512]
            y = test_labels_gpu[i:i+512]
            x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast('cuda'):
                out = model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total

os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    fp=os.path.join(CKPT_DIR,f)
    if os.path.isfile(fp):
        os.remove(fp)

start_wall = time.time()
first_ckpt_time=None
last_ckpt_time=None

# For faster data loading, create a DataLoader that yields indices and we fetch from CPU tensors
# Use simple random sampler with batch size
# We'll shuffle indices each epoch

for epoch in range(1, EPOCHS+1):
    model.train()
    if epoch <= warmup_epochs:
        warmup_lr = LR * epoch / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr']=warmup_lr
    else:
        scheduler.step()

    # shuffle
    perm = torch.randperm(N)
    epoch_loss=0
    steps=0
    for i in range(0, N, BATCH):
        idx = perm[i:i+BATCH]
        # fetch images: N H W C uint8 -> B H W C
        batch_images = train_images_cpu[idx]  # B H W C uint8
        batch_labels = train_labels_cpu[idx].to(device)
        # to GPU, HWC -> CHW, float
        x = batch_images.permute(0,3,1,2).float().div(255.0).to(device, non_blocking=True)
        # augment on GPU
        x = augment_batch(x)
        # normalize
        x = (x - mean) / std
        x = x.to(memory_format=torch.channels_last)

        # Mixup/CutMix
        r = random.random()
        if r < 0.3:
            lam = random.betavariate(0.2, 0.2)
            perm2 = torch.randperm(x.size(0), device=device)
            mixed_x = lam * x + (1-lam) * x[perm2]
            y_a, y_b = batch_labels, batch_labels[perm2]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(mixed_x)
                loss = lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        elif r < 0.6:
            lam = random.betavariate(1.0, 1.0)
            perm2 = torch.randperm(x.size(0), device=device)
            bbx1,bby1,bbx2,bby2 = rand_bbox(32,32, lam)
            mixed_x = x.clone()
            mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[perm2, :, bby1:bby2, bbx1:bbx2]
            lam = 1 - ((bbx2-bbx1)*(bby2-bby1) / (32*32))
            y_a, y_b = batch_labels, batch_labels[perm2]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(mixed_x)
                loss = lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = criterion(out, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        epoch_loss += loss.item()
        steps+=1

    acc = evaluate()
    lr_cur = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:03d}/{EPOCHS} loss {epoch_loss/steps:.4f} acc {acc*100:.2f}% lr {lr_cur:.5f} time {time.time()-start_wall:.1f}s")

    # Save scripted checkpoint (need non-compiled model)
    # Unwrap compiled model
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    cpu_model = WideResNet(depth=28, widen=10)
    cpu_model.load_state_dict(raw.state_dict())
    cpu_model.eval()
    scripted = torch.jit.script(cpu_model)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    if first_ckpt_time is None:
        first_ckpt_time=time.time()
    last_ckpt_time=time.time()
    try:
        m2=torch.jit.load(ckpt_path)
        m2.eval()
        with torch.no_grad():
            dummy=torch.randn(2,3,32,32)
            _=m2(dummy)
    except Exception as e:
        print(f"verify failed {e}")

ckpts=sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_epoch_")])
if ckpts:
    last_ckpt=os.path.join(CKPT_DIR, ckpts[-1])
    shutil.copy(last_ckpt, FINAL_CKPT)
    print(f"Copied {last_ckpt} to {FINAL_CKPT}")

total_wall = last_ckpt_time - start_wall if last_ckpt_time else time.time()-start_wall
print(f"Total wall {total_wall:.1f}s")
import json
try:
    # evaluate final via scripted model
    final_model=torch.jit.load(FINAL_CKPT).to(device).eval()
    # evaluate using test_images_gpu
    correct=0; total=0
    with torch.no_grad():
        for i in range(0, test_images_gpu.shape[0], 512):
            x=test_images_gpu[i:i+512]
            y=test_labels_gpu[i:i+512]
            with torch.amp.autocast('cuda'):
                out=final_model(x)
            pred=out.argmax(1)
            correct+=(pred==y).sum().item()
            total+=y.size(0)
    acc_final=correct/total
    print(f"Final acc {acc_final*100:.2f}%")
    with open("/workspace/agent_reported_metrics.json","w") as f:
        json.dump({"accuracy": float(acc_final), "wall_clock_sec": float(total_wall)}, f)
except Exception as e:
    print(f"final eval failed {e}")
    import traceback; traceback.print_exc()
