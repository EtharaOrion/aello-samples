import os, time, shutil, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import glob

CKPT_DIR = "/workspace/periodic_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda")
MEAN = torch.tensor([0.4914,0.4822,0.4465], device=device).view(1,3,1,1)
STD = torch.tensor([0.2470,0.2435,0.2616], device=device).view(1,3,1,1)

# --- preload data into RAM as uint8 ---
print("preloading...")
train_root = "/workspace/data/cifar10/train"
test_root = "/workspace/data/cifar10/test"
classes = sorted(os.listdir(train_root))
class_to_idx = {c:i for i,c in enumerate(classes)}

def load_folder(root):
    paths = []
    labels = []
    for c in classes:
        for p in glob.glob(os.path.join(root, c, "*.png")):
            paths.append(p)
            labels.append(class_to_idx[c])
    # load
    import torchvision.transforms.functional as TF
    import numpy as np
    N = len(paths)
    data = torch.empty(N, 3, 32, 32, dtype=torch.uint8)
    lbl = torch.tensor(labels, dtype=torch.long)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        # to tensor uint8
        arr = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).numpy() if False else None)
        # simpler: use PIL to numpy
        import numpy as np
        np_arr = np.array(img) # HWC
        data[i] = torch.from_numpy(np_arr).permute(2,0,1)
        if i % 10000 == 0:
            print(i)
    return data, lbl

train_data_u8, train_labels = load_folder(train_root)
test_data_u8, test_labels = load_folder(test_root)
print(f"train {train_data_u8.shape} test {test_data_u8.shape}")

# move test to gpu normalized for eval
test_data_f = test_data_u8.float().div(255.0)
# normalize on cpu for eval loader? We'll do gpu eval directly
# Keep test on gpu
test_data_gpu = (test_data_f.to(device) - MEAN) / STD  # NCHW normalized
test_labels_gpu = test_labels.to(device)

# --- model WRN 28-10 but slightly optimized ---
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out if isinstance(self.shortcut, nn.Conv2d) else x)
        out = self.conv1(out)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        return out + shortcut

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, dropout=0.0, num_classes=10):
        super().__init__()
        assert (depth-4)%6==0
        n = (depth-4)//6
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(16, 16*widen, n, 1, dropout)
        self.layer2 = self._make_layer(16*widen, 32*widen, n, 2, dropout)
        self.layer3 = self._make_layer(32*widen, 64*widen, n, 2, dropout)
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
        layers=[BasicBlock(in_ch, out_ch, stride, dropout)]
        for _ in range(1,n):
            layers.append(BasicBlock(out_ch, out_ch, 1, dropout))
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

# hyperparams for speed
EPOCHS = 35
BATCH = 1024
LR_MAX = 1.5
WD = 5e-4
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.2

model = WideResNet(28,10,0.0).to(device).to(memory_format=torch.channels_last)
# torch.compile for speed
model = torch.compile(model, mode="max-autotune")

optimizer = torch.optim.SGD(model.parameters(), lr=LR_MAX, momentum=0.9, weight_decay=WD, nesterov=True)
# OneCycle
steps_per_epoch = (len(train_data_u8) + BATCH -1)//BATCH
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
scaler = torch.amp.GradScaler('cuda')

# GPU augmentation helpers
def augment_batch(x): # x is float [0,1] NCHW on gpu
    # x already normalized? We'll augment before normalize
    # Input x is float [0,1] NCHW
    # Random crop with padding 4
    # pad 4
    x = F.pad(x, (4,4,4,4), mode='reflect')
    # random crop 32
    N = x.size(0)
    # generate random offsets
    # Use torch.randint
    h_off = torch.randint(0, 9, (N,), device=x.device)
    w_off = torch.randint(0, 9, (N,), device=x.device)
    # crop via gather - loop is slow, use vectorized via unfold? Simpler: use torch.stack with slicing via advanced indexing
    # Use batched crop via F.unfold trick or just use torchvision on gpu? We'll do efficient via torch
    # For speed, use random crop via indexing with for loop in python is slow for 1024
    # Alternative: use Kornia style: use grid_sample? Simpler: use torch's random crop via batched indexing using torch.arange
    # We'll implement via torch.stack of slices - but that loops in python 1024 times per batch -> slow
    # Instead use vectorized: create base grid
    # Quick hack: use F.pad then use torch's random crop via torchvision transforms on GPU using torch.roll? Not correct
    # Let's use simple: generate random crops via torch.nn.functional.unfold and select? Might be complex
    # For now do python loop but with numba? Let's try loop with 1024 iterations - might be ~1ms per batch? Let's test
    # We'll do loop but using slicing which is fast in cuda (each slice is a kernel)
    crops = []
    for i in range(N):
        crops.append(x[i:i+1, :, h_off[i]:h_off[i]+32, w_off[i]:w_off[i]+32])
    x = torch.cat(crops, dim=0)
    # flip
    flip_mask = torch.rand(N, device=x.device) < 0.5
    # flip where mask true
    if flip_mask.any():
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
    return x

# Pre-create indices
train_N = len(train_data_u8)
# Keep train data on GPU as uint8 to save memory? 50k*3*32*32 = 150M, as float 600M, okay to keep on GPU as uint8
train_data_gpu_u8 = train_data_u8.to(device)  # 150MB on GPU
train_labels_gpu = train_labels.to(device)

start_wall = time.time()
first_mtime = None
last_mtime = None
last_acc = 0

for epoch in range(1, EPOCHS+1):
    model.train()
    # shuffle
    perm = torch.randperm(train_N, device=device)
    # iterate batches
    for i in range(0, train_N, BATCH):
        idx = perm[i:i+BATCH]
        if len(idx) < BATCH:
            continue
        x_u8 = train_data_gpu_u8[idx]  # NCHW uint8
        y = train_labels_gpu[idx]
        x = x_u8.float().div(255.0)  # [0,1]
        x = augment_batch(x)
        # normalize
        x = (x - MEAN) / STD
        x = x.to(memory_format=torch.channels_last)
        # mixup 50%
        use_mixup = random.random() < 0.5
        if use_mixup:
            lam = torch.distributions.Beta(MIXUP_ALPHA, MIXUP_ALPHA).sample().item()
            perm2 = torch.randperm(x.size(0), device=device)
            x2 = x[perm2]
            y2 = y[perm2]
            x = lam * x + (1-lam) * x2
            # will compute loss mixup
            y_a, y_b = y, y2
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(x)
            if use_mixup:
                loss = lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)
            else:
                loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    # eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_data_gpu), 512):
            x = test_data_gpu[i:i+512].to(memory_format=torch.channels_last)
            y = test_labels_gpu[i:i+512]
            with torch.amp.autocast('cuda'):
                out = model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    acc = correct/total
    last_acc = acc
    print(f"epoch {epoch:02d} acc {acc*100:.2f}% lr {optimizer.param_groups[0]['lr']:.5f}")

    # save scripted checkpoint - need uncompiled model for scripting
    # Get original model without compile wrapper
    # torch.compile wraps, but we can script the underlying _orig_mod
    # For simplicity, create a fresh model and load state dict, then script
    # But we can also just save the compiled model's state via scripting the compiled model? torch.jit.script doesn't support compiled
    # So we need to keep a non-compiled copy or unwrap
    # We'll create a scriptable model instance and copy weights
    import copy
    # unwrap
    try:
        orig = model._orig_mod
    except:
        orig = model
    # create scriptable model
    script_model = WideResNet(28,10,0.0).to(device)
    script_model.load_state_dict(orig.state_dict())
    script_model.eval()
    scripted = torch.jit.script(script_model)
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    scripted.save(ckpt_path)
    if first_mtime is None:
        first_mtime = os.path.getmtime(ckpt_path)
    last_mtime = os.path.getmtime(ckpt_path)
    # keep model train
    model.train()

# final copy
shutil.copy(os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt"), "/workspace/checkpoint.pt")
wall = time.time() - start_wall
print(f"wall {wall:.1f}s span {last_mtime-first_mtime:.1f}s")
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": last_acc, "wall_clock_sec": wall}, f)
print("done")
