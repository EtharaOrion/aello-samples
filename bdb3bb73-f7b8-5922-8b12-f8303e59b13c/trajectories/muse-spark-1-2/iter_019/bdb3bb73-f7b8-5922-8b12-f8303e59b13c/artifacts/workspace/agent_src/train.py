import os, time, shutil, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import autoaugment

# --- config ---
EPOCHS = 50
BATCH_TRAIN = 1024
BATCH_TEST = 512
NUM_WORKERS = 8
LR_MAX = 1.2
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.2

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

CKPT_DIR = "/workspace/periodic_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
# clean old
for f in os.listdir(CKPT_DIR):
    os.remove(os.path.join(CKPT_DIR, f))

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- model: WideResNet 28-10 ---
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
        # need to handle shortcut after bn+relu for preact? Use original WRN preact: bn-relu-conv
        # shortcut should be from x after? For simplicity use post-bn shortcut
        shortcut = self.shortcut(out if isinstance(self.shortcut, nn.Conv2d) else x)
        out = self.conv1(out)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        return out + shortcut

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, dropout=0.3, num_classes=10):
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
        # init
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
        layers=[]
        layers.append(BasicBlock(in_ch, out_ch, stride, dropout))
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

# --- data ---
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3)),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder("/workspace/data/cifar10/train", transform=train_tf)
test_ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=False, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=False)

model = WideResNet(28,10,0.3).to(device)
model = model.to(memory_format=torch.channels_last)

optimizer = torch.optim.SGD(model.parameters(), lr=LR_MAX, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos', div_factor=10, final_div_factor=100)

scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    # ensure lam >=0.5 for stability? not needed
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

start_wall = time.time()
first_ckpt_time = None
last_ckpt_time = None

for epoch in range(1, EPOCHS+1):
    model.train()
    for x, y in train_loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        # mixup 50% of batches
        use_mixup = random.random() < 0.5
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            if use_mixup:
                x_m, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)
                out = model(x_m)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
                out = model(x)
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
        for x, y in test_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                out = model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    acc = correct/total
    print(f"epoch {epoch:02d} acc {acc*100:.2f}% lr {optimizer.param_groups[0]['lr']:.4f}")

    # save scripted checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
    # script on cpu to avoid device issues, but keep weights
    m_cpu = model  # keep on gpu, script will handle
    # Use torch.jit.script - need to ensure model is scriptable
    # Move to cpu for scripting then back? Scripting gpu model works but verifier loads on gpu
    scripted = torch.jit.script(model.eval())
    scripted.save(ckpt_path)
    # ensure mtime is set by save
    if first_ckpt_time is None:
        first_ckpt_time = os.path.getmtime(ckpt_path)
    last_ckpt_time = os.path.getmtime(ckpt_path)
    # set back to train
    model.train()
    # early copy final checkpoint if already 94?
    # continue training for full epochs to maximize chance

# final checkpoint.pt as copy of last
last_ckpt = os.path.join(CKPT_DIR, f"ckpt_epoch_{EPOCHS:04d}.pt")
shutil.copy(last_ckpt, "/workspace/checkpoint.pt")
end_wall = time.time()
wall = end_wall - start_wall
# also compute span
span = last_ckpt_time - first_ckpt_time if first_ckpt_time else wall
print(f"wall {wall:.1f}s span {span:.1f}s")
# write metrics
import json
# evaluate last ckpt accuracy for report
# use last acc
with open("/workspace/agent_reported_metrics.json","w") as f:
    json.dump({"accuracy": acc, "wall_clock_sec": wall}, f)
print("done")
