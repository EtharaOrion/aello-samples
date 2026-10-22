import os, time, json, shutil, copy
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
DEVICE = 'cuda'
ROOT = Path('/workspace')
OUT = Path(os.environ.get('CKPT_DIR', str(ROOT/'periodic_checkpoints')))
MEAN = torch.tensor((0.4914,0.4822,0.4465), device=DEVICE).view(1,3,1,1)
STD = torch.tensor((0.2470,0.2435,0.2616), device=DEVICE).view(1,3,1,1)

def load_split(split):
    base = ROOT/'data/cifar10'/split
    classes = sorted(p.name for p in base.iterdir() if p.is_dir())
    xs, ys = [], []
    for y,c in enumerate(classes):
        for fn in sorted((base/c).glob('*.png')):
            xs.append(np.asarray(Image.open(fn).convert('RGB'), dtype=np.uint8))
            ys.append(y)
    x = torch.from_numpy(np.stack(xs)).permute(0,3,1,2).cuda()
    y = torch.tensor(ys, device=DEVICE)
    return x,y

class ConvGroup(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.res1 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.res2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
    def forward(self, x):
        x = F.gelu(self.bn(F.max_pool2d(self.conv(x), 2)))
        return F.gelu(x + self.bn2(self.res2(F.gelu(self.bn1(self.res1(x))))))

class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, bias=False)
        self.g1 = ConvGroup(24, 64)
        self.g2 = ConvGroup(64, 256)
        self.g3 = ConvGroup(256, 256)
        self.head = nn.Linear(256, 10, bias=False)
        self.scale = 0.2
        self.tta = False
    def forward_once(self, x):
        x = F.gelu(self.whiten(x))
        x = self.g3(self.g2(self.g1(x)))
        x = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.head(x) * self.scale
    def forward(self, x):
        # x is already normalized exactly as required by the verifier.
        x = x.to(dtype=self.whiten.weight.dtype)
        z = self.forward_once(x)
        if self.tta:
            z = (z + self.forward_once(x.flip(-1))) * 0.5
        return z

def aug(x):
    x = x.float().div_(255.0)
    # Per-image horizontal reflection.
    m = torch.rand(x.shape[0], device=DEVICE) < .5
    x[m] = x[m].flip(-1)
    # Independent reflected crops (the usual CIFAR random translation).
    x = F.pad(x, (4,4,4,4), mode='reflect')
    b,c,_,_=x.shape
    oy=torch.randint(0,9,(b,1,1),device=DEVICE); ox=torch.randint(0,9,(b,1,1),device=DEVICE)
    yy=torch.arange(32,device=DEVICE).view(1,32,1)+oy
    xx=torch.arange(32,device=DEVICE).view(1,1,32)+ox
    ind=(yy*40+xx).reshape(b,1,1024).expand(b,c,1024)
    x=x.flatten(2).gather(2,ind).reshape(b,c,32,32)
    return (x-MEAN)/STD

def save_model(model, epoch):
    # Half-precision inference snapshots halve serialization traffic.  Forward
    # explicitly accepts the verifier's float32 tensors above.
    snap = copy.deepcopy(model).half().eval()
    snap.tta = True
    p = OUT/f'ckpt_epoch_{epoch:04d}.pt'
    scripted = torch.jit.script(snap)
    scripted.save(str(p))

def main():
    seed=int(os.environ.get('TRAIN_SEED','2026'))
    torch.manual_seed(seed); np.random.seed(seed)
    train_x, train_y = load_split('train')
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # Initialize the first layer as a ZCA-like bank from TRAIN patches only.
    # The +/- pairing retains all signs after GELU and is then fine-tuned.
    with torch.no_grad():
        sample=train_x[:5000].float().div(255)
        sample=(sample-MEAN)/STD
        patches=F.unfold(sample,kernel_size=2).transpose(1,2).reshape(-1,12)
        patches=patches[::32]
        patches-=patches.mean(0)
        cov=patches.T@patches/(len(patches)-1)
        vals,vecs=torch.linalg.eigh(cov)
        w=(vecs/torch.sqrt(vals.clamp_min(1e-4)).unsqueeze(0)).T.reshape(12,3,2,2)
        model.whiten.weight.copy_(torch.cat((w,-w))*0.25)
    # Separate decay from normalization parameters.
    wd, no = [], []
    for n,p in model.named_parameters():
        (no if p.ndim < 2 or 'bn' in n else wd).append(p)
    opt = torch.optim.SGD([{'params':wd,'weight_decay':5e-4},{'params':no,'weight_decay':0}],
                          lr=.12, momentum=.9, nesterov=True)
    epochs=40; bs=1000; steps=(len(train_y)+bs-1)//bs
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.6,epochs=epochs,steps_per_epoch=steps,
                                               pct_start=.15,div_factor=5,final_div_factor=100)
    scaler=torch.amp.GradScaler('cuda')
    OUT.mkdir(parents=True,exist_ok=True)
    for old in OUT.glob('ckpt_epoch_*.pt'): old.unlink()
    start=time.time(); rows=[]
    compiled=torch.compile(model, mode='max-autotune-no-cudagraphs')
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    for ep in range(1,epochs+1):
        perm=torch.randperm(len(train_y),device=DEVICE)
        loss_sum=0.0
        for j in range(0,len(train_y),bs):
            ix=perm[j:j+bs]; xb=aug(train_x[ix]).to(memory_format=torch.channels_last); yb=train_y[ix]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',dtype=torch.float16):
                loss=F.cross_entropy(compiled(xb),yb,label_smoothing=.1)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                for q,p in zip(ema.parameters(),model.parameters()): q.lerp_(p,0.02)
            loss_sum += float(loss.detach())
        # Current BN statistics pair better with averaged weights than stale buffers.
        with torch.no_grad():
            for qb,b in zip(ema.buffers(),model.buffers()): qb.copy_(b)
        torch.cuda.synchronize(); save_model(ema,ep)
        elapsed=time.time()-start; rows.append({'epoch':ep,'loss':loss_sum/steps,'seconds':elapsed})
        print(json.dumps(rows[-1]),flush=True)
    final=OUT/f'ckpt_epoch_{epochs:04d}.pt'
    if OUT == ROOT/'periodic_checkpoints':
        dst=ROOT/'checkpoint.pt'
        if dst.exists(): dst.unlink()
        os.link(final,dst)
        (ROOT/'train_metrics.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__': main()
