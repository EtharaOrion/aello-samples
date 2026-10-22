import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path('/workspace')
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1).cuda()
std = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1).cuda()
xs, ys = [], []
for label, name in enumerate(CLASSES):
    for p in sorted((ROOT / 'data/cifar10/test' / name).glob('*.png')):
        xs.append(torch.from_numpy(np.array(Image.open(p).convert('RGB'), copy=True)).permute(2, 0, 1))
        ys.append(label)
x = torch.stack(xs).cuda().float().div_(255).sub_(mean).div_(std)
y = torch.tensor(ys, device='cuda')
results = []
for path in sorted((ROOT / 'periodic_checkpoints').glob('ckpt_epoch_*.pt')):
    model = torch.jit.load(str(path), map_location='cuda').eval()
    correct = 0
    with torch.inference_mode():
        for i in range(0, len(y), 512):
            correct += int((model(x[i:i+512]).argmax(1) == y[i:i+512]).sum())
    acc = correct / len(y)
    result = {'checkpoint': path.name, 'accuracy': acc}
    results.append(result)
    print(json.dumps(result), flush=True)
with open(ROOT / 'ladder_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
