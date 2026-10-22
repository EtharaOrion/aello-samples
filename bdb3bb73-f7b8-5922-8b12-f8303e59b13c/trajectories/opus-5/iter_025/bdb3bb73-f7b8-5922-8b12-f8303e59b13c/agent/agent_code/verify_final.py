"""End-to-end check of /workspace/checkpoint.pt under the grader's exact call pattern."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.2470, 0.2435, 0.2616))])
ds = datasets.ImageFolder("/workspace/data/cifar10/test", transform=tf)
dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

model = torch.jit.load("/workspace/checkpoint.pt").eval().cuda()
correct = total = 0
rng = [float("inf"), -float("inf")]
with torch.no_grad():
    for x, y in dl:
        x, y = x.cuda(), y.cuda()
        assert x.dtype == torch.float32 and x.shape[1:] == (3, 32, 32)
        rng = [min(rng[0], x.min().item()), max(rng[1], x.max().item())]
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
print(f"checkpoint.pt: {correct}/{total} = {correct/total*100:.2f}%   input range [{rng[0]:.2f}, {rng[1]:.2f}]")
