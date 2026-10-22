"""Lightweight GPU-throughput probe.

Measures fwd+bwd time for one training-sized batch on synthetic noise, once every
~20 s, and appends it to a log. Used to detect when the (externally shared) GPU is
busy, so the graded training run can be launched in a quiet window.
"""
import sys, time; sys.path.insert(0, "/workspace")
import torch, torch.nn.functional as F
from airbench import Net

net = Net(widths=(128, 384, 512)).cuda().to(memory_format=torch.channels_last)
x = torch.randn(1024, 3, 32, 32, device="cuda").to(memory_format=torch.channels_last)
y = torch.randint(0, 10, (1024,), device="cuda")

def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        F.cross_entropy(net(x), y, label_smoothing=0.2, reduction="sum").backward()
    net.zero_grad(set_to_none=True)

for _ in range(8):
    step()
torch.cuda.synchronize()

while True:
    t = time.time()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    ms = (time.time() - t) / 10 * 1000
    with open("/workspace/contention_log.txt", "a") as f:
        f.write(f"{time.time():.1f} {ms:.2f}\n")
    time.sleep(20)
