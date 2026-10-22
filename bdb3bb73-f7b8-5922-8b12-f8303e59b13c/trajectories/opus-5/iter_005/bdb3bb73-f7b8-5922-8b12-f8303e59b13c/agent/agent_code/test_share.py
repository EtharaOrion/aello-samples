"""Red-line test: does a ScriptModule scripted ONCE reflect later in-place
optimizer updates and BN buffer updates?  If not, every saved checkpoint would
be stale (a scoring knockout), so this must pass before training is trusted."""
import torch
import torch.nn as nn
from train import Config, Net

dev = torch.device("cuda")
cfg = Config()
m = Net(cfg).to(dev)
s = torch.jit.script(m)

# 1) parameter storage sharing
p = m.head.weight
assert p.data_ptr() == dict(s.named_parameters())["head.weight"].data_ptr(), "param not shared"

# 2) optimizer step visible through the ScriptModule
opt = torch.optim.SGD([q for q in m.parameters() if q.requires_grad], lr=0.1, momentum=0.9)
x = torch.randn(64, 3, 32, 32, device=dev)
before = dict(s.named_parameters())["head.weight"].clone()
m.train()
loss = m.features(x).square().mean()
opt.zero_grad(); loss.backward(); opt.step()
after = dict(s.named_parameters())["head.weight"]
assert not torch.equal(before, after), "optimizer step NOT visible in ScriptModule"

# 3) BN running stats (buffers) visible too
rb = dict(s.named_buffers())["g1.bn1.running_mean"]
assert rb.data_ptr() == m.g1.bn1.running_mean.data_ptr(), "buffer not shared"
assert rb.abs().sum().item() > 0, "BN buffers did not update"

# 4) round-trip: save now, reload, outputs must match the live model exactly
s.eval(); m.eval()
s.save("/tmp/_share_test.pt")
r = torch.jit.load("/tmp/_share_test.pt").eval().to(dev)
with torch.no_grad():
    a, b = m(x), r(x)
assert torch.equal(a, b), f"reload mismatch max|d|={(a-b).abs().max().item()}"

# 5) a SECOND save after more training must differ from the first
m.train()
for _ in range(3):
    loss = m.features(x).square().mean()
    opt.zero_grad(); loss.backward(); opt.step()
s.eval(); m.eval()
s.save("/tmp/_share_test2.pt")
r2 = torch.jit.load("/tmp/_share_test2.pt").eval().to(dev)
with torch.no_grad():
    assert not torch.equal(r(x), r2(x)), "second checkpoint identical to first (STALE)"

print("PASS: ScriptModule shares live storage; per-epoch .save() is not stale.")
