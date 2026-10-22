"""Pre-flight checks: ZCA conditioning, scriptability, save/load round-trip."""
import os

import torch

from model import Net
from train import load_train, standardize, DEFAULT, build_model

dev = "cuda"
x_u8, y = load_train(dev)
xs_all = standardize(x_u8)
print("standardized range %.3f %.3f | per-channel mean %s" % (
    xs_all.min().item(), xs_all.max().item(),
    [round(v, 4) for v in xs_all.mean((0, 2, 3)).tolist()]))

m = build_model(DEFAULT, xs_all, dev)
print("params(M) %.3f  trainable(M) %.3f" % (
    sum(p.numel() for p in m.parameters()) / 1e6,
    sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6))

# ZCA check on a *held-out-from-the-estimate* slice. init_whitening uses the
# uncentered second moment (like the reference airbench), so the quantity that
# should come out as the identity is E[ww^T], not the centered covariance.
probe = xs_all[torch.randperm(xs_all.shape[0], device=dev)[:2000]]
with torch.no_grad():
    w = m.whiten(probe.to(memory_format=torch.channels_last))[:, :12]
v = w.permute(1, 0, 2, 3).reshape(12, -1)
sm2 = (v @ v.T) / v.shape[1]
eye = torch.eye(12, device=dev)
print("uncentered E[ww^T]: diag min/max %.3f/%.3f | max |off-diag| %.4f" % (
    sm2.diag().min(), sm2.diag().max(), (sm2 - eye).abs().mul(1 - eye).max()))
print("  (diag < 1 only for directions damped by zca_eps=%g)" % DEFAULT["zca_eps"])

# Script + save + reload, and confirm the state_dict transfer path works.
sm = torch.jit.script(Net(widths=DEFAULT["widths"], bn_momentum=DEFAULT["bn_momentum"],
                          logit_scale=DEFAULT["logit_scale"], tta=DEFAULT["tta"]).cuda())
sm.eval()
sm.load_state_dict(m.state_dict())
sm.save("/tmp/probe.pt")
lm = torch.jit.load("/tmp/probe.pt").eval().cuda()
m.eval()
xb = xs_all[:512].contiguous()
with torch.no_grad():
    a, b = m(xb), lm(xb)
print("scripted vs eager max abs diff: %g | out shape %s | %.1f MB" % (
    (a - b).abs().max().item(), tuple(b.shape), os.path.getsize("/tmp/probe.pt") / 1e6))
