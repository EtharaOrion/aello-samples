"""Contract tests for the exported checkpoints.

1. `torch.jit.script` shares parameter storage with the eager module, so the
   script-once/save-many-times pattern in train.py really does serialise the
   weights as of the save, not as of scripting time.
2. A loaded checkpoint gives the same predictions for standardised, [0,1] and
   [0,255] inputs -- i.e. `Net.canonicalize` is an identity on the verifier's
   convention and a corrective rescale on the other two.
"""
import os, sys, tempfile
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, "/workspace")
from model import Net, CIFAR_MEAN, CIFAR_STD  # noqa: E402


def test_script_shares_storage():
    m = Net().cuda().eval()
    s = torch.jit.script(m)
    with torch.no_grad():
        m.head.weight.mul_(0.0).add_(1.234)
        m.g1.norm1.running_mean.fill_(7.0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pt")
        s.save(p)
        r = torch.jit.load(p)
    assert torch.allclose(r.head.weight, torch.full_like(r.head.weight, 1.234)), "weights stale"
    assert torch.allclose(r.g1.norm1.running_mean,
                          torch.full_like(r.g1.norm1.running_mean, 7.0)), "buffers stale"
    print("PASS script shares storage (post-script mutations are serialised)")


def test_input_scale_invariance(ckpt):
    m = torch.jit.load(ckpt).eval().cuda()
    x255 = torch.from_numpy(np.load("/workspace/cache/test_x.npy")[:512]).cuda().float()
    norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    x01 = x255 / 255.0
    xstd = norm(x01)
    print(f"  ranges: std=[{xstd.min():.2f},{xstd.max():.2f}] "
          f"01=[{x01.min():.2f},{x01.max():.2f}] 255=[{x255.min():.0f},{x255.max():.0f}]")
    with torch.no_grad():
        p_std = m(xstd).argmax(1)
        p_01 = m(x01).argmax(1)
        p_255 = m(x255).argmax(1)
    y = torch.from_numpy(np.load("/workspace/cache/test_y.npy")[:512]).cuda()
    accs = [(p == y).float().mean().item() for p in (p_std, p_01, p_255)]
    print(f"  acc standardised={accs[0]:.4f} [0,1]={accs[1]:.4f} [0,255]={accs[2]:.4f}")
    assert (p_std == p_01).float().mean() > 0.995, "[0,1] path disagrees"
    assert (p_std == p_255).float().mean() > 0.995, "[0,255] path disagrees"
    assert accs[0] > 0.9, "standardised (verifier) path is broken"
    print("PASS all three input conventions agree")


if __name__ == "__main__":
    test_script_shares_storage()
    ck = sys.argv[1] if len(sys.argv) > 1 else "/workspace/checkpoint.pt"
    test_input_scale_invariance(ck)
