"""Ablate the test-time augmentation that is baked into the scripted forward pass.

TTA lives inside `Net.forward` (gated on `self.training`), so it ships with every
checkpoint. To measure what it buys, we pull the weights back out of a scripted
checkpoint into plain `Net` instances and evaluate three inference modes on the
verifier's exact input tensors.
"""
import json
import sys
import time

import torch

from eval_ckpt import test_batches
from model import Net


@torch.no_grad()
def run(model, mode):
    correct = total = 0
    t0 = time.time()
    for x, y in test_batches():
        if mode == "core":
            out = model.core(model.canon(x))
        elif mode == "mirror":
            out = model.mirror(model.canon(x))
        else:
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    torch.cuda.synchronize()
    return correct / total, round(time.time() - t0, 3)


def main(path, widths):
    sm = torch.jit.load(path).eval()
    model = Net(*widths, logit_scale=0.333).cuda().eval()
    missing = model.load_state_dict(sm.state_dict(), strict=True)
    print("# state_dict transferred:", missing, flush=True)
    out = {}
    for mode in ("core", "mirror", "tta"):
        acc, sec = run(model, mode)
        out[mode] = {"acc": acc, "eval_sec": sec}
        print(f"{mode:7s} acc {acc:.4f}   eval {sec:.2f}s", flush=True)
    # sanity: the plain module must reproduce the scripted checkpoint exactly
    x = test_batches()[0][0]
    same = torch.allclose(sm(x), model(x), atol=1e-5)
    out["scripted_matches_module"] = bool(same)
    print("scripted == rebuilt module:", same, flush=True)
    print(json.dumps(out), flush=True)
    with open("/workspace/ablation_tta.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], [int(v) for v in sys.argv[2:5]])
