"""Starter pipeline for AELLO-C8-S1 (CIFAR-100 at a fixed step budget).

Trains for BUDGET_STEPS optimisation steps and writes class predictions for the graded
images. Edit freely: the graded quantity is the balanced accuracy of the predictions you
write, within the training budget stated in instruction.md.

Run:  python3 train.py --instance i20k --seed 0 \
          --graded ../data/graded_images.npz --out /submission/preds_i20k.npy
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data

BUDGET_STEPS = 8000          # the graded budget stated in instruction.md; keep it as shipped
BATCH = 128
EPOCHS = 200                 # schedule length
LR = 0.1
WIDTH = 64
CROP_PAD = 4


def make_model(seed, n_classes, width=WIDTH):
    torch.manual_seed(seed)

    def blk(a, b, stride=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(a, b, 3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(b), torch.nn.ReLU(inplace=True))

    return torch.nn.Sequential(
        blk(3, width), blk(width, width * 2, 2), blk(width * 2, width * 4, 2),
        blk(width * 4, width * 4, 2), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(width * 4, n_classes))


def random_crop(x, pad):
    """Pad and take a per-sample random crop."""
    b, c, h, w = x.shape
    xp = torch.nn.functional.pad(x, (pad,) * 4, mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ar = torch.arange(b, device=x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device)[None, :])[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=x.device)[None, :])[:, None, None, :]
    return xp[ar[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None],
              rows, cols]


def predict(model, x, device, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[s:s + batch])).permute(0, 3, 1, 2)
            xb = xb.to(device).float().div_(255.)
            with torch.autocast(device, dtype=torch.bfloat16):
                out.append(model(xb).argmax(1).cpu().numpy())
    model.train()
    return np.concatenate(out)


def train(instance, seed, device="cuda"):
    x, y = data.load_train(instance)
    model = make_model(seed, data.N_CLASSES).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True,
                          weight_decay=5e-4)
    steps_per_epoch = len(y) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                                total_steps=steps_per_epoch * EPOCHS,
                                                pct_start=0.15)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for _ in range(BUDGET_STEPS):
        i = np.sort(rng.integers(0, len(y), BATCH))
        xb = torch.from_numpy(np.ascontiguousarray(x[i])).permute(0, 3, 1, 2)
        xb = xb.contiguous(memory_format=torch.channels_last).to(device).float().div_(255.)
        xb = random_crop(xb, CROP_PAD)
        if rng.random() < 0.5:
            xb = torch.flip(xb, dims=[3])
        yb = torch.from_numpy(y[i]).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.bfloat16):
            torch.nn.functional.cross_entropy(model(xb), yb).backward()
        opt.step()
        sched.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, choices=list(data.INSTANCES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graded", required=True, help="path to the images to predict")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = train(a.instance, a.seed)
    xg = np.load(a.graded)["x"]
    np.save(a.out, predict(model, xg, "cuda").astype(np.int64))
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
