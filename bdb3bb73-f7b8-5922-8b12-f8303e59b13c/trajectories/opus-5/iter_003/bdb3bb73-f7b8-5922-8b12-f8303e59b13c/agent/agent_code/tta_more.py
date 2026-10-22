import json, torch
from core import evaluate, load_split, standardise
from train_run import make_cfg, train
cfg = json.load(open("/workspace/final_cfg.json"))
raw_x, raw_y = load_split("train")
imgs = standardise(raw_x, dtype=torch.float16)
g = torch.Generator(device="cuda").manual_seed(1234)
va = torch.randperm(imgs.shape[0], device="cuda", generator=g)[-5000:]
vx = imgs[va].contiguous(memory_format=torch.channels_last); vy = raw_y[va]
rows = []
for seed in [2, 3, 4, 5]:
    _, m = train(make_cfg(cfg), holdout=5000, seed=seed, log_every=0, quiet=True)
    m.eval(); row = {}
    for lbl, (f, s) in {"none": (0, 0), "flip": (1, 0), "flip+shift": (1, 1)}.items():
        m.tta_flip, m.tta_shift = bool(f), bool(s)
        row[lbl] = evaluate(m, vx, vy)
    rows.append(row); print(seed, row, flush=True)
    del m; torch.cuda.empty_cache()
prev = [{"none":0.948,"flip":0.9522,"flip+shift":0.9508},{"none":0.946,"flip":0.9514,"flip+shift":0.952}]
allr = prev + rows
mean = {k: sum(r[k] for r in allr)/len(allr) for k in allr[0]}
wins = sum(1 for r in allr if r["flip"] > r["flip+shift"])
print("N=", len(allr), "means:", mean, " flip>flip+shift in", wins, "of", len(allr))
json.dump({"per_seed": allr, "mean": mean, "flip_wins": wins}, open("/workspace/tta_ablation.json","w"), indent=2)
