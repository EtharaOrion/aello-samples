"""S4: confirmation of the S3/S3b finalists on 3 FRESH seeds (2,3,4), plus the re-opened
epoch budget (D2). eval_every=2 so the crossing epoch is resolved. Held-out 45k/5k only."""
import sys, json
import numpy as np
sys.path.insert(0,"/workspace")
from sweep import run_grid
B = dict(scale=0.03125, lr=11.5, wd=0.0051)
CAND = {
 "combo_dirac": dict(B, epochs=32, cutout=12, pad=3, lr_end=0.02, bn_momentum=0.8, dirac=True),
 "cutout16":    dict(B, epochs=32, cutout=16),
 "cutout12_dirac": dict(B, epochs=32, cutout=12, dirac=True),
 "cutout12_e24": dict(B, epochs=24, cutout=12),
}
rows = run_grid("S4_confirm", list(CAND.values()), seeds=[2,3,4], eval_every=2)
print("\n=== S4 confirmation: 3 fresh seeds ===")
name_of = {json.dumps(v, sort_keys=True): k for k, v in CAND.items()}
for k, ov in CAND.items():
    rs = [r for r in rows if json.dumps(r["ov"], sort_keys=True) == json.dumps(ov, sort_keys=True)]
    f = np.array([r["final"] for r in rs])
    # first epoch whose val >= 0.94, per seed
    cross = []
    for r in rs:
        c = [int(e) for e, a in sorted(r["curve"].items(), key=lambda t: int(t[0])) if a >= 0.94]
        cross.append(c[0] if c else None)
    print(f"{k:16s} mean={f.mean():.4f} sd={f.std(ddof=1):.4f} worst={f.min():.4f} "
          f"rank_metric(mean-sd)={f.mean()-f.std(ddof=1):.4f} epochs={ov['epochs']} "
          f"val_cross_epochs={cross}")
print("DONE_S4", flush=True)
