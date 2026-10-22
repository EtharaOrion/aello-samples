"""Leave-one-out ablation of the PROMOTED recipe. Run after promotion; explanatory only,
it cannot and does not influence model selection. Held-out 45k/5k, no test reads."""
import sys; sys.path.insert(0,"/workspace")
from sweep import run_grid, summarize
FINAL = dict(scale=0.03125, lr=11.5, wd=0.0051, epochs=32, cutout=16)
ABL = [
    ("promoted (all on)",      {}),
    ("- cutout",               dict(cutout=0)),
    ("- TTA (single view)",    dict(tta=False)),
    ("- ZCA whitening",        dict(whiten="random")),
    ("- tuned wd (->0.0153)",  dict(wd=0.0153)),
    ("- tuned logit scale (->1/9)", dict(scale=1/9)),
]
rows = run_grid("ABL_loo", [dict(FINAL, **v) for _, v in ABL], seeds=[0,1], eval_every=32)
print("\n=== leave-one-out ablation (2 seeds, 45k->5k val) ===")
import json, numpy as np
for label, v in ABL:
    ov = dict(FINAL, **v); key = json.dumps(ov, sort_keys=True)
    f = np.array([r["final"] for r in rows if json.dumps(r["ov"], sort_keys=True)==key])
    print(f"  {label:32s} mean={f.mean():.4f} sd={f.std(ddof=1):.4f}")
print("DONE_ABL", flush=True)
