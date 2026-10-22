"""Ablations around the promoted configuration.

Reported for understanding only -- per PREREGISTRATION.md these do not feed back
into model selection.  All runs happen in one process at one seed so the
comparisons are exactly paired (see RESULTS.md on within- vs cross-process
determinism).
"""
import json, time
from sweep import trial

PROMOTED = dict(epochs=52, lr=0.6, momentum=0.9, wd=0.004, bias_scaler=1.0, cutout=8)

ABLATIONS = {
    "promoted":        {},
    "-tta":            dict(tta=0),
    "-cutout":         dict(cutout=0),
    "-translate":      dict(pad=0),
    "-mirror":         dict(flip=0),
    "-zca_whitening":  dict(zca=0),
    "-label_smooth":   dict(ls=0.0),
    "-width(64/256/256)": dict(w1=64, w2=256, w3=256),
    "-epochs(52->26)": dict(epochs=26),
}

if __name__ == "__main__":
    t0 = time.time()
    out = {}
    for tag, over in ABLATIONS.items():
        out[tag] = trial(f"ablate:{tag}", **{**PROMOTED, **over})
    base = out["promoted"]
    table = {k: {"acc": v, "delta_pp": round((v - base) * 100, 2)} for k, v in out.items()}
    json.dump(table, open("/workspace/logs/ablations.json", "w"), indent=1)
    print(json.dumps(table, indent=1), flush=True)
    print(f"ablations done in {time.time()-t0:.0f}s", flush=True)
