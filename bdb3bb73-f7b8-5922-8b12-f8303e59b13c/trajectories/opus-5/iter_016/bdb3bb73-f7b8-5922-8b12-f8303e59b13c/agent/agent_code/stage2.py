"""Stage 2: combine the axes that won the screen, then confirm on 3 seeds.

The screen moved one axis at a time; this stage checks that the winners compose,
and separately asks how to spend the ~2.3x wall-clock headroom we have under the
28 s full-credit anchor (more epochs vs. a wider net).
"""
import json, statistics, sys, time
from sweep import trial, BASE, mean_over_seeds

# Deltas from the screen (seed 0, paired -> deterministic) that beat the
# baseline by >= 1 sigma_hat (0.17 pp).
WINNERS = dict(wd=0.004, cutout=8, momentum=0.9, bias_scaler=1.0, lr=0.6)

COMBOS = {
    "c1_wd_cut":       dict(wd=0.004, cutout=8),
    "c2_plus_mom":     dict(wd=0.004, cutout=8, momentum=0.9, bias_scaler=1.0),
    "c3_plus_lr":      dict(**WINNERS),
    "c3_e40":          dict(**WINNERS, epochs=40),
    "c3_e52":          dict(**WINNERS, epochs=52),
    "c3_e40_cut10":    dict(**{**WINNERS, "cutout": 10}, epochs=40),
    "c3_wide_e26":     dict(**WINNERS, w1=160, w2=448, w3=640),
    "c3_wide_e40":     dict(**WINNERS, w1=160, w2=448, w3=640, epochs=40),
}

if __name__ == "__main__":
    t0 = time.time()
    res = {}
    for name, over in COMBOS.items():
        res[name] = trial(f"combo:{name}", **over)
    ranked = sorted(res.items(), key=lambda kv: -kv[1])
    print(json.dumps(ranked, indent=1), flush=True)

    # ---- confirm: top 3 distinct combos + incumbent baseline, seeds 0,1,2 ----
    finalists = [n for n, _ in ranked[:3]]
    summary = {}
    for n in finalists:
        m, a = mean_over_seeds(f"confirm:{n}", **COMBOS[n])
        summary[n] = {"mean": m, "accs": a, "cfg": COMBOS[n]}
    m, a = mean_over_seeds("confirm:incumbent")
    summary["incumbent"] = {"mean": m, "accs": a, "cfg": {}}
    json.dump(summary, open("/workspace/logs/confirm.json", "w"), indent=1)
    print(json.dumps({k: round(v["mean"], 5) for k, v in summary.items()}, indent=1), flush=True)
    print(f"stage2 done in {time.time()-t0:.0f}s", flush=True)
