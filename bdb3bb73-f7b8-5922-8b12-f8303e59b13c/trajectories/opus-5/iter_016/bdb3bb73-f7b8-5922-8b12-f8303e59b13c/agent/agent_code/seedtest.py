"""Is the promoted config really better than the cheaper runner-up?

The 3-seed confirmation put c3_e52 0.167 pp ahead of c1_wd_cut, barely over the
0.137 pp pre-registered threshold.  This re-runs both at 10 seeds, interleaved
in a single process so the two arms see the same allocator/autotune history and
the comparison is paired seed-by-seed.

Reporting only: PREREGISTRATION.md fixes selection at the 3-seed mean, and the
ladder was already promoted before this ran.
"""
import json, statistics, time
from sweep import trial

ARMS = {
    "c3_e52":    dict(epochs=52, lr=0.6, momentum=0.9, wd=0.004, bias_scaler=1.0, cutout=8),
    "c1_wd_cut": dict(wd=0.004, cutout=8),
}
SEEDS = range(10)

if __name__ == "__main__":
    t0 = time.time()
    acc = {k: [] for k in ARMS}
    for s in SEEDS:
        for name, over in ARMS.items():
            acc[name].append(trial(f"seedtest:{name}", seed=s, **over))
    a, b = acc["c3_e52"], acc["c1_wd_cut"]
    diff = [x - y for x, y in zip(a, b)]
    n = len(diff)
    md, sd = statistics.mean(diff), statistics.stdev(diff)
    se = sd / n ** 0.5
    out = {
        "c3_e52": {"mean": statistics.mean(a), "sd": statistics.stdev(a), "accs": a},
        "c1_wd_cut": {"mean": statistics.mean(b), "sd": statistics.stdev(b), "accs": b},
        "paired_diff_pp": {"mean": md * 100, "sd": sd * 100, "se": se * 100,
                           "t": md / se, "n": n,
                           "ci95_pp": [(md - 2.262 * se) * 100, (md + 2.262 * se) * 100]},
    }
    json.dump(out, open("/workspace/logs/seedtest.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print(f"seedtest done in {time.time()-t0:.0f}s", flush=True)
