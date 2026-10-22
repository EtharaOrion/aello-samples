"""Component ablation for the promoted recipe (28 ep / cutout 12 / lr 0.3 / 128-384-512).

Everything is measured on the same fixed 5,000-image holdout slice of the TRAIN split that
was used for tuning; the test split is not read. Two seeds per row, since the measured
seed-to-seed noise is sigma ~= 0.12 pp.

The TTA rows re-score one already-trained model at tta level 0/1/2, so they isolate the
inference-time contribution with the weights held fixed.
"""
import json
import warnings

warnings.filterwarnings("ignore")
import train as T  # noqa: E402

BASE = ["--epochs", "28", "--cutout", "12", "--lr", "0.3",
        "--widths", "128", "384", "512", "--holdout", "5000", "--save", "0",
        "--log", "/workspace/runs.jsonl"]
SEEDS = ["11", "12"]

VARIANTS = [
    ("promoted recipe", []),
    ("no cutout", ["--cutout", "0"]),
    ("no translate aug", ["--translate", "0"]),
    ("learned first layer (no frozen ZCA)", ["--whiten", "learned"]),
    ("learnable BN scales", ["--bn-affine", "1"]),
    ("no label smoothing", ["--ls", "0.0"]),
]

rows = []
for name, extra in VARIANTS:
    accs, tta_rows = [], []
    for sd in SEEDS:
        r = T.train(T.get_args(BASE + extra + ["--seed", sd]))
        accs.append(r["val_acc"])
        if name == "promoted recipe":  # re-score the same weights at each TTA level
            m, (xv, yv) = T.LAST["model"], T.LAST["val"]
            lv = {}
            for level in (0, 1, 2):
                m.tta = level
                lv[level] = T.evaluate(m, xv, yv)
            tta_rows.append(lv)
    rows.append({"variant": name, "accs": accs, "mean": sum(accs) / len(accs), "tta": tta_rows})
    print(f"RESULT {json.dumps(rows[-1])}", flush=True)

base = rows[0]["mean"]
print("\n=== ablation (2 seeds, holdout) ===")
for r in rows:
    print(f"{r['mean']:.4f}  ({r['accs'][0]:.4f}/{r['accs'][1]:.4f})  "
          f"delta={100 * (r['mean'] - base):+.2f}pp  {r['variant']}")
if rows[0]["tta"]:
    for level in (0, 1, 2):
        v = [t[level] for t in rows[0]["tta"]]
        print(f"tta_level={level}: {sum(v) / len(v):.4f}  ({'/'.join(f'{a:.4f}' for a in v)})")
json.dump(rows, open("/workspace/ablation.json", "w"), indent=1)
