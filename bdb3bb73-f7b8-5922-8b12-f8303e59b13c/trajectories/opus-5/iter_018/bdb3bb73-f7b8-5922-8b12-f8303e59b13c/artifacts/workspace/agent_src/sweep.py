"""Run holdout experiments as fresh subprocesses and collect their summaries.

Each run is a separate process on purpose: cuDNN autotuning, allocator state
and RNG all reset, so repeated runs of the *same* config measure the true
run-to-run noise rather than a partially shared warm state.
"""

import argparse
import json
import statistics
import subprocess
import sys

BASE = ["python", "/workspace/train.py", "--holdout", "5000", "--no-save"]


def run(overrides, seed, epochs=None):
    cmd = BASE + ["--seed", str(seed)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        if v == "":            # store_true flag
            cmd += [flag]
        elif isinstance(v, (list, tuple)):
            cmd += [flag] + [str(x) for x in v]
        else:
            cmd += [flag, str(v)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("SUMMARY "):
            return json.loads(line[8:])
    sys.stderr.write(out.stdout[-2000:] + out.stderr[-2000:])
    raise RuntimeError(f"run failed: {cmd}")


def describe(name, accs):
    m = statistics.mean(accs)
    s = statistics.stdev(accs) if len(accs) > 1 else float("nan")
    print(f"{name:<34} n={len(accs)} mean={m:.4f} sd={s:.4f} min={min(accs):.4f} "
          f"vals={[round(a, 4) for a in accs]}", flush=True)
    return {"name": name, "mean": m, "sd": s, "min": min(accs), "accs": accs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["noise", "screen", "finalists", "ablate"])
    ap.add_argument("--epochs", type=float, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    incumbent = dict(wd=3e-4, lr=11.5, cutout=0, label_smoothing=0.2)
    results = []

    if args.mode == "noise":
        accs = [run(incumbent, s, args.epochs)["final_val"] for s in range(args.seeds)]
        results.append(describe(f"incumbent@{args.epochs:g}ep", accs))

    elif args.mode == "screen":
        grid = [("epochs", [24, 32, 40]), ("lr", [8.0, 15.0]), ("wd", [1e-4, 8e-4]),
                ("cutout", [6, 8]), ("label_smoothing", [0.1]),
                ("widths", [[64, 256, 256]])]
        for factor, values in grid:
            for v in values:
                cfg, ep = dict(incumbent), args.epochs
                if factor == "epochs":
                    ep = v
                else:
                    cfg[factor] = v
                r = run(cfg, 0, ep)
                results.append(describe(f"{factor}={v}", [r["final_val"]]))

    elif args.mode == "finalists":
        cands = json.load(open("/workspace/finalists.json"))
        for c in cands:
            accs = [run(c["cfg"], s, c.get("epochs", args.epochs))["final_val"]
                    for s in range(args.seeds)]
            results.append(describe(c["name"], accs))

    elif args.mode == "ablate":
        final = json.load(open("/workspace/final_cfg.json"))
        cfg, ep = final["cfg"], final["epochs"]
        variants = [("full", {}), ("no-TTA", None), ("no-translate", {"translate": 0}),
                    ("no-flip", {"no_flip": ""}),
                    ("no-cutout", {"cutout": 0}), ("no-whiten", {"ablate": "whiten"}),
                    ("no-dirac", {"ablate": "dirac"}), ("no-frozen-gamma", {"ablate": "gamma"})]
        for name, extra in variants:
            merged = dict(cfg)
            if extra is None:  # read the no-TTA number the same run already reports
                r = run(merged, 0, ep)
                results.append(describe(name, [r["final_val_notta"]]))
                continue
            merged.update(extra)
            r = run(merged, 0, ep)
            results.append(describe(name, [r["final_val"]]))

    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
