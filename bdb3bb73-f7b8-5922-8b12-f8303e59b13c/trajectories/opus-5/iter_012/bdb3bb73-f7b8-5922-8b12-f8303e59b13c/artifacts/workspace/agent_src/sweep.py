"""Run several training configs inside one process (amortises import + data load).

Usage: python sweep.py '<json list of arg-dicts>' [--tag NAME]
All sweeps use --holdout 5000 (a fixed, seed-1234 slice of the TRAIN split); the test
split is never read here.
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
import train as T  # noqa: E402


def run(cfgs, tag=""):
    out = []
    for i, c in enumerate(cfgs):
        argv = []
        for k, v in c.items():
            if isinstance(v, (list, tuple)):
                argv += [f"--{k}"] + [str(x) for x in v]
            else:
                argv += [f"--{k}", str(v)]
        a = T.get_args(argv)
        r = T.train(a)
        rec = {"tag": tag, "cfg": c, "val_acc": r.get("val_acc"), "wall": round(r["wall_clock_sec"], 2),
               "t_star": round(r.get("t_star_if_last", 0), 2)}
        out.append(rec)
        print("RESULT " + json.dumps(rec), flush=True)
    print("\n=== summary (" + tag + ") ===")
    for rec in sorted(out, key=lambda r: -(r["val_acc"] or 0)):
        print(f"{rec['val_acc']:.4f}  t*={rec['t_star']:6.2f}  {json.dumps(rec['cfg'])}")
    return out


if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    for c in cfgs:
        c.setdefault("holdout", 5000)
        c.setdefault("save", 0)
        c.setdefault("log", "/workspace/runs.jsonl")
    run(cfgs, tag)
