"""Apply the pre-registered decision rule to the sweep output.

Nothing here has a free parameter: the noise floor is the per-seed standard
deviation of the baseline, and the significance bar is
`2 * sigma * sqrt(2/n)` as committed in PREREGISTRATION.md section 5.
"""
import argparse
import json
import math
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="/workspace/logs/sweep_r1.json")
    ap.add_argument("--baseline", default="A_baseline")
    args = ap.parse_args()

    res = json.load(open(args.sweep))
    base = [v for _, v in sorted(res[args.baseline].items())]
    sigma = st.stdev(base)
    n_base = len(base)
    mu_base = st.mean(base)

    print(f"baseline {args.baseline}: n={n_base} mean={mu_base:.4f} "
          f"sigma={sigma * 100:.3f} pp   seeds=" +
          " ".join(f"{a:.4f}" for a in base))
    # sigma from n seeds is itself uncertain; chi-square CI for the sd.
    dof = n_base - 1
    from math import sqrt
    try:
        from scipy.stats import chi2
        lo = sigma * sqrt(dof / chi2.ppf(0.975, dof))
        hi = sigma * sqrt(dof / chi2.ppf(0.025, dof))
        print(f"  95% CI on sigma: [{lo * 100:.3f}, {hi * 100:.3f}] pp "
              f"({dof} dof - a 4-seed sigma is a coarse instrument)")
    except Exception:
        pass

    rows = []
    for name, per_seed in res.items():
        v = [x for _, x in sorted(per_seed.items())]
        n = len(v)
        mu = st.mean(v)
        bar = 2 * sigma * math.sqrt(1 / n + 1 / n_base)
        rows.append((mu - mu_base, name, n, mu, bar, mu - mu_base > bar, v))

    rows.sort(reverse=True)
    print(f"\n{'candidate':18s} {'n':>2s} {'mean':>8s} {'delta_pp':>9s} "
          f"{'bar_pp':>7s}  verdict")
    winners = []
    for d, name, n, mu, bar, win, v in rows:
        verdict = "REAL WINNER" if win else "indistinguishable"
        if name == args.baseline:
            verdict = "(baseline)"
        elif win:
            winners.append(name)
        print(f"{name:18s} {n:2d} {mu:8.4f} {d * 100:+9.3f} {bar * 100:7.3f}  {verdict}")

    print("\nDecision:", f"promote {winners}" if winners else
          "NO candidate clears the bar -> baseline ships unchanged "
          "(pre-registration section 5.4)")


if __name__ == "__main__":
    main()
