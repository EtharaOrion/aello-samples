"""Recompute this slot's graded axis from delivered bytes, so it can be checked rather than read.

WHY A SEPARATE MODULE. tests/constants.json publishes the window geometry and the admissible
forecast domain. A published constant that nothing recomputes is an assertion, and an assertion
in a constants file is exactly the kind of thing that goes stale without anyone noticing. Every
number this module can derive from the shipped corpus, it derives, and `python tests/anchor.py`
prints the derivation beside the publication so a disagreement is visible in one line.

WHAT THE AXIS IS. The graded quantity is an ERROR and errors are lower-better, so it cannot be
the axis: the reward map needs higher-better and bounded above. The axis is therefore a SKILL
SCORE against a persistence baseline,

    skill = 1 - MAE(forecast) / MAE(persistence)

with persistence being the last OBSERVED oil temperature held flat across the whole graded
window. Skill is 1 for a perfect forecast, 0 for a forecast exactly as good as persistence, and
negative for one that is worse; the reward ramp clamps the negative half.

WHY THAT PERSISTENCE AND NOT ANOTHER. The measured ladder at seed/ladders/ladder_c8s9.json
anchors on OT one step before each target, which is the right anchor for the protocol IT ran:
that ladder rebuilt the series with OT observed throughout and forecast 24 steps ahead with
feedback. This bundle ships no OT inside the graded window at all, so an anchor that reads OT
one row before each block is an anchor the agent could never compute and every honest submission
would score deeply negative against it. Measured on the delivered corpus: block-origin
persistence reaches MAE 1.3707 while the strongest causal forecast in this bundle's own power
set reaches 3.3345. Anchoring on a baseline no submission can construct would put the whole
admissible field below zero and flatten the ramp, so the delivered axis anchors on the last
observation, which every submission can construct and which the contract names as CTL-PERSIST.

CONSEQUENCE, stated because it is load-bearing: skill(persistence) is exactly 0.0 by
construction, not approximately. `python tests/anchor.py` checks that identity, and an identity
that fails is a broken axis rather than a bad forecast.

PRIVACY. The private targets live under tests/heldout/ and are not readable from the agent
sandbox. Everything this module derives from the public corpus is printed unconditionally;
anything that needs the private window is printed only when the private file is actually
readable, and its absence is reported as an absence rather than a failure.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BUNDLE, "environment"))
import ettm1_window as W  # noqa: E402

HELDOUT = os.path.join(HERE, "heldout", "ettm1_graded_targets.npy")


def constants():
    with open(os.path.join(HERE, "constants.json")) as handle:
        return json.load(handle)


def private_targets():
    """The graded window's OT, or None when the private file is not readable from here."""
    if not os.path.exists(HELDOUT):
        return None
    import numpy as np
    return np.load(HELDOUT, allow_pickle=False).astype("float64")[:W.GRADED_ROWS]


def persistence_forecast():
    """The last observed oil temperature, held flat across every graded block."""
    import numpy as np
    return np.full((W.BLOCKS, W.HORIZON), float(W.observed_target()[-1]))


def block_absolute_error(forecast, truth):
    """Mean absolute error per graded block. Shape (108,)."""
    import numpy as np
    f = np.asarray(forecast, dtype="float64").reshape(W.BLOCKS, W.HORIZON)
    t = np.asarray(truth, dtype="float64").reshape(W.BLOCKS, W.HORIZON)
    return np.abs(f - t).mean(1)


def skill(forecast, truth):
    """Pooled skill against persistence, and the per-block distribution behind it.

    Pooled is computed on the pooled errors rather than by averaging the per-block skills. The
    two differ whenever the blocks carry unequal persistence error, and they do here: block
    persistence MAE spans 0.5789 to 10.8070. Pooled is the graded number because it weights
    every graded row equally, which is what 'mean absolute error over the private future
    window' says. The per-block distribution is reported alongside it because a pooled number
    hides whether a forecast is uniformly decent or excellent on eight blocks and broken on two.
    """
    import numpy as np
    baseline = persistence_forecast()
    pooled_ref = float(np.abs(np.asarray(truth, dtype="float64").reshape(W.BLOCKS, W.HORIZON)
                              - baseline).mean())
    f = np.asarray(forecast, dtype="float64").reshape(W.BLOCKS, W.HORIZON)
    t = np.asarray(truth, dtype="float64").reshape(W.BLOCKS, W.HORIZON)
    pooled_mae = float(np.abs(f - t).mean())
    per_block = 1.0 - block_absolute_error(f, t) / block_absolute_error(baseline, t)
    return {"mae": pooled_mae, "mae_persistence": pooled_ref,
            "skill": 1.0 - pooled_mae / pooled_ref,
            "per_block_mean": float(per_block.mean()),
            "per_block_sd": float(per_block.std(ddof=1)),
            "blocks_above_persistence": int((per_block > 0).sum())}


def derived():
    """Everything derivable from the PUBLIC corpus alone."""
    lo, hi = W.admissible_domain()
    geometry = W.describe()
    geometry["domain_low"] = lo
    geometry["domain_high"] = hi
    geometry["anchor_value"] = float(W.observed_target()[-1])
    return geometry


def compare():
    """Derived against published, key by key, for every key the constants file carries."""
    published, out = constants(), {}
    for key, value in sorted(derived().items()):
        if key not in published:
            continue
        got = published[key]
        agree = (got == value if isinstance(value, (int, list)) and not isinstance(value, bool)
                 else got is not None and abs(float(got) - float(value)) <= 1e-9)
        out[key] = {"derived": value, "published": got, "agree": bool(agree)}
    return out


def main():
    report = {"geometry": derived(), "constants_agree": compare()}
    truth = private_targets()
    if truth is None:
        report["private_window"] = "unreadable-from-here"
    else:
        identity = skill(persistence_forecast(), truth)
        report["private_window"] = {
            "persistence_mae": round(identity["mae_persistence"], 6),
            "persistence_skill_is_exactly_zero": identity["skill"] == 0.0,
            "block_mae_min": round(float(block_absolute_error(persistence_forecast(),
                                                              truth).min()), 6),
            "block_mae_max": round(float(block_absolute_error(persistence_forecast(),
                                                              truth).max()), 6)}
    disagreements = sorted(k for k, v in report["constants_agree"].items() if not v["agree"])
    report["disagreements"] = disagreements
    print(json.dumps(report, indent=1, sort_keys=True))
    if isinstance(report["private_window"], dict) and \
            not report["private_window"]["persistence_skill_is_exactly_zero"]:
        print("FAILED: the persistence anchor does not score exactly zero on its own axis",
              file=sys.stderr)
        return 1
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
