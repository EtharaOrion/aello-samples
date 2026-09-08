"""Oracle solve for AELLO-C8-S9. PRIVATE. Never reaches the agent-visible surface.

WHAT THIS IS. A causal, closed-form forecaster for OT over the private future window, built from
the five levers seed/contract.yaml records for this slot and from nothing else. It is measured,
not asserted: solution/grounding.yaml carries its graded skill, its per-block distribution, and
the full 32-point power set over these five levers, all produced by running this file's own
`fit_and_forecast` under each configuration.

WHY A CLOSED FORM. The graded quantity is forecast error on a genuine future, and the thing that
decides that error on this substrate is not model capacity. It is the two decisions that come
before any fitting -- which rows you fit on, and what level you anchor the forecast to. Ridge
with an iteratively reweighted L1 refit is enough to expose those decisions and small enough to
leave them nowhere to hide. A larger learner would move the number without moving the lesson,
and it would make the reference's own determinism harder to prove than the task it grades.

THE MEASUREMENT THAT MOTIVATES THE WHOLE DESIGN. Over the three delivered segments:

    segment   OT mean   OT sd   sign of corr(OT, HUFL)
    train      16.290   8.346   +0.157
    val         3.764   2.215   -0.294
    graded      9.019   2.846   -0.231

The level moves by more than the within-segment spread, and the covariate-to-target relation
REVERSES SIGN between train and val. A model fitted on all of history, centred on history's own
mean, is fitting a regime that has stopped existing. That single fact is why L1 and L5 carry the
ramp on this slot and why a leave-one-out ablation of either of them costs about 45 percent of
the floor-to-reference span while the covariate lags cost 11.

THE FIVE LEVERS, as this file binds them:

  L1  drift anchoring        centre the forecast on the last OBSERVED OT and standardise every
                             feature with statistics from the fit window alone. Off: centre on
                             the fit window's mean, which is the drifted level.
  L2  lookback allocation    lags 0, 1, 4 and 96 on all six covariates, spending the 96-row
                             lookback budget on a spread of scales. Off: the contemporaneous
                             row only.
  L3  seasonal decomposition daily and weekly harmonics on the absolute row index, taken out
                             before the learned component sees the covariates. Off: none.
  L4  loss shaping           an IRLS refit toward the L1 loss the slot is graded on, instead of
                             stopping at the L2 solution the normal equations hand you.
  L5  time-ordered selection choose the fit window and the ridge penalty on a holdout that is
                             the LAST 15 percent of the fit rows. Off: the same 15 percent drawn
                             by a shuffled permutation, which is the trap this slot names.

CAUSALITY, stated so it can be checked rather than believed. Features for graded row t use
covariate rows t, t-1, t-4 and t-96 of the concatenated stream, and t-96 is the deepest reach,
so no feature for a row in block b touches a row above b's own ceiling. Standardisation
statistics come from the fit window, which ends at HISTORY_ROWS. Nothing in this file opens
tests/heldout/. The conditioning manifest it writes is therefore true, and tests/test_output.py
checks it against the same arithmetic in environment/ettm1_window.py.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "environment"))
import ettm1_window as W  # noqa: E402

SUBMISSION_DIR = os.environ.get("AELLO_SUBMISSION", "/submission")
FORECAST_PATH = os.path.join(SUBMISSION_DIR, "forecast.npy")
MANIFEST_PATH = os.path.join(SUBMISSION_DIR, "conditioning.json")

# Candidate fit windows, named by how far back they reach from the end of observed history.
# Every one of them ENDS at HISTORY_ROWS; only the start moves, so no candidate can reach the
# graded window and L5 chooses between them on evidence rather than on taste.
FIT_WINDOWS = (("val_only", W.VAL_ROWS),
               ("recent_30d", 96 * 30),
               ("all_history", W.HISTORY_ROWS - W.LOOKBACK))
RIDGE_PENALTIES = (1.0, 10.0, 100.0, 1000.0)
COVARIATE_LAGS = (0, 1, 4, 96)
HARMONIC_PERIODS = (96.0, 672.0)              # one day and one week, in rows
HARMONIC_ORDERS = (1, 2)
HOLDOUT_FRACTION = 0.15
IRLS_ROUNDS = 6
IRLS_RESIDUAL_FLOOR = 0.05
SHUFFLE_SEED = 20260902                       # used ONLY by the L5-off arm, which is the trap
ALL_LEVERS = (1, 1, 1, 1, 1)


def design_matrix(stream, start, count, use_lags, use_harmonics):
    """Features for `count` consecutive rows beginning at absolute index `start`.

    `stream` is the whole covariate series. The deepest lag is 96, so `start` must be at least
    96, and every column is read at an index at or below the row it describes. That is the only
    property that makes this design causal, and it is a property of the slicing, not of a
    promise made about it.
    """
    import numpy as np
    lags = COVARIATE_LAGS if use_lags else (0,)
    columns = [stream[start - lag: start - lag + count] for lag in lags]
    if use_harmonics:
        rows = (start + np.arange(count)).astype("float64")
        for period in HARMONIC_PERIODS:
            for order in HARMONIC_ORDERS:
                angle = 2.0 * np.pi * order * rows / period
                columns.append(np.sin(angle)[:, None])
                columns.append(np.cos(angle)[:, None])
    return np.hstack(columns)


def ridge_weights(features, residual, penalty, robust):
    """Ridge, then optionally an IRLS refit toward absolute error.

    The normal equations minimise squared error; the slot is graded on absolute error. The
    reweighting below is the cheapest honest way to move the solution toward the loss that is
    actually scored, and the residual floor keeps a row whose residual is near zero from
    acquiring unbounded weight.
    """
    import numpy as np
    identity = penalty * np.eye(features.shape[1])
    weights = np.linalg.solve(features.T @ features + identity, features.T @ residual)
    if not robust:
        return weights
    for _ in range(IRLS_ROUNDS):
        scale = 1.0 / np.maximum(np.abs(residual - features @ weights), IRLS_RESIDUAL_FLOOR)
        scaled = features * scale[:, None]
        weights = np.linalg.solve(scaled.T @ features + identity, scaled.T @ residual)
    return weights


def holdout_indices(n, time_ordered):
    """Split the fit rows into an inner fit part and an inner holdout part.

    Time-ordered: the holdout is the newest rows, so selecting on it asks the question the
    graded window will ask. Shuffled: the holdout is drawn at random from the whole span, so
    every held-out row sits between two rows the model has already seen and the selection
    measures interpolation. The second is the failure mode this slot exists to catch, and it is
    reproduced here exactly so its cost can be measured rather than argued about.
    """
    import numpy as np
    cut = int((1.0 - HOLDOUT_FRACTION) * n)
    if time_ordered:
        return np.arange(cut), np.arange(cut, n)
    order = np.random.default_rng(SHUFFLE_SEED).permutation(n)
    return order[:cut], order[cut:]


def select(stream, observed, cfg):
    """Choose the fit window and the ridge penalty under this configuration's own rules."""
    import numpy as np
    use_anchor, use_lags, use_harmonics, robust, time_ordered = cfg
    anchor = float(observed[-1])
    best = None
    for name, length in FIT_WINDOWS:
        start = W.HISTORY_ROWS - length
        features = design_matrix(stream, start, W.HISTORY_ROWS - start, use_lags, use_harmonics)
        target = observed[start:]
        mean, sd = features.mean(0), features.std(0)
        sd[sd == 0] = 1.0
        scaled = (features - mean) / sd
        fit_rows, holdout = holdout_indices(len(scaled), time_ordered)
        centre = anchor if use_anchor else float(target.mean())
        for penalty in RIDGE_PENALTIES:
            weights = ridge_weights(scaled[fit_rows], target[fit_rows] - centre, penalty, robust)
            error = float(np.abs(scaled[holdout] @ weights + centre - target[holdout]).mean())
            if best is None or error < best["inner_mae"]:
                best = {"inner_mae": error, "window": name, "start": start,
                        "penalty": penalty, "mean": mean, "sd": sd, "centre": centre}
    return best


def fit_and_forecast(cfg=ALL_LEVERS, data_dir=None):
    """The whole solve. Returns the (108, 96, 1) forecast and the record of how it was chosen."""
    import numpy as np
    stream, graded_start = W.covariate_stream(data_dir)
    observed = W.observed_target(data_dir)
    chosen = select(stream, observed, cfg)
    use_anchor, use_lags, use_harmonics, robust, _ = cfg
    features = design_matrix(stream, chosen["start"], W.HISTORY_ROWS - chosen["start"],
                             use_lags, use_harmonics)
    scaled = (features - chosen["mean"]) / chosen["sd"]
    weights = ridge_weights(scaled, observed[chosen["start"]:] - chosen["centre"],
                            chosen["penalty"], robust)
    graded = design_matrix(stream, graded_start, W.GRADED_ROWS, use_lags, use_harmonics)
    flat = ((graded - chosen["mean"]) / chosen["sd"]) @ weights + chosen["centre"]
    forecast = flat.reshape(W.BLOCKS, W.HORIZON, W.N_TARGETS)
    record = {"cfg": list(cfg), "window": chosen["window"], "penalty": chosen["penalty"],
              "inner_mae": round(chosen["inner_mae"], 6), "centre": round(chosen["centre"], 6),
              "features": int(features.shape[1]), "fit_rows": int(W.HISTORY_ROWS - chosen["start"])}
    return forecast.astype("float32"), record


def conditioning_manifest(cfg=ALL_LEVERS):
    """What this forecast consumed, per block, in the terms the checker reads.

    Deepest reach for a row at context index i is i - 96, and the newest row any block-b feature
    touches is b's own last row. So the honest declaration for block b is its ceiling. When the
    lag lever is off the design still reads the contemporaneous row, so the ceiling is unchanged
    and only `lookback_rows` moves.
    """
    return {"block_last_context_row": [W.conditioning_ceiling(b) for b in range(W.BLOCKS)],
            "lookback_rows": max(COVARIATE_LAGS) if cfg[1] else 0,
            "fit_rows_end": W.HISTORY_ROWS,
            "method": "ridge_irls_on_lagged_covariates_anchored_on_last_observation",
            "seed": SHUFFLE_SEED if not cfg[4] else None}


def main():
    import time
    import numpy as np
    started = time.time()
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    forecast, record = fit_and_forecast(ALL_LEVERS)
    np.save(FORECAST_PATH, forecast, allow_pickle=False)
    manifest = conditioning_manifest(ALL_LEVERS)
    manifest["budget"] = {"attempts_used": 1, "wall_seconds": round(time.time() - started, 3),
                          "horizon": W.HORIZON, "lookback_rows": manifest["lookback_rows"],
                          "device": "cpu", "fitted_parameters": record["features"]}
    with open(MANIFEST_PATH, "w") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"forecast": FORECAST_PATH, "manifest": MANIFEST_PATH,
                      "selection": record}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-3338d48cf6aaaab9fed70d3999c4eef3
#   slot1: AELLO-CANARY-59f56b16c9be9274bfe9c2f55513d4ee
#   slot2: AELLO-CANARY-3f73f40ec5070d2d02b1d437e606d45b
#   slot3: AELLO-CANARY-d1e3166e80845b74480c81931491d895
# AELLO-CANARY-END
