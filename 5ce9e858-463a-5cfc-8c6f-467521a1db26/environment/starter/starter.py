"""Runnable baseline for AELLO-C8-S9. Correct, honest, causal -- and deliberately weak.

WHAT IT DOES. It reads the last 96 observed OT rows, treats them as a diurnal profile, and
repeats that profile across all 108 graded blocks. That is a seasonal-naive forecast: it carries
the shape of a day forward and assumes nothing else. It writes the graded artifact, and it writes
the conditioning manifest that every submission owes, so the delivery path is exercised end to
end before you change a line of it.

WHAT IT IS NOT. It is not a good forecast and it is not trying to be. Its purpose is to give you
a working shape to replace. Replace all of it if you like; nothing here is load-bearing except
the two files it writes and where it writes them.

ONE THING WORTH KNOWING BEFORE YOU BUILD ON IT. This baseline picks its diurnal profile from a
particular place in the series. Whether that place is the right one is a question about time
order, and time order is what this task is about. The profile it uses is stale by construction:
read `profile_rows()` and decide for yourself which rows a forecast made at the boundary of the
observed history should actually be carrying forward. That decision is worth more than any of
the modelling that comes after it.

CAUSALITY. This forecast consumes no graded-context row at all -- it never looks at the future
covariates it is given -- so every entry of its conditioning manifest is -1. That is an honest
declaration, not an empty one: -1 means 'consumed nothing from the graded window', and it is the
only declaration a forecast built purely from observed history is entitled to make.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ettm1_window as W  # noqa: E402

SUBMISSION_DIR = os.environ.get("AELLO_SUBMISSION", "/submission")
FORECAST_PATH = os.path.join(SUBMISSION_DIR, "forecast.npy")
MANIFEST_PATH = os.path.join(SUBMISSION_DIR, "conditioning.json")


def profile_rows():
    """Which observed rows this baseline carries forward as its diurnal profile.

    It reaches into the TRAIN segment and takes the last day there. The val segment sits between
    that day and the graded window, so the profile this returns is 10452 rows -- one hundred and
    nine days -- older than the newest observation the bundle actually gives you.
    """
    import numpy as np
    train = W.load_segment(W.TRAIN_FILE).astype("float64")
    return np.asarray(train[-W.HORIZON:, -1], dtype="float64")


def forecast():
    """Tile the diurnal profile across every graded block. Shape (108, 96, 1)."""
    import numpy as np
    profile = profile_rows()
    if profile.shape != (W.HORIZON,):
        raise ValueError("a diurnal profile must be %d rows, got %r" % (W.HORIZON, profile.shape))
    tiled = np.tile(profile, (W.BLOCKS, 1))
    return tiled.reshape(W.BLOCKS, W.HORIZON, W.N_TARGETS).astype("float32")


def conditioning_manifest():
    """The per-block declaration of what this forecast looked at.

    Every entry is -1 because the baseline reads no graded-context row. `fit_rows_end` is the
    exclusive end of the observed rows it fitted on, which for a seasonal-naive forecast is the
    end of the observed history; nothing beyond it exists in the delivered tree anyway.
    """
    return {"block_last_context_row": [-1] * W.BLOCKS,
            "lookback_rows": 0,
            "fit_rows_end": W.HISTORY_ROWS,
            "method": "seasonal_naive_tiled_diurnal_profile",
            "seed": None}


def budget_declaration(seconds):
    """Your own accounting for the graded run. The declaration is what is graded."""
    return {"attempts_used": 1, "wall_seconds": round(seconds, 3),
            "horizon": W.HORIZON, "lookback_rows": 0,
            "device": "cpu", "fitted_parameters": 0}


def main():
    import time
    import numpy as np
    started = time.time()
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    np.save(FORECAST_PATH, forecast(), allow_pickle=False)
    manifest = conditioning_manifest()
    manifest["budget"] = budget_declaration(time.time() - started)
    with open(MANIFEST_PATH, "w") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"forecast": FORECAST_PATH, "manifest": MANIFEST_PATH,
                      "shape": list(W.FORECAST_SHAPE)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
