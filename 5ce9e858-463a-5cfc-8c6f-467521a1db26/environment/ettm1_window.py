"""Graded-window geometry for AELLO-C8-S9, shared by everything that touches the corpus.

WHY THIS FILE EXISTS. The whole slot turns on ONE question -- which rows of the series a given
forecast is allowed to have looked at -- and that question has an arithmetic answer, not a
stylistic one. If the starter, the reference, the checker and the verifier each recompute the
index arithmetic from the shapes they happen to load, they will eventually disagree by one row,
and a one-row disagreement in a causality boundary is indistinguishable from a leak. So the
arithmetic lives here, once, and every consumer imports it.

THE SERIES. ETTm1 at a fifteen-minute cadence, seven columns: six load covariates followed by
OT, the oil temperature that is the forecast target. The bundle ships the series cut into three
contiguous, time-ordered segments and never shuffles them:

    rows      0 .. 48775   environment/data/ettm1_train.npy          7 columns, OT observed
    rows  48776 .. 59227   environment/data/ettm1_val.npy            7 columns, OT observed
    rows  59228 .. 69679   environment/data/ettm1_graded_context.npy 6 columns, OT WITHHELD

The third segment is the private future. Its covariates ship; its OT does not exist anywhere the
agent can read. That withholding is not a convention to be respected, it is a property of the
delivered bytes, and tests/isolation.py proves it rather than asserting it.

THE GRADED SURFACE. The graded context is tiled into non-overlapping blocks of HORIZON rows.
HORIZON is 96, which is exactly one diurnal cycle at this cadence: 96 * 15 minutes = 24 hours.
10452 rows do not divide by 96, so the final partial block is dropped and the graded surface is
108 whole blocks covering 10368 rows. The drop lands at the END of the series, which is the safe
direction: removing the newest rows removes nothing from anyone's conditioning set, whereas
dropping from the front would silently change which rows a lookback may reach.

THE CAUSALITY RULE. A forecast for block b may condition on:

    every row of train and val, all of which precede the graded window; and
    graded-context covariate rows with index strictly less than (b + 1) * HORIZON,
    reaching back at most LOOKBACK rows before the row being predicted.

It may not condition on a graded-context row belonging to a LATER block. That last clause is the
one a global normalisation breaks: computing a mean or a standard deviation over all 10452
context rows uses block 107 to forecast block 0, and a statistic is a use. The rule is stated
here in the same terms the submitted conditioning manifest is written in, so an honest agent can
check its own compliance before the verifier does.
"""
import os

# --- The three delivered segments, by row count. Fixed by the shipped bytes. -------------
TRAIN_ROWS = 48776
VAL_ROWS = 10452
CONTEXT_ROWS = 10452
HISTORY_ROWS = TRAIN_ROWS + VAL_ROWS          # rows with OT observed: 59228
SERIES_ROWS = HISTORY_ROWS + CONTEXT_ROWS     # the whole series: 69680

# --- Column layout. Six covariates, then the target. -------------------------------------
COVARIATES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL")
TARGET = "OT"
N_COVARIATES = len(COVARIATES)
N_TARGETS = 1

# --- Task parameters. AUTHORED, and authored is the right word: the objective names a fixed
# --- horizon and a fixed lookback budget, so these are terms of the problem rather than
# --- measurements of it. Both are one diurnal cycle at the substrate's own cadence.
HORIZON = 96                                  # 96 * 15 minutes = 24 hours
LOOKBACK = 96                                 # rows of covariate history a forecast may reach
CADENCE_MINUTES = 15
BLOCKS = CONTEXT_ROWS // HORIZON              # 108
GRADED_ROWS = BLOCKS * HORIZON                # 10368
DROPPED_ROWS = CONTEXT_ROWS - GRADED_ROWS     # 84, the final partial block
FORECAST_SHAPE = (BLOCKS, HORIZON, N_TARGETS)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_FILE = "ettm1_train.npy"
VAL_FILE = "ettm1_val.npy"
CONTEXT_FILE = "ettm1_graded_context.npy"


def block_bounds(b):
    """Half-open [start, end) row range of block b inside the graded context."""
    if not 0 <= b < BLOCKS:
        raise IndexError("block %r is outside 0..%d" % (b, BLOCKS - 1))
    return b * HORIZON, (b + 1) * HORIZON


def block_of(row):
    """Which graded block a context row belongs to; None for a dropped tail row."""
    b = row // HORIZON
    return b if 0 <= b < BLOCKS else None


def conditioning_ceiling(b):
    """The highest graded-context row index block b's forecast may condition on.

    Its own block's last row. Anything above this belongs to a later block and is the future
    relative to the forecast being made, whatever mechanism reaches it.
    """
    return block_bounds(b)[1] - 1


def manifest_row_admissible(b, row):
    """Is a declared last-consumed context row legal for block b?

    -1 declares that the forecast consumed no graded-context row at all, which a pure
    climatology or a pure persistence forecast honestly does. Any other value must land inside
    block b: below its start is not wrong but is unrepresentable in this manifest, because the
    manifest records the LAST row consumed and a forecast that reached block b-1 and stopped
    there declares b-1's ceiling under its own entry.
    """
    if row == -1:
        return True
    lo, hi = block_bounds(b)
    return lo - LOOKBACK <= row <= hi - 1


def load_segment(name, data_dir=None):
    import numpy as np
    return np.load(os.path.join(data_dir or DATA_DIR, name), allow_pickle=False)


def load_history(data_dir=None):
    """train and val stacked: every row whose OT is observed, in time order."""
    import numpy as np
    train = load_segment(TRAIN_FILE, data_dir)
    val = load_segment(VAL_FILE, data_dir)
    stacked = np.vstack([train, val]).astype("float64")
    if stacked.shape != (HISTORY_ROWS, N_COVARIATES + N_TARGETS):
        raise ValueError("observed history is %r, not %r"
                         % (stacked.shape, (HISTORY_ROWS, N_COVARIATES + N_TARGETS)))
    return stacked


def load_context(data_dir=None):
    """The graded window's covariates. Six columns; OT is absent by construction."""
    ctx = load_segment(CONTEXT_FILE, data_dir).astype("float64")
    if ctx.shape != (CONTEXT_ROWS, N_COVARIATES):
        raise ValueError("graded context is %r, not %r"
                         % (ctx.shape, (CONTEXT_ROWS, N_COVARIATES)))
    return ctx


def covariate_stream(data_dir=None):
    """Covariates for the whole series, history first, so a lookback can cross the seam.

    Returned with the index of the first graded-context row, because every causality question
    in this slot is asked as 'is this index below that one'.
    """
    import numpy as np
    history = load_history(data_dir)
    ctx = load_context(data_dir)
    return np.vstack([history[:, :N_COVARIATES], ctx]), HISTORY_ROWS


def observed_target(data_dir=None):
    """OT over the observed history. The last entry is the anchor every control is read against."""
    return load_history(data_dir)[:, -1]


def admissible_domain(data_dir=None):
    """The interval a forecast value must lie inside, MEASURED from observed OT alone.

    Bounds are the observed OT range widened by the largest single-step change the sensor has
    ever made across 59228 observations. No margin is authored: a forecast that leaves the
    envelope the instrument has recorded, by more than the instrument's own largest jump, is
    out of domain rather than merely wrong, and it is rejected before any error is computed.
    """
    import numpy as np
    ot = observed_target(data_dir)
    jump = float(np.abs(np.diff(ot)).max())
    return float(ot.min()) - jump, float(ot.max()) + jump


def describe():
    """Everything a consumer needs about the geometry, as plain data."""
    return {"cadence_minutes": CADENCE_MINUTES, "horizon": HORIZON, "lookback": LOOKBACK,
            "blocks": BLOCKS, "graded_rows": GRADED_ROWS, "dropped_rows": DROPPED_ROWS,
            "context_rows": CONTEXT_ROWS, "history_rows": HISTORY_ROWS,
            "series_rows": SERIES_ROWS, "n_covariates": N_COVARIATES,
            "n_targets": N_TARGETS, "forecast_shape": list(FORECAST_SHAPE)}
