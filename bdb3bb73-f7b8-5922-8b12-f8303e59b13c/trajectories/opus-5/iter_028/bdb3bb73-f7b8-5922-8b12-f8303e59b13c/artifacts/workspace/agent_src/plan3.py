# Pre-registered round 3. Rounds 1-2 tuned augmentation to a local optimum
# (cutout=12, translate=2). Round 3 leaves augmentation fixed and probes the
# optimizer/loss constants, which were inherited unchanged from airbench's
# 64/256/256 net and have never been re-tuned for this wider 128/384/512 trunk.
# That is the most likely remaining source of a systematic gap.
#
# Incumbent = cutout12 @ 24ep (round-1 val 94.58). Noise floor 0.06pp.
# Decision rule unchanged: promote only on a 2-seed mean beating the incumbent
# by more than the noise floor, and only if it survives a 3-seed re-check.

BASE = dict(epochs=24, lr_end=0.02, cutout=12)

PLAN = [
    ("incumbent",   [0, 1], dict(BASE)),
    ("lr9",         [0, 1], dict(BASE, lr=9.0)),
    ("lr14",        [0, 1], dict(BASE, lr=14.0)),
    ("ls0.10",      [0, 1], dict(BASE, label_smoothing=0.10)),
    ("wd0.0230",    [0, 1], dict(BASE, wd=0.0230)),
]
