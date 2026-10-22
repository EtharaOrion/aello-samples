# Pre-registered screening plan (written BEFORE the runs).
#
# Observation driving it: full-train test accuracy plateaus at ~93.9% for both
# 18 and 24 epochs, so the schedule is not the binding constraint -- the model
# is regularization-limited. Each hypothesis below targets that.
#
# Baseline gets 3 seeds so the seed-to-seed spread gives a noise floor to judge
# every other delta against. Screening arms get 2 seeds.
#
# Decision rule (fixed in advance): promote an arm only if its 2-seed mean
# beats the baseline mean by more than the baseline seed standard deviation.

BASE = dict(epochs=24, lr_end=0.02)

PLAN = [
    ("baseline-24ep",      [0, 1, 2], dict(BASE)),
    ("cutout8",            [0, 1],    dict(BASE, cutout=8)),
    ("cutout12",           [0, 1],    dict(BASE, cutout=12)),
    ("translate4",         [0, 1],    dict(BASE, translate=4)),
    ("cutout8-translate4", [0, 1],    dict(BASE, cutout=8, translate=4)),
    ("whitenbias6",        [0, 1],    dict(BASE, whiten_bias_epochs=6)),
]
