"""S3b: extend the one factor that beat the noise floor (cutout), test dirac init,
and screen combinations. Held-out 45k/5k only."""
import sys
sys.path.insert(0,"/workspace")
from sweep import run_grid, summarize
B = dict(scale=0.03125, epochs=32, lr=11.5, wd=0.0051)
COMBO = dict(cutout=12, pad=3, lr_end=0.02, bn_momentum=0.8)
ovs = [
    dict(B, cutout=12), dict(B, cutout=14), dict(B, cutout=16),
    dict(B, dirac=True), dict(B, dirac=True, cutout=12),
    dict(B, **COMBO),
    dict(B, **COMBO, dirac=True),
    dict(B, cutout=12, epochs=48),
]
rows = run_grid("S3b_cutout_dirac", ovs, seeds=[0,1], eval_every=8)
print("\n=== S3b summary (base 32ep = 0.9410, cutout10 = 0.9449) ===")
summarize(rows)
print("DONE_S3B", flush=True)
