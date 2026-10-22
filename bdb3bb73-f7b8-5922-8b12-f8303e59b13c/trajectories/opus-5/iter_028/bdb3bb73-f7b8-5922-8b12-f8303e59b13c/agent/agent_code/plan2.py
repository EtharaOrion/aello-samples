# Pre-registered round 2. Round 1 confirmed the regularization diagnosis:
# cutout is monotone increasing in strength up to 12 (+0.57pp vs a 0.06pp
# seed noise floor), while extra geometric translation and a longer whitening
# bias warmup were both null. Round 2 asks two things:
#   (a) where does cutout stop helping (14, 16)?
#   (b) now that the model is regularized, does a longer schedule finally pay
#       off (it did not at cutout=0)?
# Decision rule unchanged: beat the incumbent by more than the 0.06pp noise
# floor, judged on the 2-seed mean.

BASE = dict(epochs=24, lr_end=0.02)

PLAN = [
    ("cutout14",         [0, 1], dict(BASE, cutout=14)),
    ("cutout16",         [0, 1], dict(BASE, cutout=16)),
    ("cutout12-ep32",    [0, 1], dict(BASE, cutout=12, epochs=32)),
    ("cutout12-trans4",  [0, 1], dict(BASE, cutout=12, translate=4)),
]
