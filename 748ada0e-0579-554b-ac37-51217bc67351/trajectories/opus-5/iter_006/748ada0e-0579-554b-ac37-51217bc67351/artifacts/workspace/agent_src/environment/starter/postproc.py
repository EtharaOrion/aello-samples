"""Prediction-time prior correction, tuned on the public val split.

The graded metric is BALANCED accuracy (mean per-class recall).  A model trained with
plain cross-entropy estimates p(y|x) under the *training* class prior, which is mildly
non-uniform here (1.39x between the largest and smallest class).  Subtracting tau *
log p_train(y) from the log-probs converts that towards a likelihood, which is the
Bayes-optimal rule for balanced accuracy when tau = 1 and the test prior is uniform.

tau is a single scalar chosen on the 2000-image val split; no retraining, no extra
presentations.  Run after a training run that was launched with --save-logits.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aello_data as data


def balanced_acc(pred, y, n_classes=100):
    return float(np.mean([(pred[y == c] == c).mean()
                          for c in range(n_classes) if (y == c).sum() > 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True, help="graded logits saved by train.py")
    ap.add_argument("--out", default=None, help="where to write corrected predictions")
    a = ap.parse_args()

    lg_g = np.load(a.logits).astype(np.float32)
    lg_v = np.load(a.logits.replace(".npy", "_val.npy")).astype(np.float32)
    _, yv = data.load_val()
    _, ytr = data.load_train("i20k")
    cnt = np.bincount(ytr, minlength=data.N_CLASSES).astype(np.float64)
    logprior = np.log(cnt / cnt.sum()).astype(np.float32)
    print("class counts: min=%d max=%d ratio=%.2f" % (cnt.min(), cnt.max(),
                                                      cnt.max() / cnt.min()))

    best = (None, -1.0)
    for tau in np.arange(0.0, 1.51, 0.1):
        acc = balanced_acc((lg_v - tau * logprior).argmax(1), yv)
        print("  tau=%.2f  val_balanced_acc=%.4f" % (tau, acc))
        if acc > best[1]:
            best = (float(tau), acc)
    print("best tau=%.2f  val=%.4f  (tau=0 baseline %.4f)"
          % (best[0], best[1], balanced_acc(lg_v.argmax(1), yv)))

    if a.out:
        pred = (lg_g - best[0] * logprior).argmax(1).astype(np.int64)
        assert pred.shape == (lg_g.shape[0],) and pred.min() >= 0 and pred.max() < 100
        np.save(a.out, pred)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
