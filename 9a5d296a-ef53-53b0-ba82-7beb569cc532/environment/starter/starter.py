"""Shipped baseline for AELLO-C8-S10. Correct, unoptimised, and beatable. Replace it.

It reads what ships, trains something real, writes a well-formed artifact at the graded
path, and declares its own budget accounting. It is deliberately weak, in three ways it
does not hide:

  1. It ignores `neighbour_mean` entirely and classifies each paper from its own embedding.
     The graph is the strongest single lever on this substrate and this baseline declines
     it.
  2. It does not standardise the features, so the fit is at the mercy of whatever scale the
     embedding columns happen to have.
  3. It weights the loss by inverse class frequency.

THE THIRD ONE IS NOT LIKE THE OTHER TWO, and this is the whole task.

OGBN-arxiv is imbalanced by a factor of 942 between its largest and smallest class. Every
instinct trained on imbalanced data says: reweight. Reweighting is what the shipped
baseline does, and it is textbook-correct -- for a metric that averages over CLASSES.

The graded quantity here averages over ROWS. It is plain accuracy over the 2019 papers, and
under plain accuracy a correct prediction on a 27321-row class and a correct prediction on
a 29-row class are worth exactly the same. Inverse-frequency weighting spends the model's
capacity on the classes that carry almost none of the rows, and it buys that with rows it
was already getting right.

Both effects are measured in solution/grounding.yaml, on the delivered bytes, through this
file. Neither is asserted. You are not being told which way to go; you are being told that
the question exists and that the answer is not the one the imbalance headline suggests.

The other trap is the calendar. train_idx is every paper up to 2017 and the graded fold is
2019. The label prior is not the same in those two years -- it is not even close, and the
single most common class changes. Anything you fit to the training prior, including a
reweighting derived from it, is fitted to a year that has already ended.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aello_graph as G  # noqa: E402

SEED = 0
USE_NEIGHBOURS = False       # lever: bring the one-hop aggregate in
KEEP_SELF = True             # lever: keep the paper's own embedding beside the aggregate
STANDARDISE = False          # lever: centre and scale on training-year statistics
CLASS_WEIGHTED = True        # the planted defect. See the module docstring.

# Declared budget. instruction.md fixes these; a submitted pipeline that changes them is
# outside the graded envelope and tests/budget.py reads them without executing anything.
EPOCH_BUDGET = G.EPOCH_BUDGET
EXPANSIONS_PER_EPOCH = G.EXPANSIONS_PER_EPOCH
HOP_DEPTH = G.HOP_DEPTH


def main():
    graph = G.load_graph()
    features = G.build_features(graph, use_neighbours=USE_NEIGHBOURS,
                                keep_self=KEEP_SELF, standardise=STANDARDISE)
    W, spend = G.fit_softmax(features, graph["train_idx"], graph["y_train"], SEED,
                             weighted=CLASS_WEIGHTED, epochs=EPOCH_BUDGET)

    val_pred = G.predict(W, features, graph["val_idx"])
    val_acc = G.accuracy(val_pred, graph["y_val"])
    val_bacc = G.balanced_accuracy(val_pred, graph["y_val"])

    graded_pred = G.predict(W, features, graph["graded_idx"])
    G.write_submission(graded_pred)

    G.write_run_record(
        {"epochs": spend["epochs"],
         "nodes_presented": spend["nodes_presented"],
         "hop_depth": spend["hop_depth"],
         "neighbourhood_expansions": spend["neighbourhood_expansions"],
         "expansions_per_epoch": EXPANSIONS_PER_EPOCH,
         "seed": SEED},
        measured_on_val={"accuracy": round(val_acc, 6),
                         "balanced_accuracy": round(val_bacc, 6)})

    # Reported on the PUBLIC 2018 fold, labelled as such. The 2019 fold is not measurable
    # from here and nothing printed below is a claim about it.
    print("val 2018: accuracy %.5f  balanced_accuracy %.5f" % (val_acc, val_bacc))
    print("graded rows written: %d" % len(graded_pred))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
