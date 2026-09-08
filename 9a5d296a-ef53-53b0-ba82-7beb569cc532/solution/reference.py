"""Reference solve for AELLO-C8-S10. PRIVATE. Never assembled into the agent's package.

It is the shipped starter with three levers moved and nothing else added. It imports
environment/aello_graph.py rather than restating it, so the difference between the starter
and the reference is exactly the four constants below, and every anchor this bundle records
was measured through the same loop the agent can read.

    USE_NEIGHBOURS  False -> True   bring the one-hop aggregate in at all
    STANDARDISE     False -> True   centre and scale on training-year moments
    CLASS_WEIGHTED  True  -> False  the planted defect, repaired
    KEEP_SELF       True  -> True   unchanged; kept explicit because it is a lever

The third is the one the task is built around. It is the only one of the three whose
correct setting is the opposite of what the substrate's headline fact recommends, and it is
the only one whose repair moves the graded metric and the balanced metric in OPPOSITE
directions. solution/grounding.yaml carries both movements, measured over eight seeds.

WHAT THIS IS NOT. It is not a strong solution and it is not meant to be one. It is one
linear model over 256 shipped columns, inside the shipped epoch budget, with no depth, no
attention, no label reuse and no ensembling. It exists to fix the far end of the reward
ramp at something a careful agent can pass rather than at something nothing can reach, and
to prove the delivery path end to end before any grading constant exists. The floor and the
knee themselves are measured by the Phase 2 wave on the grading host and are null here.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "environment"))

import aello_graph as G  # noqa: E402

SEED = 0
USE_NEIGHBOURS = True
KEEP_SELF = True
STANDARDISE = True
CLASS_WEIGHTED = False

EPOCH_BUDGET = G.EPOCH_BUDGET
EXPANSIONS_PER_EPOCH = G.EXPANSIONS_PER_EPOCH
HOP_DEPTH = G.HOP_DEPTH


def solve(seed=SEED, class_weighted=CLASS_WEIGHTED, use_neighbours=USE_NEIGHBOURS,
          keep_self=KEEP_SELF, standardise=STANDARDISE, graph=None):
    """Fit and return (graded predictions, val predictions, budget spend).

    Every keyword is a lever the ablation record varies. Calling it with the defaults is
    the reference; calling it with class_weighted=True is the reference carrying the
    planted defect, which is how the defect's worth was measured.
    """
    graph = G.load_graph() if graph is None else graph
    features = G.build_features(graph, use_neighbours=use_neighbours,
                                keep_self=keep_self, standardise=standardise)
    W, spend = G.fit_softmax(features, graph["train_idx"], graph["y_train"], seed,
                             weighted=class_weighted, epochs=EPOCH_BUDGET)
    return (G.predict(W, features, graph["graded_idx"]),
            G.predict(W, features, graph["val_idx"]), spend)


def main():
    graph = G.load_graph()
    graded_pred, val_pred, spend = solve(graph=graph)
    G.write_submission(graded_pred)
    G.write_run_record(
        {"epochs": spend["epochs"],
         "nodes_presented": spend["nodes_presented"],
         "hop_depth": spend["hop_depth"],
         "neighbourhood_expansions": spend["neighbourhood_expansions"],
         "expansions_per_epoch": EXPANSIONS_PER_EPOCH,
         "seed": SEED},
        measured_on_val={"accuracy": round(G.accuracy(val_pred, graph["y_val"]), 6),
                         "balanced_accuracy": round(
                             G.balanced_accuracy(val_pred, graph["y_val"]), 6)})
    print("reference val 2018 accuracy %.5f" % G.accuracy(val_pred, graph["y_val"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-f4779ea1f0db2ce35b6f9fbf1f352ced
#   slot1: AELLO-CANARY-cb5afcea4331ba82432379b38ea41063
#   slot2: AELLO-CANARY-23bcbe3d45419a0c71ce6b35c3cf53b5
#   slot3: AELLO-CANARY-6b03daab822b0d4c49342646141a7551
# AELLO-CANARY-END
