# AELLO-C8-S10 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

Maximise plain accuracy on the private 2019 nodes of OGBN-arxiv, inside a fixed neighbourhood-sampling budget per epoch. The graded artifact is a one-dimensional integer label array with one row per graded node, written to /submission/preds_c8s10.npy in graded_idx order.

## The single most important insight

This corpus is imbalanced by a factor of 942 and it is graded on plain accuracy. Those two facts pull in opposite directions, and almost every instinct a competent practitioner brings to an imbalanced classification problem is calibrated for the axis this slot does not grade. Inverse-frequency class weighting, the single most standard response to the imbalance, is measured here at minus 0.10712 accuracy and plus 0.09928 balanced accuracy on the same eight seeds. The whole task is one question asked twice: which axis are you actually being scored on, and did your pipeline agree with your answer?

## The ideal solve, step by step

1. **Read the split as a calendar, not as a shuffle** train_idx is every paper up to 2017, val_idx is 2018, graded_idx is 2019, and the three sets are disjoint and cover every node. The label prior moves 0.346777 in total variation between the training years and the graded year, and the single most common class changes from 28 to 24 across that boundary. A constant predictor of the training majority scores 0.05862 on the graded fold while a constant predictor of the graded majority would score 0.22097. Nothing calibrated to the training prior arrives at 2019 intact.

2. **Bring the neighbourhood in, because it is the dominant lever** graph.npz ships neighbour_mean, the mean of the feature matrix over each paper's one-hop neighbours. Using it is worth 0.12311 accuracy over the same model on raw features alone. It is the largest single movement available on the delivered bytes and the contract calls it the dominant lever for this slot. The environment has already spent the fan-out half of the sampling budget by shipping the aggregate instead of the edge list.

3. **Keep the paper's own embedding beside the aggregate** Replacing the feature row with the neighbour mean rather than concatenating the two costs 0.01511 accuracy. It is the smallest of the four measured levers and it is the delivered form of the residual connection the contract records as L3, since there is no second layer for a skip to skip over.

4. **Standardise on training-year moments only** Centring and scaling each of the 256 columns is worth 0.04950 accuracy. Take the moments from train_idx alone. Taking them over all 169343 rows folds the graded year into the fitting statistics, which is a small transductive leak into exactly the fold this slot defines as a future.

5. **Decide the metric axis on purpose, and then check that your code agrees** This is the step that carries the task. Class reweighting costs 0.10712 accuracy, which is 87 percent of what the whole graph lever buys, and it buys 0.09928 balanced accuracy in exchange. The gap is visible on the public 2018 fold at 0.10469 against a per-seed spread of 0.00337, so it is findable without touching anything private. Over the 32-point lever ladder the two axes correlate at 0.9344 and still disagree about which configuration is best. A high correlation between axes is not agreement between them.

## Traps that catch agents that are not thinking carefully

- Reweighting the loss by inverse class frequency because the corpus is 942 to 1 imbalanced, on a slot graded by plain accuracy, at a measured cost of 0.10712.
- Calibrating a prior, a threshold or a reweighting on the training years and carrying it across a boundary the label distribution moves 0.346777 in total variation across.
- Taking standardisation moments, or any other fitted statistic, over all 169343 rows and folding the graded year into the fit.
- Stacking depth until representations over-smooth and accuracy falls below the feature-only control, which measures 0.52325.
- Spending the whole fan-out budget on the first hop.
- Transferring an OGB leaderboard recipe for this exact split, when the leaderboard assumes an edge list this bundle does not ship and an epoch budget far above the authored one.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path, with the checker identifier each step satisfies:

1. `graded_accuracy` (compiled, weight 10, knockout) -- The graded metric clears the published bar and is mapped through the reward ramp.
2. `graded_labels_isolated` (compiled, weight 5, knockout) -- The agent holds no copy of the private graded labels anywhere in the files it wrote.
3. `no_graded_writes` (compiled, weight 5, knockout) -- The graded run writes nothing into the checker tree or the private graded fold.
4. `predictions_not_degenerate` (compiled, weight 3) -- The agent produces a real classifier rather than a constant or near-constant output.
5. `predictions_present` (compiled, weight 5, knockout) -- The agent writes a prediction file for the graded nodes at the declared submission path.
6. `predictions_wellformed` (compiled, weight 5, knockout) -- The prediction array has exactly one integer label per graded node, in graded_idx order, inside the label domain.
7. `sampling_budget_declared` (compiled, weight 3) -- The agent keeps the graded neighbourhood-sampling budget as published.

Rejected routes, each bound to the control that measures it wrong:

- reweight the loss by inverse class frequency, because the corpus is 942 to 1 imbalanced -- control `REF-DEFECT`, measured 0.53924 against reference 0.64636
- ignore the graph and classify each paper from its own embedding -- control `CTL-MLP`, measured 0.52325 against reference 0.64636
- transfer a recipe tuned where one epoch is one update -- control `CTL-STALE`, measured 0.57836 against reference 0.64636
- predict the training years' most common class for every graded row -- control `CTL-PRIOR`, measured 0.05862 against reference 0.64636
- ship the starter unchanged -- control `CTL-NOOP`, measured 0.32703 against reference 0.64636

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-f4779ea1f0db2ce35b6f9fbf1f352ced
  slot1: AELLO-CANARY-cb5afcea4331ba82432379b38ea41063
  slot2: AELLO-CANARY-23bcbe3d45419a0c71ce6b35c3cf53b5
  slot3: AELLO-CANARY-6b03daab822b0d4c49342646141a7551
-->
