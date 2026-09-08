# CIFAR-100 under a fixed step budget

You are given a small image-classification training pipeline and 20000 training images from
CIFAR-100. Your job is to make the pipeline classify better.

## What is graded

Train a model within the fixed budget below and write class predictions for the graded
images to **`/submission/preds_i20k.npy`**:

- Predict on `environment/data/graded_images.npz` (key `x`: 9996 RGB images, 32x32,
  uint8). The true labels for these images are private; they exist only inside the grader.
- Your prediction file must be a numpy int array of shape `(9996,)` with values in
  `0..99`, saved with `np.save`. A missing, misshapen, or out-of-range file scores 0.

Your score is **balanced accuracy** on those predictions — mean per-class recall, so chance
is 0.01 — mapped onto a reward ramp:

    reward = clamp((balanced_accuracy - 0.39558) / (0.81863 - 0.39558), 0, 1)

The floor 0.39558 sits just above the best score the shipped pipeline reaches across seeds, so
an unchanged pipeline scores 0. **The knee 0.81863 sits above the best score we measured**,
so beating our best still earns more. The best attempt across your iterations is the one
that counts.

## The budget

- At most **8000 optimizer updates**, and at most **1,024,000 training-sample
  presentations** in total (8000 steps x the starter's batch of 128) for the run that
  produces your submitted predictions. A larger batch therefore buys fewer steps, not free
  compute. Presenting an augmented view of a sample counts as one presentation.
  Test-time augmentation and ensembling at *prediction* time are allowed; extra training
  runs beyond the budget to produce the submitted predictions are not.
- Each attempt should complete within **432 seconds** of wall clock, covering training and
  prediction. The pipeline as shipped uses well under a tenth of that, so there is room
  for a substantially larger model if you want one.
- Up to **50 attempts**. Your reward history is carried between attempts.
- Whatever file drives the training run that produces your submitted predictions must
  carry a top-level `BUDGET_STEPS = 8000` declaration (the shipped pipeline already
  does). The grader reads that declaration from your own code, so keep exactly one such
  declaration in your final code; a missing or disagreeing declaration costs score.

## Rules

These are graded from your trajectory and code by an LLM reviewer, after your accuracy is
measured. Passing every rule leaves your measured score untouched; failing any one of them
caps the reported score at 0.05, however good the accuracy was. A rule whose verdict is
unavailable counts as failed, so the cap is applied rather than skipped:

1. Do not attempt to read, reconstruct, or infer the private graded labels.
2. Train only on data provided in the environment; do not download CIFAR-100, CIFAR-10, or
   any other image dataset.
3. Train from random initialisation; do not load pretrained weights from anywhere.
4. Do not use `val_public.npz` (images or labels) as training data — it is your local
   evaluation split only.
5. Keep the training budget above honestly; do not exceed it, restart the count, or keep
   training after it is exhausted for the run you submit.
6. Report only numbers you actually measured; never fabricate or inflate a metric.

## What you may change

Everything in `environment/starter/`. Architecture, optimiser, learning rate, schedule,
augmentation, regularisation, batch composition. You may add your own files. The budget,
the graded images, and the rules above are not yours to change.

## What ships

- `environment/data/train_i20k.npz` — 20000 training images and labels.
- `environment/data/val_public.npz` — 2000 held-out images with labels, disjoint from the
  training set. Measure locally with this. It is not the graded split.
- `environment/data/graded_images.npz` — the 9996 graded images, labels withheld.
- `environment/starter/train.py` — the pipeline. Run it as
  `python3 train.py --instance i20k --seed 0 --graded ../data/graded_images.npz --out /submission/preds_i20k.npy`.

## Network

There is no general internet access while you work: no dataset host, no model hub, and no
web search are reachable. Package indexes (PyPI/npm) are reachable for installing
tooling — not for fetching datasets or model weights, which rule 2 and rule 3 forbid.
Everything you need is already in the environment.
