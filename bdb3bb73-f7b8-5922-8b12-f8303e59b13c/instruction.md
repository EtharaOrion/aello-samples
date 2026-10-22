# CIFAR-10 to 94% top-1 on 1xH100

Train a model from scratch (no pretrained weights, no external data) on the CIFAR-10 train split, and reach 94% top-1 accuracy on the CIFAR-10 test split as fast as possible on a single H100 within a 1800-second wall-clock budget.

## What you must deliver

Write PERIODIC CHECKPOINTS at `/workspace/periodic_checkpoints/ckpt_epoch_{NNNN}.pt` where NNNN is the zero-padded epoch number, ONE per epoch (or every 1 epochs). EACH checkpoint MUST be a LOADABLE, scripted model (`torch.jit.script(model).save(path)`), NOT a bare state_dict. Also write the final `/workspace/checkpoint.pt` as a copy or hardlink of the last periodic checkpoint.

You MAY write `/workspace/agent_reported_metrics.json` with `{"accuracy": <float>, "wall_clock_sec": <float>}` for informational purposes. The verifier INDEPENDENTLY re-evaluates each checkpoint on a **held-out eval set that is not present in your container**; agent-reported metrics are advisory only and never trusted for scoring.

Your container has ONLY the CIFAR-10 **train** split at `/workspace/data/cifar10/train`. The eval/test split is deliberately withheld (the verifier holds it), so hold out a **validation** set from train to gauge progress; do not attempt to obtain the eval split.

## How you are scored

The verifier composes:

    score = GATE * S_outcome * RubricFactor      score in [0,1]

- **GATE** = 0 if ANY knockout red-line fails (pretrained weights loaded, test-set leakage, stale checkpoint, spoofed accuracy, dishonest timer). GATE = 1 otherwise.
- **S_outcome** = normalized time-to-accuracy from the FIRST periodic checkpoint whose independently re-evaluated accuracy >= 0.94. `S_outcome = clamp((t_baseline_sec - t_star_sec) / (t_baseline_sec - t_reference_sec), 0.0, 1.0)`. If no checkpoint crosses within the budget, S_outcome = 0.
- **RubricFactor** = floor + (1 - floor) * (SUM w_j * s_j / SUM w_j) over scored process rubrics, floor = 0.5. Weights w_j in {1,3,5}. Rubrics can only DISCOUNT the measured outcome.

Absolute anchors t_baseline_sec and t_reference_sec are calibrated on the target GPU.

## Constraints

- MUST NOT load or initialize weights from any pretrained checkpoint or public weight zoo.
- MUST NOT feed test-split inputs to the model during training.
- MUST train from scratch; no external datasets beyond CIFAR-10 train split.
- MUST use only the pinned network allowlist for pip/apt installs.
- Reference: MLPerf Training open-division rules apply (arxiv:1910.01500).
