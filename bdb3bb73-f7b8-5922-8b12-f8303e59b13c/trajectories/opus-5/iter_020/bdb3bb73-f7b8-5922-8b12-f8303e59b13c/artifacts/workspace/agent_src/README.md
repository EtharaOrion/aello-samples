# CIFAR-10 → 94% from scratch, 1×H100

Graded artifacts: `periodic_checkpoints/ckpt_epoch_0001.pt … 0020.pt` (20 contiguous
TorchScript modules, one per epoch) and `checkpoint.pt` (hard link to epoch 20).

**First checkpoint ≥ 94% on the test split: epoch 19 at 94.20%** (94.60% at epoch 20),
reconstructed `t*` = 16.12 s against a 28 s reference / 70 s baseline. Trained from scratch
on the CIFAR-10 train split only — no pretrained weights, no external data.

The exported module takes a float32 NCHW batch that is **already standardized** with
`ToTensor() → Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))`, exactly the
evaluator's convention, and applies 6-way TTA internally.

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | protocol, objective and candidate list, written before the sweep |
| `RESULTS.md` | outcomes, ablation, deviations from the protocol, and what was skipped |
| `train.py` | model + training loop; writes one scripted checkpoint per epoch |
| `verify_ckpt.py` | re-evaluates checkpoints through the evaluator's exact pipeline |
| `eval_repeat.py` | repeatability of that evaluation across fresh processes |
| `prep_data.py` | one-off cache of the PNG folders to uint8 `.npy` |
| `sweep.jsonl`, `runs.jsonl` | every run executed, winners and losers alike |

```bash
python prep_data.py
python train.py --epochs 20 --lr 6 --seed 0 --out /workspace/periodic_checkpoints
python verify_ckpt.py --ckpt-dir /workspace/periodic_checkpoints --split test
```
