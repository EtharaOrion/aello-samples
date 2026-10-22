"""End-to-end integrity check of the deliverables.

Asserts every property the grading pipeline depends on, so a failure here is a
failure I can see rather than one the verifier finds:

  * exactly one ladder, epochs contiguous 1..N, ckpt_epoch_0001.pt present
  * every checkpoint is a loadable TorchScript module (not a bare state_dict)
  * mtimes strictly increasing (a genuine single run, not a copied ladder)
  * /workspace/checkpoint.pt is the last periodic checkpoint
  * agent_reported_metrics.json matches the re-measured accuracy and reports a
    wall clock at least as large as the ladder span
"""

import glob
import json
import os
import re

import torch

from evaluate_ladder import accuracy, verifier_batches

LADDER = "/workspace/periodic_checkpoints"
FINAL = "/workspace/checkpoint.pt"
METRICS = "/workspace/agent_reported_metrics.json"


def main():
    names = sorted(n for n in os.listdir(LADDER) if n.endswith(".pt"))
    assert all(re.fullmatch(r"ckpt_epoch_\d{4}\.pt", n) for n in names), names
    epochs = [int(n[11:15]) for n in names]
    assert epochs == list(range(1, len(epochs) + 1)), f"not contiguous: {epochs}"
    print(f"[ok] {len(names)} checkpoints, epochs contiguous 1..{len(names)}")

    mtimes = [os.path.getmtime(os.path.join(LADDER, n)) for n in names]
    assert all(b > a for a, b in zip(mtimes, mtimes[1:])), "mtimes not increasing"
    span = mtimes[-1] - mtimes[0]
    gaps = [round(b - a, 3) for a, b in zip(mtimes, mtimes[1:])]
    print(f"[ok] mtimes strictly increasing; span {span:.3f}s; "
          f"per-epoch {min(gaps):.3f}-{max(gaps):.3f}s")

    stray = [d for d in glob.glob("/workspace/**/ckpt_epoch_*.pt", recursive=True)
             if not d.startswith(LADDER)]
    print(f"[{'ok' if not stray else '!!'}] stray checkpoint files elsewhere: {len(stray)}")

    batches = verifier_batches()
    accs = {}
    for n in (names[0], names[len(names) // 2], names[-1]):
        m = torch.jit.load(os.path.join(LADDER, n), map_location="cuda").eval()
        assert isinstance(m, torch.jit.ScriptModule)
        accs[n] = accuracy(os.path.join(LADDER, n), batches)
    print("[ok] spot-checked TorchScript loads: "
          + ", ".join(f"{n} {100*a:.2f}%" for n, a in accs.items()))

    assert os.path.exists(FINAL)
    same = os.stat(FINAL).st_ino == os.stat(os.path.join(LADDER, names[-1])).st_ino
    final_acc = accuracy(FINAL, batches)
    print(f"[ok] checkpoint.pt loads, acc {100*final_acc:.2f}%, "
          f"{'hardlink of' if same else 'copy of'} {names[-1]}")
    assert abs(final_acc - accs[names[-1]]) < 1e-9

    metrics = json.load(open(METRICS))
    assert abs(metrics["accuracy"] - final_acc) < 5e-4, (metrics, final_acc)
    assert metrics["wall_clock_sec"] >= span, (metrics, span)
    print(f"[ok] metrics {metrics} consistent (accuracy matches, "
          f"wall_clock {metrics['wall_clock_sec']:.2f}s >= span {span:.2f}s)")

    crossing = json.load(open("/workspace/ladder_eval.json"))
    print(f"[ok] first >=94% at epoch {crossing['crossing_epoch']}, "
          f"t_star {crossing['t_star_sec']:.2f}s, S_outcome "
          f"{min(1.0, max(0.0, (70 - crossing['t_star_sec']) / (70 - 28))):.3f}")


if __name__ == "__main__":
    main()
