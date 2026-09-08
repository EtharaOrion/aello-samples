#!/usr/bin/env bash
# Produce the reference submission: predictions for the graded images.
#
# Runs under Harbor's oracle (solution mounted at /solution, environment at
# /workspace/environment, no /tests present) AND on the authoring host (bundle-relative).
# The graded images ship label-free inside the environment, so the oracle needs nothing
# from tests/. FAILS LOUDLY: the earlier revision swallowed a python import error inside a
# command substitution and exited 0 with an empty submission.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/submission}"; mkdir -p "$OUT"

if [ -d /workspace/environment ]; then ENV_DIR=/workspace/environment; else ENV_DIR="$HERE/../environment"; fi
GRADED="$ENV_DIR/data/graded_images.npz"
[ -f "$GRADED" ] || { echo "solve.sh: graded images not found at $GRADED" >&2; exit 1; }

# The instruction makes a top-level BUDGET_STEPS declaration mandatory in the code
# that drives the final training run, and the reference is graded by the same checkers
# as any agent. So the oracle does what a compliant agent does: it writes its driver
# into the workspace, declaring the budget, and that driver performs the run. Only this
# generated driver enters the workspace; the private reference stays under /solution,
# because reference.py is a canary carrier and must never cross into collected trees.
if [ -d /workspace ] && [ -w /workspace ]; then
  cat > /workspace/train.py <<DRIVER
"""Reference driver. Declares the graded budget and drives the run that produces the
submitted predictions, as instruction.md requires of whatever file drives training."""
BUDGET_STEPS = 8000
import subprocess
import sys
subprocess.run(
    [sys.executable, "$HERE/reference.py", "--instance", "i20k", "--seed", "0",
     "--graded", "$GRADED", "--out", "$OUT/preds_i20k.npy"],
    check=True)
DRIVER
  python3 /workspace/train.py
else
  python3 "$HERE/reference.py" --instance i20k --seed 0 --graded "$GRADED" --out "$OUT/preds_i20k.npy"
fi

[ -f "$OUT/preds_i20k.npy" ] || { echo "solve.sh: reference wrote no predictions" >&2; exit 1; }
echo "solve.sh: wrote $OUT/preds_i20k.npy"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-bea18ed252d9784fdf61fe87fa7f4998
#   slot1: AELLO-CANARY-c1d6ab4e4a9586741a987c0f788e87f9
#   slot2: AELLO-CANARY-55f6e11e7b352cbdd5a41275b962be86
#   slot3: AELLO-CANARY-5c9e29f9f3d650b9fee3b9cb4577a4d5
# AELLO-CANARY-END
