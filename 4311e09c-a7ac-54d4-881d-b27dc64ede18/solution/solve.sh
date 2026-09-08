#!/usr/bin/env bash
# Produce the reference submission: one verdict in {A, B, ABSTAIN} for every item of both
# private graded shards. Runs under Harbor's oracle (solution at /solution, environment at
# /workspace/environment) and on the authoring host (bundle-relative). FAILS LOUDLY: no
# swallowed import and no empty submission.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/submission}"; mkdir -p "$OUT"

if [ -d /workspace/environment ]; then ENV_DIR=/workspace/environment; else ENV_DIR="$HERE/../environment"; fi
if [ -d /workspace/environment/qwen2.5-1.5b-instruct ]; then BACKBONE=/workspace/environment/qwen2.5-1.5b-instruct
elif [ -d "$ENV_DIR/qwen2.5-1.5b-instruct" ]; then BACKBONE="$ENV_DIR/qwen2.5-1.5b-instruct"
else BACKBONE=/aello/corpora/qwen2.5-1.5b-instruct; fi

python3 "$HERE/reference.py" \
    --train   "$ENV_DIR/data/train.jsonl" \
    --shards  "$ENV_DIR/data/p1_items.jsonl" "$ENV_DIR/data/p2_items.jsonl" \
    --backbone "$BACKBONE" \
    --out     "$OUT/verdicts.json"

[ -f "$OUT/verdicts.json" ] || { echo "solve.sh: reference wrote no verdicts" >&2; exit 1; }
echo "solve.sh: wrote $OUT/verdicts.json"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-5a2a398da0b1caaab4680dfe60f25158
#   slot1: AELLO-CANARY-e63e4a2ffec892f6e66052747d2ca73b
#   slot2: AELLO-CANARY-d5486f12956ce7d142627e039e421dc1
#   slot3: AELLO-CANARY-28761ed455783590174668c81e473ff2
# AELLO-CANARY-END
