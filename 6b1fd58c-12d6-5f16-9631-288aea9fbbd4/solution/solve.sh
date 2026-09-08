#!/usr/bin/env bash
# Produce the reference submission: one pairwise verdict for every held-out and sibling item.
# Runs under Harbor's oracle (solution at /solution, environment at /workspace/environment) and
# on the authoring host (bundle-relative). FAILS LOUDLY: no swallowed import or empty submission.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/submission}"; mkdir -p "$OUT"

if [ -d /workspace/environment ]; then ENV_DIR=/workspace/environment; else ENV_DIR="$HERE/../environment"; fi
if [ -d /workspace/environment/qwen2.5-1.5b-instruct ]; then BACKBONE=/workspace/environment/qwen2.5-1.5b-instruct
elif [ -d "$ENV_DIR/qwen2.5-1.5b-instruct" ]; then BACKBONE="$ENV_DIR/qwen2.5-1.5b-instruct"
else BACKBONE=/aello/corpora/qwen2.5-1.5b-instruct; fi

TRAIN="$ENV_DIR/data/train.jsonl"
# The graded items are the held-out set plus the sibling set, both shipped label-free.
ITEMS="$OUT/all_items.jsonl"
cat "$ENV_DIR/data/heldout_items.jsonl" "$ENV_DIR/data/sibling_items.jsonl" > "$ITEMS"

python3 "$HERE/reference.py" --train "$TRAIN" --heldout "$ITEMS" \
    --backbone "$BACKBONE" --out "$OUT/verdicts.json"

[ -f "$OUT/verdicts.json" ] || { echo "solve.sh: reference wrote no verdicts" >&2; exit 1; }
echo "solve.sh: wrote $OUT/verdicts.json"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-57650fcc9e7844e96102aff3d153140e
#   slot1: AELLO-CANARY-24df3ccf3ea91ccb94901fb7d0030017
#   slot2: AELLO-CANARY-d51868227907c063ec512b655ddabdf7
#   slot3: AELLO-CANARY-56d7ecf4d68f6f277b17869ce9e8bdc6
# AELLO-CANARY-END
