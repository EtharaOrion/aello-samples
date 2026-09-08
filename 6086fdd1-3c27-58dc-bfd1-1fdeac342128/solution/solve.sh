#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S7. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-34477cdc8d45a4268664ce6af225566f
#   slot1: AELLO-CANARY-be3116e30ccdfbd449346d06d270ee88
#   slot2: AELLO-CANARY-aa4237d8ef3a4e17d01dff3c00bf57f1
#   slot3: AELLO-CANARY-5994bef66a56e9cfd85895e8bafffa3f
# AELLO-CANARY-END
