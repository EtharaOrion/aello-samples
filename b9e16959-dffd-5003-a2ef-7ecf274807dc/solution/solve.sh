#!/usr/bin/env bash
# Oracle solve path for C1-S2. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-db8b1b3ed618b1b560f45ca25e91ec57
#   slot1: AELLO-CANARY-45914acb9cc37312a164ee7b509ad070
#   slot2: AELLO-CANARY-8ec4a1b6c989d9006c88fd0fa7ddf649
#   slot3: AELLO-CANARY-32f7fc7c78381adbde8d73756d0bae61
# AELLO-CANARY-END
