#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S8. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-499b6720890d6f693baa338ed2604792
#   slot1: AELLO-CANARY-28b9e5a75be70673ff77e11abc1a8559
#   slot2: AELLO-CANARY-fe83372c9521a691808f135217540f44
#   slot3: AELLO-CANARY-149bdddb0ee8828149dfabe093875559
# AELLO-CANARY-END
