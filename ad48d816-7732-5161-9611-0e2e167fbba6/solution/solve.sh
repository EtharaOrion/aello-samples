#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S13. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-61cc355fac243dd3dad4afc3b9a84a97
#   slot1: AELLO-CANARY-a107ee07d664b4b1f66d41dffb3ee882
#   slot2: AELLO-CANARY-306614f257370612315448f37e23f919
#   slot3: AELLO-CANARY-21d137251a71485a1d61069e01973a95
# AELLO-CANARY-END
