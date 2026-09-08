#!/usr/bin/env bash
# Oracle solve path for C4-S2. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-961e9ee1d902e446af77186075176d82
#   slot1: AELLO-CANARY-1d39c05019a2bd79ede04131af92c686
#   slot2: AELLO-CANARY-d1ee3e1b4e6d5d751fe497411cb0eeac
#   slot3: AELLO-CANARY-9cc5c5f0ae72eed9c3f0774ba1de8eb5
# AELLO-CANARY-END
