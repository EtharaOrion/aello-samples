#!/usr/bin/env bash
# Oracle solve path for C1-S1. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-609155f956d380f15b3b4d72406398e3
#   slot1: AELLO-CANARY-340a728dcc236349192fdbce4d39ad6f
#   slot2: AELLO-CANARY-62c169bfe63371d0f6ea5cd7ec88f0a9
#   slot3: AELLO-CANARY-dcde825fef9015fb05e906169c5c3183
# AELLO-CANARY-END
