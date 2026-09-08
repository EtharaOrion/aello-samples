#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S2. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-f52bbbe19703ede56d5674eb4c5d7dc3
#   slot1: AELLO-CANARY-23a009b98bd344917e87ef548b584156
#   slot2: AELLO-CANARY-3c8086a68a27b36b1f2ae16874efe9da
#   slot3: AELLO-CANARY-90b0dc8de7fd77d4eb5b7482318e3440
# AELLO-CANARY-END
