#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S12. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-67655e24f59c37a970327975520caf38
#   slot1: AELLO-CANARY-f3e8eca232d018e338784243dcffc30b
#   slot2: AELLO-CANARY-ef51c9a8a85c57f764e8ab7422b035bf
#   slot3: AELLO-CANARY-52d049979efe780c34f5f68358752138
# AELLO-CANARY-END
