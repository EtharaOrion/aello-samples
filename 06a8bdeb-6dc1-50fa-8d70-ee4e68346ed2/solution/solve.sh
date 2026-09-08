#!/usr/bin/env bash
# Oracle solve path for C2-S2. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-323a294673512fc8b207f58340b7b58f
#   slot1: AELLO-CANARY-3f898a18d5e6f83c266c22f0b561dd0e
#   slot2: AELLO-CANARY-1ac3fab5ecd8397d1398bde2c191f6c6
#   slot3: AELLO-CANARY-e00cbd212d4e05bf3576f0b24bc52caa
# AELLO-CANARY-END
