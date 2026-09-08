#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S4. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-d6bc271759959524cc4b33281711e915
#   slot1: AELLO-CANARY-33083685cac396e42d8570cc3369d271
#   slot2: AELLO-CANARY-67da12c6a884a149696f4dc0fa2caeca
#   slot3: AELLO-CANARY-ae6fe18fd95ce501f5e15fe5ce2bf9d6
# AELLO-CANARY-END
