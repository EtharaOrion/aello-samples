#!/usr/bin/env bash
# Oracle solve path for C3-S1. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-f6e63dc5578069a7c003d86e035cf00e
#   slot1: AELLO-CANARY-932aca95af39a2245aad33557f82df8b
#   slot2: AELLO-CANARY-4312fead687026fb0eadc784a8254489
#   slot3: AELLO-CANARY-fae240e6e015c5a73d05714e4672b2dd
# AELLO-CANARY-END
