#!/usr/bin/env bash
# Oracle solve path for C7-S1. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-028d61af2552c5b85c0f40c26e71a403
#   slot1: AELLO-CANARY-776e291fc5c990ea7fe9a467594be7a5
#   slot2: AELLO-CANARY-1d7e725946e835850e2abc6b662e51cc
#   slot3: AELLO-CANARY-1732723eb731ff57050b5dc23a6e1bb7
# AELLO-CANARY-END
