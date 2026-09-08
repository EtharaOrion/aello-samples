#!/usr/bin/env bash
# Oracle solve path for A5-02. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-7b162fe9b60213a21fd30c9da15a2253
#   slot1: AELLO-CANARY-8d9e78a1f4db48d5b1fa3795637569cc
#   slot2: AELLO-CANARY-43110090c35dee48e46c8cc99dc5830d
#   slot3: AELLO-CANARY-a9789180a5f53d2358dd783cc17a74a4
# AELLO-CANARY-END
