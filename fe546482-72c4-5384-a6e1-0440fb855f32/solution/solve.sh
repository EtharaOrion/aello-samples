#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S6. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-a0e42c0065f28a3ae7cd80750e6deb25
#   slot1: AELLO-CANARY-78fa5ff042735b8899aa461525adfeb5
#   slot2: AELLO-CANARY-b9f439e31b6c2b3bf2a0eca0f82e6642
#   slot3: AELLO-CANARY-398a99cbdda81fcdf9fc88a19f7daa24
# AELLO-CANARY-END
