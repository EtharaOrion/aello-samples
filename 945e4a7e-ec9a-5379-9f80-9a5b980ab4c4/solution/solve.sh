#!/usr/bin/env bash
# Oracle solve path for C2-S1. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-d5d4aea4b1a3b0f4d0689df46bcec558
#   slot1: AELLO-CANARY-ef614c8e00a927d1a673d6eabeca605a
#   slot2: AELLO-CANARY-9ba15fdbbd64d92cd77e283cb768bebe
#   slot3: AELLO-CANARY-536c3121dcffa7610e081294f1107812
# AELLO-CANARY-END
