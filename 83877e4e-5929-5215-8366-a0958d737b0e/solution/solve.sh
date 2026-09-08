#!/usr/bin/env bash
# Oracle solve path for A5-01. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-ce91d786a4385a70bb0a317142d02254
#   slot1: AELLO-CANARY-ba6d0f8c04db42677376dc3be6a94452
#   slot2: AELLO-CANARY-d7df51fe959ef0a4fa36da6b784c8c65
#   slot3: AELLO-CANARY-072efa8e79e13f69b7fb43c25232a6d6
# AELLO-CANARY-END
