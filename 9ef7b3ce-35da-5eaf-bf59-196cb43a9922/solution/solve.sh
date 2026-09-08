#!/usr/bin/env bash
# Oracle solve path for C3-S3. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-f42c199cd5a0d3d34bcae0194d0a49cb
#   slot1: AELLO-CANARY-e5dda9fc7062a2acc5607acd68841fc0
#   slot2: AELLO-CANARY-c334cff533ec729899ef99136cfa6e30
#   slot3: AELLO-CANARY-2f96edbdd8a3d045c2102804507a2679
# AELLO-CANARY-END
