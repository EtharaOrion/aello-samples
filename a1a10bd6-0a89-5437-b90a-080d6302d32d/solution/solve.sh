#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S3. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-ad0326b653051bf3fff9b3d0f400c264
#   slot1: AELLO-CANARY-16319a20be1c38a127ad5533a33025a5
#   slot2: AELLO-CANARY-7b2bbeeef9a8c1bf504167a51fdb9fb3
#   slot3: AELLO-CANARY-a13aa21e8dec1742842ba8b70712bb43
# AELLO-CANARY-END
