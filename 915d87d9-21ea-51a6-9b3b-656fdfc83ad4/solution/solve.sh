#!/usr/bin/env bash
# Oracle solve path for C4-S3. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-984f0bb9464dd324952dbd7d9fad7601
#   slot1: AELLO-CANARY-6bd7f7dd136f19f5856e9f97a07752ba
#   slot2: AELLO-CANARY-6a1fe153309429d4a4f5408c6b2efb52
#   slot3: AELLO-CANARY-ad3cfdc08b790d8bc1f5c6b3a110df31
# AELLO-CANARY-END
