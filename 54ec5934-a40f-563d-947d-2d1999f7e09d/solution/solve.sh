#!/usr/bin/env bash
# Oracle solve path for C7-S2. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-061f6f6b7b2a54a9210b0e6700581284
#   slot1: AELLO-CANARY-cd4ebec219f03a949b2629082c125744
#   slot2: AELLO-CANARY-ae1bbcf0a8a058eee6be875bed4c1963
#   slot3: AELLO-CANARY-e0fbd2f0bc1ee664455d6ba374c691f1
# AELLO-CANARY-END
