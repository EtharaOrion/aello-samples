#!/usr/bin/env bash
# Oracle solve path for C3-S2. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-6b43a83e0d9ceb85305f340ced6b5f99
#   slot1: AELLO-CANARY-9383ea62e6f2331240c3312fe7e79f81
#   slot2: AELLO-CANARY-6541eb481d5bfc618e97be0c9968f0a8
#   slot3: AELLO-CANARY-825ff8b67f70266cb9b5c33c45596132
# AELLO-CANARY-END
