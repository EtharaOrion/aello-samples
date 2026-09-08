#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S9. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-3338d48cf6aaaab9fed70d3999c4eef3
#   slot1: AELLO-CANARY-59f56b16c9be9274bfe9c2f55513d4ee
#   slot2: AELLO-CANARY-3f73f40ec5070d2d02b1d437e606d45b
#   slot3: AELLO-CANARY-d1e3166e80845b74480c81931491d895
# AELLO-CANARY-END
