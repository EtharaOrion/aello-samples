#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S5. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-64e40139a41090860fd302035370d48d
#   slot1: AELLO-CANARY-5bd0be3a160298993782b68cfcf9f8a5
#   slot2: AELLO-CANARY-53a8ccda1f8385db6db18a47947ded69
#   slot3: AELLO-CANARY-6e98d3431d8759bf062af90956533987
# AELLO-CANARY-END
