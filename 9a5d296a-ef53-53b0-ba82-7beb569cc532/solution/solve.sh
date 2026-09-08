#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S10. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-f4779ea1f0db2ce35b6f9fbf1f352ced
#   slot1: AELLO-CANARY-cb5afcea4331ba82432379b38ea41063
#   slot2: AELLO-CANARY-23bcbe3d45419a0c71ce6b35c3cf53b5
#   slot3: AELLO-CANARY-6b03daab822b0d4c49342646141a7551
# AELLO-CANARY-END
