#!/usr/bin/env bash
# Oracle solve path for A5-03. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-cd2709372b26a344b6894f9329e0c198
#   slot1: AELLO-CANARY-c4325951b4235a6931e7ab18bcfc6ec8
#   slot2: AELLO-CANARY-b21b8b15e54a1fe3887f52fb268b4ee6
#   slot3: AELLO-CANARY-29a2a57471ec09178a9ab0d0f4fff0a4
# AELLO-CANARY-END
