#!/usr/bin/env bash
# Oracle solve path for AELLO-C8-S11. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-741a9ceccad662e7022448e7f797fc44
#   slot1: AELLO-CANARY-521001e5368bb237f8f227a575ae6e33
#   slot2: AELLO-CANARY-b2e27e5644d2d284c7954bf9baaca2d1
#   slot3: AELLO-CANARY-492407b078d8e0821be18962ad1f325f
# AELLO-CANARY-END
