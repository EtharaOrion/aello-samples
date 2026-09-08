#!/usr/bin/env bash
# Oracle solve path for AELLO-C6-S2. Runs only with AELLO_ORACLE=1 in the solution environment.
set -euo pipefail
python "$(dirname "$0")/reference.py"

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-6fba8de0ad64e1b823dba2be01e8ffac
#   slot1: AELLO-CANARY-f316b670f6f37e03b147995e2ec96c47
#   slot2: AELLO-CANARY-0fe6e7abe0d04af8a82cc4703cc2353d
#   slot3: AELLO-CANARY-a056eaed8430320e4f458335f5f1f722
# AELLO-CANARY-END
