#!/usr/bin/env bash
# Verifier entry point for AELLO-C8-S10. Runs in the separate verifier environment, no network, no GPU.
set -euo pipefail
python "$(dirname "$0")/verifier.py"
