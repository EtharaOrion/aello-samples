#!/usr/bin/env bash
# Verifier entry point for C7-S2. Runs in the separate verifier environment, no network, no GPU.
set -euo pipefail
python "$(dirname "$0")/verifier.py"
