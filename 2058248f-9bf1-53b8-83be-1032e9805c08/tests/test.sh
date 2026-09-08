#!/usr/bin/env bash
# Verifier entry point for A5-02. Runs in the separate verifier environment, no network, no GPU.
set -euo pipefail
python "$(dirname "$0")/verifier.py"
