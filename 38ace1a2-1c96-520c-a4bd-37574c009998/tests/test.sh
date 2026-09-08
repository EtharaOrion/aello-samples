#!/usr/bin/env bash
# Grade the submission. The graded split lives here and never inside the environment.
set -euo pipefail
python3 "$(dirname "$0")/verifier.py" "${1:-/submission}"
