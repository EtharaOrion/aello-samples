#!/bin/bash
# Run a list of train.py argument strings (one per line on stdin) with P-way
# GPU sharing. Every run scores against the held-out train slice only.
P=${P:-3}
LOG=${LOG:-/workspace/logs/sweep.jsonl}
xargs -P "$P" -I{} bash -c "python /workspace/train.py {} --eval holdout --save 0 --log $LOG >/dev/null 2>&1"
