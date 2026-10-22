#!/usr/bin/env bash
# Harbor verifier entry point. P0.3 fail-closed: if the verifier exits non-zero and wrote no result, emit score 0.0 with a machine-readable reason (reward.json is the harness result sink).
set -o pipefail
cd "$(dirname "$0")"
python verifier.py
rc=$?
if [ $rc -ne 0 ] && [ ! -f /logs/verifier/score.json ]; then
    mkdir -p /logs/verifier
    echo '{"score": 0.0}' > /logs/verifier/score.json
    echo '{"score": 0.0, "zero_score_reason": "verifier_exited_non_zero_and_wrote_no_score"}' > /logs/verifier/score_detail.json
fi
exit $rc
