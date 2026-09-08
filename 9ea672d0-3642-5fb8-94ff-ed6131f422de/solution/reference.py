"""Reference solve for C4-S2.

NOT A MEASUREMENT. This produces a well-formed artifact by the shortest correct route so the
delivery path and the checker set can be exercised end to end before any constant exists. The
reference DISTRIBUTION that anchors the reward ramp is measured at Phase 1 on the grading host,
not here.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "environment", "starter"))
from starter import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-961e9ee1d902e446af77186075176d82
#   slot1: AELLO-CANARY-1d39c05019a2bd79ede04131af92c686
#   slot2: AELLO-CANARY-d1ee3e1b4e6d5d751fe497411cb0eeac
#   slot3: AELLO-CANARY-9cc5c5f0ae72eed9c3f0774ba1de8eb5
# AELLO-CANARY-END
