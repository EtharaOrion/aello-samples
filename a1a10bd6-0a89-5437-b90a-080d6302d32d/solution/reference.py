"""Reference solve for AELLO-C8-S3.

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
#   slot0: AELLO-CANARY-ad0326b653051bf3fff9b3d0f400c264
#   slot1: AELLO-CANARY-16319a20be1c38a127ad5533a33025a5
#   slot2: AELLO-CANARY-7b2bbeeef9a8c1bf504167a51fdb9fb3
#   slot3: AELLO-CANARY-a13aa21e8dec1742842ba8b70712bb43
# AELLO-CANARY-END
