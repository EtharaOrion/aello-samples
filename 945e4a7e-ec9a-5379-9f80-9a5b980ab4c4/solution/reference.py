"""Reference solve for C2-S1.

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
#   slot0: AELLO-CANARY-d5d4aea4b1a3b0f4d0689df46bcec558
#   slot1: AELLO-CANARY-ef614c8e00a927d1a673d6eabeca605a
#   slot2: AELLO-CANARY-9ba15fdbbd64d92cd77e283cb768bebe
#   slot3: AELLO-CANARY-536c3121dcffa7610e081294f1107812
# AELLO-CANARY-END
