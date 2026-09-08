"""Reference solve for C3-S3.

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
#   slot0: AELLO-CANARY-f42c199cd5a0d3d34bcae0194d0a49cb
#   slot1: AELLO-CANARY-e5dda9fc7062a2acc5607acd68841fc0
#   slot2: AELLO-CANARY-c334cff533ec729899ef99136cfa6e30
#   slot3: AELLO-CANARY-2f96edbdd8a3d045c2102804507a2679
# AELLO-CANARY-END
