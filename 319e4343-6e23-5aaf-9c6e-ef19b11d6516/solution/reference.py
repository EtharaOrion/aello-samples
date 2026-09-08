"""Reference solve for AELLO-C8-S4.

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
#   slot0: AELLO-CANARY-d6bc271759959524cc4b33281711e915
#   slot1: AELLO-CANARY-33083685cac396e42d8570cc3369d271
#   slot2: AELLO-CANARY-67da12c6a884a149696f4dc0fa2caeca
#   slot3: AELLO-CANARY-ae6fe18fd95ce501f5e15fe5ce2bf9d6
# AELLO-CANARY-END
