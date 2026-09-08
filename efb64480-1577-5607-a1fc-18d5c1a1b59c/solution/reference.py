"""Reference solve for AELLO-C8-S8.

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
#   slot0: AELLO-CANARY-499b6720890d6f693baa338ed2604792
#   slot1: AELLO-CANARY-28b9e5a75be70673ff77e11abc1a8559
#   slot2: AELLO-CANARY-fe83372c9521a691808f135217540f44
#   slot3: AELLO-CANARY-149bdddb0ee8828149dfabe093875559
# AELLO-CANARY-END
