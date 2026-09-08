"""Reference solve for AELLO-C8-S5.

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
#   slot0: AELLO-CANARY-64e40139a41090860fd302035370d48d
#   slot1: AELLO-CANARY-5bd0be3a160298993782b68cfcf9f8a5
#   slot2: AELLO-CANARY-53a8ccda1f8385db6db18a47947ded69
#   slot3: AELLO-CANARY-6e98d3431d8759bf062af90956533987
# AELLO-CANARY-END
