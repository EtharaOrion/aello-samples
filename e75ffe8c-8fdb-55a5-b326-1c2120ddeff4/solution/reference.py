"""Reference solve for A5-03.

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
#   slot0: AELLO-CANARY-cd2709372b26a344b6894f9329e0c198
#   slot1: AELLO-CANARY-c4325951b4235a6931e7ab18bcfc6ec8
#   slot2: AELLO-CANARY-b21b8b15e54a1fe3887f52fb268b4ee6
#   slot3: AELLO-CANARY-29a2a57471ec09178a9ab0d0f4fff0a4
# AELLO-CANARY-END
