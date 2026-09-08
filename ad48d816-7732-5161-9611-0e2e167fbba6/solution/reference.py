"""Reference solve for AELLO-C8-S13.

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
#   slot0: AELLO-CANARY-61cc355fac243dd3dad4afc3b9a84a97
#   slot1: AELLO-CANARY-a107ee07d664b4b1f66d41dffb3ee882
#   slot2: AELLO-CANARY-306614f257370612315448f37e23f919
#   slot3: AELLO-CANARY-21d137251a71485a1d61069e01973a95
# AELLO-CANARY-END
