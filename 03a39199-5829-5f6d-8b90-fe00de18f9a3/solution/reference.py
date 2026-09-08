"""Reference solve for AELLO-C8-S2.

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
#   slot0: AELLO-CANARY-f52bbbe19703ede56d5674eb4c5d7dc3
#   slot1: AELLO-CANARY-23a009b98bd344917e87ef548b584156
#   slot2: AELLO-CANARY-3c8086a68a27b36b1f2ae16874efe9da
#   slot3: AELLO-CANARY-90b0dc8de7fd77d4eb5b7482318e3440
# AELLO-CANARY-END
