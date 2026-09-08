"""Reference solve for A5-01.

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
#   slot0: AELLO-CANARY-ce91d786a4385a70bb0a317142d02254
#   slot1: AELLO-CANARY-ba6d0f8c04db42677376dc3be6a94452
#   slot2: AELLO-CANARY-d7df51fe959ef0a4fa36da6b784c8c65
#   slot3: AELLO-CANARY-072efa8e79e13f69b7fb43c25232a6d6
# AELLO-CANARY-END
