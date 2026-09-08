"""Reference solve for AELLO-C8-S12.

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
#   slot0: AELLO-CANARY-67655e24f59c37a970327975520caf38
#   slot1: AELLO-CANARY-f3e8eca232d018e338784243dcffc30b
#   slot2: AELLO-CANARY-ef51c9a8a85c57f764e8ab7422b035bf
#   slot3: AELLO-CANARY-52d049979efe780c34f5f68358752138
# AELLO-CANARY-END
