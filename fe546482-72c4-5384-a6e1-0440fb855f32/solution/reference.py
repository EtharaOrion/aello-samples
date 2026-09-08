"""Reference solve for AELLO-C8-S6.

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
#   slot0: AELLO-CANARY-a0e42c0065f28a3ae7cd80750e6deb25
#   slot1: AELLO-CANARY-78fa5ff042735b8899aa461525adfeb5
#   slot2: AELLO-CANARY-b9f439e31b6c2b3bf2a0eca0f82e6642
#   slot3: AELLO-CANARY-398a99cbdda81fcdf9fc88a19f7daa24
# AELLO-CANARY-END
