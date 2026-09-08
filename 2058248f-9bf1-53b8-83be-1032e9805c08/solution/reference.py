"""Reference solve for A5-02.

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
#   slot0: AELLO-CANARY-7b162fe9b60213a21fd30c9da15a2253
#   slot1: AELLO-CANARY-8d9e78a1f4db48d5b1fa3795637569cc
#   slot2: AELLO-CANARY-43110090c35dee48e46c8cc99dc5830d
#   slot3: AELLO-CANARY-a9789180a5f53d2358dd783cc17a74a4
# AELLO-CANARY-END
