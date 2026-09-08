"""Reference solve for AELLO-C8-S7.

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
#   slot0: AELLO-CANARY-34477cdc8d45a4268664ce6af225566f
#   slot1: AELLO-CANARY-be3116e30ccdfbd449346d06d270ee88
#   slot2: AELLO-CANARY-aa4237d8ef3a4e17d01dff3c00bf57f1
#   slot3: AELLO-CANARY-5994bef66a56e9cfd85895e8bafffa3f
# AELLO-CANARY-END
