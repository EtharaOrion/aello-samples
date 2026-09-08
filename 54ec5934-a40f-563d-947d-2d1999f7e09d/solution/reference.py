"""Reference solve for C7-S2.

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
#   slot0: AELLO-CANARY-061f6f6b7b2a54a9210b0e6700581284
#   slot1: AELLO-CANARY-cd4ebec219f03a949b2629082c125744
#   slot2: AELLO-CANARY-ae1bbcf0a8a058eee6be875bed4c1963
#   slot3: AELLO-CANARY-e0fbd2f0bc1ee664455d6ba374c691f1
# AELLO-CANARY-END
