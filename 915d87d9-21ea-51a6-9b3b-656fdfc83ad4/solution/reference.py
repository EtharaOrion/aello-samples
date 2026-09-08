"""Reference solve for C4-S3.

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
#   slot0: AELLO-CANARY-984f0bb9464dd324952dbd7d9fad7601
#   slot1: AELLO-CANARY-6bd7f7dd136f19f5856e9f97a07752ba
#   slot2: AELLO-CANARY-6a1fe153309429d4a4f5408c6b2efb52
#   slot3: AELLO-CANARY-ad3cfdc08b790d8bc1f5c6b3a110df31
# AELLO-CANARY-END
