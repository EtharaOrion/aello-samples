"""Reference solve for C2-S2.

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
#   slot0: AELLO-CANARY-323a294673512fc8b207f58340b7b58f
#   slot1: AELLO-CANARY-3f898a18d5e6f83c266c22f0b561dd0e
#   slot2: AELLO-CANARY-1ac3fab5ecd8397d1398bde2c191f6c6
#   slot3: AELLO-CANARY-e00cbd212d4e05bf3576f0b24bc52caa
# AELLO-CANARY-END
