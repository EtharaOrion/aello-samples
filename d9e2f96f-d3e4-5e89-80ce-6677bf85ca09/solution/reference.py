"""Reference solve for AELLO-C8-S11.

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
#   slot0: AELLO-CANARY-741a9ceccad662e7022448e7f797fc44
#   slot1: AELLO-CANARY-521001e5368bb237f8f227a575ae6e33
#   slot2: AELLO-CANARY-b2e27e5644d2d284c7954bf9baaca2d1
#   slot3: AELLO-CANARY-492407b078d8e0821be18962ad1f325f
# AELLO-CANARY-END
