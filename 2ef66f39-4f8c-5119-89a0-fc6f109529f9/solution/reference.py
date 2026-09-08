"""Reference solve for C1-S1.

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
#   slot0: AELLO-CANARY-609155f956d380f15b3b4d72406398e3
#   slot1: AELLO-CANARY-340a728dcc236349192fdbce4d39ad6f
#   slot2: AELLO-CANARY-62c169bfe63371d0f6ea5cd7ec88f0a9
#   slot3: AELLO-CANARY-dcde825fef9015fb05e906169c5c3183
# AELLO-CANARY-END
