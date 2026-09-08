"""Reference solve for C3-S1.

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
#   slot0: AELLO-CANARY-f6e63dc5578069a7c003d86e035cf00e
#   slot1: AELLO-CANARY-932aca95af39a2245aad33557f82df8b
#   slot2: AELLO-CANARY-4312fead687026fb0eadc784a8254489
#   slot3: AELLO-CANARY-fae240e6e015c5a73d05714e4672b2dd
# AELLO-CANARY-END
