"""Reference solve for C7-S1.

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
#   slot0: AELLO-CANARY-028d61af2552c5b85c0f40c26e71a403
#   slot1: AELLO-CANARY-776e291fc5c990ea7fe9a467594be7a5
#   slot2: AELLO-CANARY-1d7e725946e835850e2abc6b662e51cc
#   slot3: AELLO-CANARY-1732723eb731ff57050b5dc23a6e1bb7
# AELLO-CANARY-END
