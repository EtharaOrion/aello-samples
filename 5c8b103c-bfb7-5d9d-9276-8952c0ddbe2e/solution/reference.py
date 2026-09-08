"""Reference solve for AELLO-C6-S2.

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
#   slot0: AELLO-CANARY-6fba8de0ad64e1b823dba2be01e8ffac
#   slot1: AELLO-CANARY-f316b670f6f37e03b147995e2ec96c47
#   slot2: AELLO-CANARY-0fe6e7abe0d04af8a82cc4703cc2353d
#   slot3: AELLO-CANARY-a056eaed8430320e4f458335f5f1f722
# AELLO-CANARY-END
