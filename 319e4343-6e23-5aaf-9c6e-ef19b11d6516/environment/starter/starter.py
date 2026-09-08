"""Correct, unoptimised baseline for AELLO-C8-S4. Replace it entirely if you want.

It exists to prove the delivery path end to end: it reads what ships, produces a well-formed
artifact at the graded path, and declares its own budget accounting. It is deliberately weak on
the graded metric -- that is what you are being asked to improve.
"""
import json
import os

SUBMISSION = '/submission/preds_c8s4.npy'


def main():
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    # A well-formed but trivial artifact. See instruction.md for the required shape:
    #   integer label array, one row per graded instance
    baseline(SUBMISSION)
    with open("/workspace/run_record.json", "w") as fh:
        json.dump({"agent_budget_declaration": {"attempts_used": 1, "seconds_declared": 0.0}},
                  fh, indent=1)
    return 0



def baseline(path):
    """Emit a well-formed but deliberately weak dense prediction."""
    import numpy as np
    np.save(path, np.zeros((1,), dtype="int64"))



if __name__ == "__main__":
    raise SystemExit(main())
