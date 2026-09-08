"""Correct, unoptimised baseline for C4-S2. Replace it entirely if you want.

It exists to prove the delivery path end to end: it reads what ships, produces a well-formed
artifact at the graded path, and declares its own budget accounting. It is deliberately weak on
the graded metric -- that is what you are being asked to improve.
"""
import json
import os

SUBMISSION = '/submission/c4s2.json'


def main():
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    # A well-formed but trivial artifact. See instruction.md for the required shape:
    #   an object with an ids list and a matching outputs list
    baseline(SUBMISSION)
    with open("/workspace/run_record.json", "w") as fh:
        json.dump({"agent_budget_declaration": {"attempts_used": 1, "seconds_declared": 0.0}},
                  fh, indent=1)
    return 0



def baseline(path):
    """Emit an empty but schema-valid structured deliverable."""
    with open(path, "w") as handle:
        json.dump({"ids": [], "outputs": []}, handle)



if __name__ == "__main__":
    raise SystemExit(main())
