"""Reference solve for C3-S2.

NOT A MEASUREMENT. This exercises the delivery path -- an ordered manifest, a corpus digest that
manifest re-executes to, one span triple per emitted record, and a declaration whose operator
order equals the manifest's -- so the checker set can be run end to end before any constant
exists. It delegates to the shipped baseline for that path.

The reference CURATION RECIPE that anchors the ramp is a different object and is measured at
Phase 1 on the grading host, not here. Its executed order is recorded in seed/contract.yaml:
byte normalisation and encoding repair, exact document dedup, held-out fingerprint
decontamination applied to the NORMALISED text, paragraph-granularity near-dedup with parameters
measured on this slice rather than copied from the published MinHash settings, a
length-and-repetition filter tuned on the raw slice, an interleaving emission curriculum, and
truncation-safe emission across the whole [560M, 600M] band. None of those seven worths is
asserted here; each is measured on the Phase 2 ladder over the full lever power set.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "environment", "starter"))
from starter import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-6b43a83e0d9ceb85305f340ced6b5f99
#   slot1: AELLO-CANARY-9383ea62e6f2331240c3312fe7e79f81
#   slot2: AELLO-CANARY-6541eb481d5bfc618e97be0c9968f0a8
#   slot3: AELLO-CANARY-825ff8b67f70266cb9b5c33c45596132
# AELLO-CANARY-END
