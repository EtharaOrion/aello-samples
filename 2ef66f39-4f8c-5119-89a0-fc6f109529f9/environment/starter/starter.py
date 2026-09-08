"""Correct, unoptimised baseline for C1-S1. Replace it entirely if you want.

It exists to prove the delivery path end to end: it reads what ships and writes a well-formed
artifact at the graded path. It is deliberately weak on the graded metric -- that is what you
are being asked to improve.

It does NOT write run_record.json or the horizon report. Both are harness-owned: the run record
carries the clock origin, the harness-assigned checkpoint stamps and the shim's token
attribution, and the horizon report carries every checkpoint's bits-per-byte on each shard. The
five sections below are declarations, and each one is reconciled against those carriers.
"""
import json
import os

SUBMISSION = '/submission/c1s1.json'


def main():
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    # A well-formed but empty artifact. See instruction.md for the field-by-field shape:
    #   era_calendar, schedule, token_ledger, checkpoint_ledger, tokenizer
    baseline(SUBMISSION)
    return 0


def baseline(path):
    """Emit an empty but structurally valid deliverable.

    Every number here is a placeholder. era_calendar is the one section that cannot be filled in
    from this file at all: cutoff_epoch_day is disclosed in instruction.md, and
    corpus_last_pub_epoch_day is a measurement over environment/data/docmeta.bin.
    """
    empty = {
        "era_calendar": {"cutoff_epoch_day": None, "corpus_last_pub_epoch_day": None,
                         "blind_band_days": None, "recency_window_start_epoch_day": None},
        "schedule": {"oldest_era_floor_share": None, "phases": []},
        "token_ledger": {"served_tokens_total": 0, "per_phase": []},
        "checkpoint_ledger": {"setup_seconds": None, "joint_crossing": None,
                              "seconds_to_joint_bar": None, "declared_forward_slope": None,
                              "checkpoints": []},
        "tokenizer": {"entry_count": None, "fertility_slice_start_epoch_day": None,
                      "realized_bytes_per_token": None, "trained_bytes_total": None},
    }
    with open(path, "w") as handle:
        json.dump(empty, handle, indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
