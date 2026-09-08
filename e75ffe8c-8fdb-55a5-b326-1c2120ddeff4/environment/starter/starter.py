"""Correct, unoptimised streaming baseline for A5-03. Replace it entirely if you want.

It exists to prove the delivery path end to end: it writes an export directory whose manifest the
verifier can read, keeps its log-mel front end inside the export, declares a per-layer lookahead
that sums well inside the 320 ms bound, and decodes greedily with the collapse in the right order.
It is deliberately weak on word error rate -- a small unidirectional LSTM stack, no augmentation,
no plateau handling -- and that is what you are being asked to improve.

Two things it does NOT do, deliberately:

  * it does not write run_record.json. That file is the HARNESS's record of what it observed, and
    a run in which the agent writes both sides of the comparison in chk_log_fidelity measures
    nothing. Your own claims go in reward_log.jsonl.
  * it does not read any transcript but environment/data/transcripts.jsonl. Every transcript byte
    read as supervision is digested against the pinned shipped-shard table, and a digest that is
    not in that table scores zero with reason EXTERNAL_SUPERVISION.
"""
import json
import os

EXPORT_DIR = "/submission/export"
TRAIN_ENTRYPOINT = "/submission/train.py"
MANIFEST = "export_manifest.json"
PARAMETERS = "encoder.fp16.bin"
REWARD_LOG = "/workspace/reward_log.jsonl"

DATA = "/workspace/environment/data"
TRANSCRIPTS = os.path.join(DATA, "transcripts.jsonl")
LABEL_VOCABULARY = os.path.join(DATA, "label_vocabulary.json")

# The front end lives INSIDE the export, so these five constants are graded bytes. Compute them in
# exactly one place and have train.py import that place: the single most expensive silent failure
# available on this slot is a training shift that the export does not share.
FRONTEND = {"sample_rate": 16000, "frame_shift_ms": 10.0, "frame_length_ms": 25.0,
            "n_mels": 80, "preemphasis": 0.97}

# Right context is a SUM down the stack. 32 frames is the whole allowance at a 10 ms shift; this
# baseline spends 6 and leaves the rest on the table.
LAYER_RIGHT_CONTEXT_FRAMES = [1, 1, 1, 1, 1, 1]


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    parameters = export_baseline_encoder(os.path.join(EXPORT_DIR, PARAMETERS))
    manifest = {
        "parameter_file": PARAMETERS,
        "parameter_bytes": os.path.getsize(parameters),
        "frontend": dict(FRONTEND),
        "graph_ops": ["stft", "mel", "log", "subsample", "lstm_cell", "layernorm", "linear",
                      "softmax", "argmax"],
        "layer_right_context_frames": list(LAYER_RIGHT_CONTEXT_FRAMES),
        "decode": {"method": "greedy", "blank_index": blank_index(),
                   "collapse": ["merge_repeats", "remove_blank"]},
        "train_entrypoint": os.path.basename(TRAIN_ENTRYPOINT),
        "label_vocabulary_digest": digest(LABEL_VOCABULARY),
    }
    with open(os.path.join(EXPORT_DIR, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    log_attempt(wer_librispeech=None, wer_commonvoice=None, elapsed_seconds=None)
    return 0


def blank_index():
    """A reserved index, not an aliased character.

    The shipped vocabulary is a private permutation of the character index space, so an index you
    assume is spare may well be a letter. Take one the vocabulary assigns to nothing, and use the
    SAME index in the CTC loss during training.
    """
    if not os.path.exists(LABEL_VOCABULARY):
        return 0
    with open(LABEL_VOCABULARY) as fh:
        vocab = json.load(fh)
    taken = {v for v in vocab.values() if isinstance(v, int) and not isinstance(v, bool)}
    index = 0
    while index in taken:
        index += 1
    return index


def digest(path):
    import hashlib
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def export_baseline_encoder(path):
    """Emit a well-formed but deliberately weak set of serialized parameters.

    A real solve writes the tensors its training run produced. This writes a placeholder of the
    right shape and well under the 31457280-byte ceiling, so the delivery path can be exercised
    before any encoder exists.
    """
    with open(path, "wb") as fh:
        fh.write(b"\x00" * (1 << 20))
    return path


def log_attempt(wer_librispeech, wer_commonvoice, elapsed_seconds):
    """Append one record to YOUR log. The harness writes its own, and the two are compared."""
    record = {"wer_librispeech": wer_librispeech, "wer_commonvoice": wer_commonvoice,
              "elapsed_seconds": elapsed_seconds}
    with open(REWARD_LOG, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
