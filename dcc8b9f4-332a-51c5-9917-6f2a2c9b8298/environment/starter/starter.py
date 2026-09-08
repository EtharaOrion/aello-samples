"""Correct, unoptimised baseline for C3-S1. Replace it entirely if you want.

It exists to prove the delivery path end to end: it builds a vocabulary from scratch, encodes and
decodes losslessly, records the raw UTF-8 byte denominator, declares a stage manifest, and writes
the run record the checkers read. It is deliberately weak on the graded metric -- a 256-entry
byte-level vocabulary costs a flat 8 bits per byte under a uniform code, which is the worst
defensible number on this axis and exactly what you are being asked to improve.

Two things it does NOT do, on purpose, because doing them is how the metric moves:
  - it learns no merges, so every byte is its own token and the token count equals the byte count;
  - it samples nothing by script, so it has no opinion about Arabic, Devanagari or Tamil.

One thing it refuses to do: normalise. NFKC would fold the Arabic ligature and the fi ligature in
the fixtures below and reorder the Tamil combining marks, decode would not return the input bytes,
and the round-trip check zeroes the run with reason roundtrip-not-identity.
"""
import hashlib
import json
import os
import time

SUBMISSION = '/submission/c3s1.json'
RUN_RECORD = '/workspace/run_record.json'

# One entry per byte value. This is the whole vocabulary: byte fallback and nothing else.
BYTE_ENTRIES = ["<0x%02X>" % value for value in range(256)]

# The five disclosed adversarial classes. Each string is chosen because a normalising pipeline
# changes it: the ligature decomposes, the tatweel is stripped, the Tamil marks reorder, the lone
# surrogate becomes U+FFFD, and NFKC rewrites the last one outright.
FIXTURES = [
    ("arabic_presentation_forms", "ﻻ العربية"),
    ("tatweel", "مـــرحبا"),
    ("tamil_combining", "நிலம் க்ஷ"),
    ("lone_surrogate", "before\ud800after"),
    ("nfkc_unstable", "ﬁve ① ½ ㎒"),
]


def utf8(text):
    """Raw UTF-8 bytes, surrogatepass so the lone-surrogate fixture is representable."""
    return text.encode("utf-8", "surrogatepass")


def encode(text):
    """Every byte becomes the token whose id is that byte. Lossless by construction."""
    return list(utf8(text))


def decode(token_ids):
    """Inverse of encode. Nothing is folded, stripped, reordered or replaced."""
    return bytes(token_ids).decode("utf-8", "surrogatepass")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    started = time.time()
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    vocabulary_bytes = json.dumps(BYTE_ENTRIES, sort_keys=True).encode("utf-8")
    source = digest(open(os.path.abspath(__file__), "rb").read())

    rows, byte_total, token_total = [], 0, 0
    for name, text in FIXTURES:
        token_ids = encode(text)
        rows.append({"fixture_class": name, "raw_utf8": text, "token_ids": token_ids,
                     "decoded_utf8": decode(token_ids), "raw_utf8_bytes": len(utf8(text))})
        byte_total += len(utf8(text))
        token_total += len(token_ids)

    # A uniform code over 256 symbols spends exactly 8 bits on every byte. The denominator is the
    # RAW byte total, never the token total, and for this baseline the two happen to coincide
    # because the tokenizer is byte-level -- which is precisely why the baseline is weak.
    bits_per_byte = 8.0
    submission = {
        "vocabulary": {"entries": BYTE_ENTRIES,
                       "declared_entry_count": len(set(BYTE_ENTRIES)),
                       "artifact_sha256": digest(vocabulary_bytes)},
        "stage_manifest": [
            {"stage": "enumerate_byte_alphabet", "entry_point": "starter.py:main",
             "sha256": source},
            {"stage": "encode_stream", "entry_point": "starter.py:encode", "sha256": source},
        ],
        "roundtrip_fixtures": rows,
        "bpb_accounting": {"denominator_basis": "raw_utf8_bytes",
                           "raw_utf8_byte_total": byte_total,
                           "token_total": token_total,
                           "nll_bits_total": bits_per_byte * byte_total,
                           "declared_bits_per_byte": bits_per_byte},
    }
    with open(SUBMISSION, "w") as handle:
        json.dump(submission, handle)

    with open(RUN_RECORD, "w") as handle:
        json.dump({
            "stage_execution_order": ["enumerate_byte_alphabet", "encode_stream"],
            "manifest_replay_vocabulary_sha256": digest(vocabulary_bytes),
            "frozen_trainer_digests": {"before": frozen_trainer_digests(),
                                       "after": frozen_trainer_digests()},
            "guarded_paths_touched": [],
            "graded_region_wall_seconds": time.time() - started,
            "agent_budget_declaration": {"vocab_size": len(BYTE_ENTRIES),
                                         "encode_seconds": round(time.time() - started, 6),
                                         "train_seconds": 0.0,
                                         "script_sampling": "none: the baseline samples nothing"},
        }, handle, indent=1)
    return 0


def frozen_trainer_digests():
    """SHA-256 of every file under the frozen trainer, taken the same way before and after.

    The graded run must leave these unchanged. Reading them twice from the same tree is what a
    real run does either side of training; a run that modifies the trainer moves an entry here and
    scores zero with reason frozen-trainer-modified.
    """
    root = os.path.join("/workspace", "environment", "trainer")
    if not os.path.isdir(root):
        return {}
    out = {}
    for base, _dirs, names in os.walk(root):
        for name in sorted(names):
            path = os.path.join(base, name)
            out[os.path.relpath(path, root)] = digest(open(path, "rb").read())
    return out


if __name__ == "__main__":
    raise SystemExit(main())
