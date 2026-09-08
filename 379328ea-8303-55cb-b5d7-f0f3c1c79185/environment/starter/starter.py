"""Correct, unoptimised curation baseline for C3-S2. Replace it entirely if you want.

This is the published C4-style heuristic pipeline: language identification, a badwords pass, and
three-sentence-span deduplication, in the published order. It is legal, it re-executes to its own
corpus digest, and it is deliberately weak on the graded quantity -- reproducing it is worth one
tenth of the range and no more. What it exists to prove is the DELIVERY PATH: an ordered manifest,
a corpus digest that manifest reproduces, one (source_id, offset, length) span per emitted record,
and an agent-authored declaration whose operator order equals the manifest's.

Two orderings are already correct here and both are load-bearing. Decontamination runs AFTER byte
normalisation, so the fingerprint filter sees the same normalised text the containment check will;
inverting those two lines yields a pipeline that exits 0, reports zero fingerprint hits under its
own pre-normalisation count, and scores exactly 0.0 with reason eval-leak. Deduplication runs
BEFORE truncation to the token budget, so the survivor set is chosen by content rather than by
where the single pass happens to stop.
"""
import hashlib
import json
import os

SUBMISSION = "/submission/c3s2.json"
RUN_RECORD = "/workspace/run_record.json"
RAW = "/workspace/environment/data/raw"
FINGERPRINTS = "/workspace/environment/decontam/heldout_fingerprints.bin"

# The executed operator order. Array position IS the order; the verifier re-executes this list.
RECIPE = [
    {"op": "byte_normalise", "params": {"form": "NFC", "encoding_repair": True}},
    {"op": "exact_dedup_document", "params": {"digest": "sha256"}},
    {"op": "decontaminate_fingerprints", "params": {"ngram_width": 13, "source": FINGERPRINTS}},
    {"op": "near_dedup_span", "params": {"granularity": "three_sentence_span"}},
    {"op": "quality_filter_badwords", "params": {"strength": "published_c4"}},
    {"op": "language_id_filter", "params": {"language": "en", "threshold": 0.8}},
    {"op": "emit_interleaved", "params": {"buckets": 1}},
    {"op": "truncate_token_budget", "params": {"upper_band_tokens": 600000000}},
]


def main():
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    records = curate(RAW)
    emit(SUBMISSION, records)
    declare(RUN_RECORD, records)
    return 0


def curate(raw_dir):
    """Emit (record_id, source_id, offset, length, payload) tuples in emission order.

    Every payload is an EXACT SPAN of a provided raw document. The baseline never rewrites a
    byte, because span containment is a red line: select, order and truncate is the whole of
    what a legal recipe may do.
    """
    records = []
    for source_id in sorted(os.listdir(raw_dir)) if os.path.isdir(raw_dir) else []:
        blob = open(os.path.join(raw_dir, source_id), "rb").read()
        offset, index = 0, 0
        for chunk in blob.split(b"\n\n"):
            if len(chunk) >= 512:
                records.append(("%s:%06d" % (source_id, index), source_id, offset,
                                len(chunk), chunk))
                index += 1
            offset += len(chunk) + 2
    return records


def emit(path, records):
    """Write the artifact: ids and per-record digests in emission order, manifest, spans."""
    corpus = hashlib.sha256()
    for _, _, _, _, payload in records:
        corpus.update(payload)
    doc = {
        "ids": [r[0] for r in records],
        "outputs": [hashlib.sha256(r[4]).hexdigest() for r in records],
        "recipe_manifest": RECIPE,
        "corpus_sha256": corpus.hexdigest(),
        "spans": [{"source_id": r[1], "offset": r[2], "length": r[3]} for r in records],
        "consumed_tokens": sum(len(r[4].split()) for r in records),
    }
    with open(path, "w") as handle:
        json.dump(doc, handle)


def declare(path, records):
    """The AGENT'S OWN accounting. The harness measures the walls separately and its
    measurement is not what this check grades; this declaration is."""
    with open(path, "w") as handle:
        json.dump({"curation_declaration": {
            "operator_order": [step["op"] for step in RECIPE],
            "emission_curriculum": "raw shard order, single bucket",
            "build_seconds": 0.0,
            "train_seconds": 0.0,
            "records_emitted": len(records),
        }}, handle, indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
