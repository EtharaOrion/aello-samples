"""Correct, unoptimised baseline for C3-S3. Replace it entirely if you want.

It exists to prove the delivery path end to end: it reads what ships, produces a well-formed
artifact at the graded path in the shape instruction.md publishes, and declares its own budget
accounting and its own selection evidence. It is deliberately weak on the graded metric -- that is
what you are being asked to improve.

WHAT IT DOES. It takes the published-style mixture (web-heavy, the proportions every public recipe
agrees on), emits whole documents in natural order round-robin across the four domains, declares
one epoch everywhere and document-granularity deduplication at both scopes. It measures nothing.
That is the point of the starter and it is also the first trap: the published proportions are
tuned for aggregate loss, and this slot is graded on the WORST of four domains.
"""
import json
import os

SUBMISSION = '/submission/c3s3.json'
RUN_RECORD = '/workspace/run_record.json'
SOURCES = '/workspace/environment/data/sources.json'
DOMAINS = ("web", "encyclopedic", "scientific", "mathematical")

# The published-style mixture, i.e. what a public recipe table would tell you. It is legal, it is
# in budget, and it is not the answer.
PUBLISHED_WEIGHTS = {"web": 0.67, "encyclopedic": 0.13, "scientific": 0.13, "mathematical": 0.07}


def main():
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    baseline(SUBMISSION)
    with open(RUN_RECORD, "w") as fh:
        json.dump({
            "agent_budget_declaration": {
                "build_seconds": 0.0,
                "train_seconds": 0.0,
                "per_domain_weights": dict(PUBLISHED_WEIGHTS),
                "repetition_epochs": dict((d, 1) for d in DOMAINS),
                "emission_curriculum": "round-robin over the four domains, natural order within",
            },
            # HONEST AND EMPTY. The starter measures nothing here, so it claims nothing here, and
            # this check fails for the starter rather than being satisfied by a citation of the
            # shipped reference artifacts. Fill it with measurements you actually ran.
            "selection_evidence": {"cited_sources": [], "own_measurements": []},
        }, fh, indent=1, sort_keys=True)
    return 0


def load_sources():
    """source_id -> domain, from the shipped four-corpus manifest; {} before it lands."""
    if not os.path.exists(SOURCES):
        return {}
    with open(SOURCES) as handle:
        doc = json.load(handle)
    doc = doc.get("sources", doc) if isinstance(doc, dict) else {}
    out = {}
    for source_id, entry in doc.items():
        if isinstance(entry, dict) and entry.get("domain") in DOMAINS:
            out[str(source_id)] = entry
    return out


def baseline(path):
    """Whole documents, natural order, round-robin across the four domains."""
    sources = load_sources()
    by_domain = dict((d, sorted(s for s, e in sources.items() if e["domain"] == d))
                     for d in DOMAINS)
    ids, outputs, spans = [], [], []
    depth = max([len(v) for v in by_domain.values()] or [0])
    for index in range(depth):
        for domain in DOMAINS:
            bucket = by_domain[domain]
            if index >= len(bucket):
                continue
            source_id = bucket[index]
            length = int(sources[source_id].get("bytes", 0))
            if length <= 0:
                continue
            ids.append("rec-%s-%06d" % (domain, index))
            outputs.append(sources[source_id].get("sha256"))
            spans.append([{"source_id": source_id, "offset": 0, "length": length}])
    with open(path, "w") as handle:
        json.dump({
            "ids": ids,
            "outputs": outputs,
            "spans": spans,
            "corpus_sha256": None,
            "consumed_tokens": None,
            "mixture_manifest": {
                "weights": dict(PUBLISHED_WEIGHTS),
                "weight_basis": "post_dedup_tokens",
                "repetition_epochs": dict((d, 1) for d in DOMAINS),
                "dedup": {"within_domain_granularity": "document",
                          "across_domain_granularity": "document"},
                "selection": dict((d, {"policy": "keep_everything"}) for d in DOMAINS),
                "emission": {"curriculum": "round_robin"},
            },
        }, handle, indent=1, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
