"""Runnable baseline for AELLO-C8-S7. Correct, honest -- and deliberately weak on both axes.

WHAT IT DOES. It frames each graded utterance at a 400-sample window and a 160-sample hop (25 ms
and 10 ms at 16 kHz), takes 40 coarse log-magnitude bands per frame, averages them over the
utterance, and picks the nearest class centroid estimated from the training split. That is a real
classifier, it is cheap enough to sit well inside the 20 ms per-utterance bound, and it is far
below what this task is asking for. Replace all of it.

IT DOES NOT TIME ITSELF. There is no clock in this file, and there is none in your solve either:
`harness_utterance_latency_ms` is written by the harness, and a millisecond figure the graded
process reported about itself is red line 2. Profile your design during development with whatever
you like -- that is how you form the `latency_gate_passed` claim this record carries -- but the
number that scores is the harness's.

IT FABRICATES NOTHING. If the audio has not landed in your tree, it writes a constant decision
vector of the published length, records `corpus_status: not-landed`, and lets
`no_keyword_claims_more_than_the_balanced_share_cap` fail on it -- which is the correct outcome for
a vector that decided nothing. It never writes a label it derived from audio it did not read.

THREE THINGS WORTH KNOWING BEFORE YOU BUILD ON IT.

  THE LABEL MAPPING IS PUBLISHED, NOT DISCOVERED. `KEYWORD_ROSTER` below is the graded mapping:
  index 0 is `backward`, index 34 is `zero`. Build your encoder from it. A model trained against a
  different ordering scores near 1/35 and looks exactly like a model that did not learn.

  ONE FRONT END SERVES BOTH MEASUREMENTS. The digest recorded at timing and the digest recorded at
  prediction are the same digest here because the same block ran both passes. If you sweep the
  front end, sweep it before the graded run, not between the two passes of it.

  THE GATE IS A MAXIMUM. This baseline's cost per utterance is a few hundred multiply-accumulates
  and it clears the bound with room. Yours will not, if the waveform branch runs at 16 kHz or the
  mel resolution climbs while the hop drops -- and one late utterance out of 3500 is a zero.
"""
import hashlib
import json
import os

BUNDLE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BUNDLE, "environment", "data")
SUBMISSION = os.environ.get("AELLO_SUBMISSION", "/submission/preds_c8s7.npy")
RECORD = os.environ.get("AELLO_RUN_RECORD", "/workspace/run_record.json")
TRAIN_AUDIO = os.path.join(DATA, "speech_commands_v002_train.npz")
GRADED_AUDIO = os.path.join(DATA, "graded_utterances.npz")
KEYWORD_INDEX_FILE = os.path.join(DATA, "keyword_index.json")

GRADED_UTTERANCE_COUNT = 3500
UTTERANCE_SAMPLES = 16000
FRONT_END = {"sample_rate_hz": 16000, "window_samples": 400, "hop_samples": 160,
             "mel_bands": 40, "frames_per_utterance": 1 + (16000 - 400) // 160}

# The graded mapping, mirroring environment/data/keyword_index.json. Embedded so the starter runs
# before the corpus lands; the shipped file is authoritative when it is there.
KEYWORD_ROSTER = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward",
    "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off",
    "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two",
    "up", "visual", "wow", "yes", "zero",
]


def keyword_roster():
    """The shipped roster when it has landed, the embedded pin otherwise."""
    if not os.path.exists(KEYWORD_INDEX_FILE):
        return list(KEYWORD_ROSTER)
    try:
        with open(KEYWORD_INDEX_FILE) as handle:
            doc = json.load(handle)
    except Exception:
        return list(KEYWORD_ROSTER)
    words = doc.get("keywords", doc) if isinstance(doc, dict) else doc
    return list(words) if isinstance(words, list) and words else list(KEYWORD_ROSTER)


def front_end_digest(block):
    """sha256 over the canonical form of the front-end block, as the checker recomputes it."""
    canonical = json.dumps(dict((str(k), block[k]) for k in sorted(block)),
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def archive_digest(path):
    """sha256 of a corpus archive, or None when it has not landed."""
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def band_features(waveforms):
    """Mean over frames of 40 coarse log-magnitude bands, one row per utterance.

    The whole front end, and the only place the three declared levers are used: the window and
    hop set the framing, the band count sets the width. It is a plain magnitude spectrum binned
    uniformly rather than a mel filterbank -- weaker, and honest about being weaker.
    """
    import numpy as np
    window, hop = FRONT_END["window_samples"], FRONT_END["hop_samples"]
    bands = FRONT_END["mel_bands"]
    starts = range(0, UTTERANCE_SAMPLES - window + 1, hop)
    taper = np.hanning(window).astype("float32")
    rows = []
    for clip in waveforms:
        clip = np.asarray(clip, dtype="float32")
        frames = np.stack([clip[s:s + window] * taper for s in starts])
        spectrum = np.abs(np.fft.rfft(frames, axis=1))
        edges = np.linspace(0, spectrum.shape[1], bands + 1).astype("int64")
        binned = np.stack([spectrum[:, edges[b]:max(edges[b + 1], edges[b] + 1)].mean(axis=1)
                           for b in range(bands)], axis=1)
        rows.append(np.log1p(binned).mean(axis=0))
    return np.stack(rows)


def centroids(features, labels, classes):
    """One mean feature vector per keyword; the global mean stands in for an unseen keyword."""
    import numpy as np
    overall = features.mean(axis=0)
    table = np.stack([features[labels == k].mean(axis=0) if (labels == k).any() else overall
                      for k in range(classes)])
    return table


def decide(graded_features, table):
    """Nearest centroid in Euclidean distance. One decision per utterance, no batching tricks."""
    import numpy as np
    distances = ((graded_features[:, None, :] - table[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1).astype("int64")


def predict():
    """(decisions, corpus_status). A constant vector, honestly labelled, when nothing landed."""
    if not (os.path.exists(TRAIN_AUDIO) and os.path.exists(GRADED_AUDIO)):
        import numpy as np
        return np.zeros((GRADED_UTTERANCE_COUNT,), dtype="int64"), "not-landed"
    import numpy as np
    with np.load(TRAIN_AUDIO, allow_pickle=False) as train:
        table = centroids(band_features(train["waveform"]), train["y"].astype("int64"),
                          len(keyword_roster()))
    with np.load(GRADED_AUDIO, allow_pickle=False) as graded:
        return decide(band_features(graded["waveform"]), table), "landed"


def write_predictions(path, decisions):
    import numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, decisions)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    before = archive_digest(GRADED_AUDIO)
    decisions, corpus_status = predict()
    delivered = write_predictions(SUBMISSION, decisions)

    # A real replay: the whole decision path runs again and the bytes are digested again. The
    # labels must come back identical. No timing is compared, because none was taken here.
    replayed, _ = predict()
    replay_digest = write_predictions(SUBMISSION + ".replay", replayed)
    os.remove(SUBMISSION + ".replay")

    digest = front_end_digest(FRONT_END)
    os.makedirs(os.path.dirname(RECORD), exist_ok=True)
    with open(RECORD, "w") as handle:
        json.dump({
            "keyword_index": keyword_roster(),
            "front_end": dict(FRONT_END),
            # The same block ran the timed pass and the scored pass, so the two digests are the
            # same digest. That is the point of recording two of them.
            "front_end_digest_at_timing": digest,
            "front_end_digest_at_prediction": digest,
            "latency_source": "harness_per_utterance_timer",
            # A claim, from this front end's arithmetic -- 98 frames of a 400-point transform and
            # a 35-way distance -- and not a measurement. The harness measures.
            "latency_gate_passed": True,
            "graded_audio_digest_before": before,
            "graded_audio_digest_after": archive_digest(GRADED_AUDIO),
            "paths_written": [SUBMISSION, RECORD],
            # This baseline mixes no background noise. Declaring the lever as unused is a
            # statement about it; leaving the key out is silence about it.
            "augmentation": {"applied_to": "none", "background_clips": [],
                             "snr_db_range": [0.0, 0.0]},
            "replay": {"artifact_sha256": replay_digest, "front_end_digest": digest},
            "agent_budget_declaration": {"train_seconds": 0, "inference_utterances_timed": 0,
                                         "attempts_used": 1},
            "corpus_status": corpus_status,
            "artifact_sha256": delivered,
        }, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
