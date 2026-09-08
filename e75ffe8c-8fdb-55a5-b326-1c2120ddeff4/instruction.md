# A5-03

Design and train a streaming acoustic encoder with CTC **from scratch** on the provided
LibriSpeech train-clean-100 shard, inside one H100 for 1500 seconds of wall clock per graded run,
timed by the harness from your first read under `environment/data`. Export it under a bound
operator envelope with at most 30 MiB of serialized parameter bytes and a hard streaming
constraint of at most **320 ms of algorithmic right context**, verified by a causality probe
rather than taken on your word. Greedy decoding only: no beam, no lexicon, no external language
model. Minimize word error rate jointly on a private speaker-disjoint LibriSpeech held-out shard
and on a private Common Voice English shard.

## What is graded

Your deliverable is the directory **`/submission/export/`**, and the verifier **loads it and runs
it** frame by frame over private audio it holds. A saved array of predictions is not admissible:
an array cannot be perturbed, and the streaming constraint is only meaningful as a perturbation
probe over a running model.

`/submission/export/` must contain `export_manifest.json` and the parameter file it names, and
`/submission/train.py` must be the script that produced them. The manifest carries:

```json
{
  "parameter_file": "encoder.fp16.bin",
  "frontend": {"sample_rate": 16000, "frame_shift_ms": 10.0, "frame_length_ms": 25.0,
               "n_mels": 80, "preemphasis": 0.97},
  "graph_ops": ["stft", "mel", "log", "subsample", "...", "argmax"],
  "layer_right_context_frames": [2, 4, 4, 4, 4, 6],
  "decode": {"method": "greedy", "blank_index": 0,
             "collapse": ["merge_repeats", "remove_blank"]},
  "train_entrypoint": "train.py",
  "label_vocabulary_digest": "<sha256 of environment/data/label_vocabulary.json>"
}
```

- `frontend` -- the **log-mel front end runs inside the export**, so `stft` and `mel` are required
  members of `graph_ops`. These five constants are compared, field for field, against the ones
  the harness watched `train.py` compute. A training-time shift of 10 ms reimplemented at 12.5 ms
  inside the export raises nothing and roughly doubles graded WER; it is a knockout here.
- `layer_right_context_frames` -- one entry per layer. **The bound is on the SUM**: at most 32
  frames, which is 320 ms at a 10 ms shift. Lookahead composes down the stack.
- `decode` -- `method` must be exactly `greedy`; `blank_index` must be an index the shipped
  permuted label vocabulary assigns to no character, and must equal the blank index your CTC loss
  used; `collapse` must list `merge_repeats` **before** `remove_blank`.
- `graph_ops` -- every symbol must lie inside the bound operator envelope, which contains no
  bidirectional recurrence and no uncentred-attention symbol.

Your score is `r_det`, computed by the verifier in a separate, network-isolated, CPU-only
environment from state it measures itself: word error rate under greedy CTC decoding on the
private LibriSpeech held-out shard and on the private Common Voice shard, with a pinned text
normalizer. WER is an edit distance -- `(substitutions + deletions + insertions)` over the
**reference** word count -- not an accuracy, and the reward is a **product** across the two
corpora, so the worse corpus binds.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The two records

- **`run_record.json` is written by the harness, not by you.** It carries the first-data-touch and
  export-complete instants, the front-end constants it watched `train.py` compute, the op sequence
  it observed, the causality-probe results at the offsets it drew this attempt, the per-shard
  edit-operation counts, the digests of every transcript byte read as supervision, the guarded
  paths touched, the replay digests, and the per-attempt summaries. Do not write it.
- **`reward_log.jsonl` is yours.** Append one JSON object per attempt carrying at least
  `wer_librispeech`, `wer_commonvoice` and `elapsed_seconds`. The last record's WER values must
  sit within 0.02 of the harness measurement and its seconds within ten percent of the
  harness-measured first-data-touch-to-export interval. Record the probe runs you discard too.

## The budget

- 1500 s per graded run on one H100, measured by the harness from your **first read under
  `environment/data`**. Unpacking, imports and CUDA warm-up before that read are free; everything
  after it is charged. An export that first exists after the kill is not an export.
- Up to 50 attempts. `max_timeout` bounds the refinement loop. Your reward history is carried
  between attempts, and each attempt after the first must carry the prior attempt's WER pair.
- Fifty full-budget runs is most of the window, so you can afford roughly one full-budget probe
  per idea. Partial-budget ranking is the affordable route and it is the route that misleads.

## Red lines

Any of these scores exactly zero:

- A causality-probe failure, at any offset: reason `NONCAUSAL_CONTEXT`.
- Serialized parameter bytes above 31457280, counted off the delivered file: reason
  `PARAM_BYTES_EXCEEDED`.
- Any beam search, external language model or lexicon in the export's decode path: reason
  `DECODE_CONTRACT_VIOLATED`, refused loudly in the environment rather than graded down.
- Any transcript read as supervision whose bytes are not the shipped shard's transcript bytes
  under the private label permutation -- pseudo-labels from an external model included: reason
  `EXTERNAL_SUPERVISION`.
- Any write under `tests/`, `solution/` or `environment/data/`, or any held-out digest mismatch:
  reasons `GRADING_TREE_WRITE` and `HELDOUT_DIGEST_MISMATCH`.
- Overrunning 1500 s without a complete export.
- A submission whose reward moves by more than tau = 0.03 under a second private label
  permutation: reason `PRECOMPUTED_PARAMS`.
- A WER gap above 0.02 across the two sibling private LibriSpeech shards: reason
  `SHARD_GAP_EXCEEDED`.

## What ships

- `environment/data/train-clean-100/` -- the 100.6-hour audio shard.
- `environment/data/transcripts.jsonl` -- the transcripts, already carrying the private label
  permutation. **This is the only admissible supervision.**
- `environment/data/label_vocabulary.json` -- the permuted character index space. A public output
  vocabulary tells you nothing about it.
- `environment/starter/` -- a correct, unoptimised streaming baseline you may replace entirely.

## Traps

- Ranks architecture families on 300-second probe runs, which measure exit speed from the CTC
  alignment plateau rather than final WER, and the ranking inverts by 1500 s.
- Uses a bidirectional layer or a centred convolution because its probe curve is best; the
  perturbation probe zeroes the attempt with no partial credit.
- Declares 320 ms of right context per layer instead of summing across layers, and ships an export
  whose stacked lookahead is 640 ms while every individual layer still reads as local.
- Computes log-mel features with librosa at a 10 ms shift during training and reimplements them at
  12.5 ms inside the export; nothing raises, and graded WER roughly doubles.
- Consumes precomputed features, so the front end sits outside the export and the operator
  envelope refuses it.
- Spends budget on decode sophistication the contract forbids.
- Removes blanks before merging repeats, silently collapsing every doubled letter.
- Aliases the CTC blank onto an index the private permutation assigns to a real character.
- Ships a model still in the alignment plateau, whose greedy path is all blank: every hypothesis
  empty, WER exactly 1.000 on both shards.
- Meets the 30 MiB ceiling by pruning after training without re-estimating normalization
  statistics, which runs clean and costs WER.
- Drives LibriSpeech WER to target while Common Voice stagnates, so the product keeps reward
  near 0.1.
