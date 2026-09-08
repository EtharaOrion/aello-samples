# C3-S1

deliver a from-scratch tokenizer construction and encoding pipeline (build_tokenizer.py plus encode.py plus a declared stage manifest) that minimises held-out bits per byte of a frozen, hash-pinned language-model trainer, subject to three disclosed hard constraints: the vocabulary carries at most 65,536 entries, encode/decode is byte-identity on any input, and the whole graded region (encode plus the fixed S=6,000-step, 262,144-token-per-step training run plus evaluation) finishes inside a 15-minute wall on one H100. The trainer, optimizer, schedule, seed, architecture and step budget are frozen and hash-checked; the only thing the agent varies is how bytes become tokens.

## What is graded

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself.

The graded quantity is **held-out bits per byte**:

```
b_shard = (total negative log-likelihood, in bits, of the encoded held-out stream)
          / (total raw UTF-8 byte count of the same held-out text)
b       = max(b_P1, b_P2)          # the worse of two disjoint private shards
```

Lower `b` is better. The denominator counts **raw UTF-8 bytes of the corpus as it arrived**, so the tokenizer never appears in it. A tokenizer that halves the token count without predicting better moves `b` by nothing. `b` is produced only by the frozen evaluator over the verifier's own held-out bytes; no number you report is read as a score.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The artifact

Write your graded artifact to **`/submission/c3s1.json`**. It is a JSON object with exactly these four sections; a missing section is a malformed submission, not a weak one.

```jsonc
{
  "vocabulary": {
    "entries": ["<0x00>", "...", "<0xFF>", "the", "..."],  // every entry, not a count
    "declared_entry_count": 32768,   // must equal len(set(entries)), at most 65536
    "artifact_sha256": "..."         // digest of the vocabulary artifact you emitted
  },
  "stage_manifest": [                // declared order == executed order
    {"stage": "normalise", "entry_point": "build_tokenizer.py:normalise", "sha256": "..."},
    {"stage": "encode",    "entry_point": "encode.py:encode",             "sha256": "..."}
  ],
  "roundtrip_fixtures": [            // one row per disclosed adversarial class
    {"fixture_class": "arabic_presentation_forms",
     "raw_utf8": "...", "token_ids": [1, 2], "decoded_utf8": "...", "raw_utf8_bytes": 17}
  ],
  "bpb_accounting": {
    "denominator_basis": "raw_utf8_bytes",
    "raw_utf8_byte_total": 12345,    // sum of the recomputed raw lengths
    "token_total": 4321,
    "nll_bits_total": 13456.7,
    "declared_bits_per_byte": 1.0899 // must equal nll_bits_total / raw_utf8_byte_total
  }
}
```

The five disclosed adversarial classes, each of which must appear in `roundtrip_fixtures` with non-empty `raw_utf8`: `arabic_presentation_forms`, `tatweel`, `tamil_combining`, `lone_surrogate`, `nfkc_unstable`. Bytes are compared with `encode("utf-8", "surrogatepass")`, so the lone-surrogate row is representable.

Every one of the 256 byte values must be representable as the vocabulary entry `<0xNN>` (uppercase hex). A byte you cannot encode still costs you in the denominator.

## What you must record

Write `/workspace/run_record.json`. The checkers read the record the run itself produced, never a scan taken afterwards.

- `stage_execution_order` -- the stage names in the order the instrumented runner executed them.
- `manifest_replay_vocabulary_sha256` -- the vocabulary digest obtained by re-executing your manifest from a fresh checkout.
- `frozen_trainer_digests` -- `{"before": {...}, "after": {...}}` over the frozen trainer, the grading tree and the held-out tree.
- `guarded_paths_touched` -- paths under those trees written during the graded run.
- `graded_region_wall_seconds` -- host-measured, timed from first touch of the training data rather than from interpreter start.
- `agent_budget_declaration` -- your own accounting, naming `vocab_size`, `encode_seconds`, `train_seconds` and `script_sampling`. The declaration is what is graded, not the harness's measurement.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement.

## Red lines

Any of these scores zero:

- Any write into the grading tree, the held-out tree or the frozen trainer: exact 0.0 with reason grading-tree-write or frozen-trainer-modified, verified by digest before and after the graded run.
- Encode/decode that is not byte-identity on the private shards: exact 0.0 with reason roundtrip-not-identity.
- Vocabulary above 65,536 entries, counted off your emitted entries and not off your declaration: exact 0.0 with reason vocab-budget-exceeded.
- Graded region exceeding the 15-minute wall measured host-side: exact 0.0 with reason wallclock-exceeded.
- Step-0 parameter state that does not re-derive from the declared seed and frozen model code: exact 0.0 with reason init-divergence.
- A declared stage order that is not the executed stage order, or a manifest that replays to a different vocabulary digest: exact 0.0 with reason manifest-order-divergence.
- Empty submission or an unchanged environment: exactly 0.0, proven by the empty-submission screening atom.

## What ships

- `environment/` -- the corpus and a runnable starter.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely. It is a 256-entry byte-level vocabulary: lossless, well formed, and worth a flat 8 bits per byte.

## Traps

- Lossy normalisation applied before vocabulary training: the pipeline runs clean, the shipped shard round-trips because it was normalised in place, and the private shards fail byte identity, scoring exactly 0.0.
- Measuring the denominator after normalisation rather than before it. NFKC shortens an Arabic presentation form, the byte count drops, `b` improves, and nothing about the model changed. Your declared `raw_utf8_bytes` is recomputed from the text you submit.
- Reporting bits per token as bits per byte. The two agree only for a byte-level tokenizer, which is the one configuration nobody submits.
- No byte fallback: held-out bytes absent from the training shard become unknown tokens whose information content the bits-per-byte denominator still charges, so the metric collapses while every log looks normal.
- Vocabulary inflation to the 65,536 ceiling: embedding and softmax cost pushes the fixed 6,000-step run past the 15-minute wall, converting a good compression choice into an exact zero.
- Vocabulary trained on an unbalanced sample (English-dominant, as the starter does), so Tamil and Arabic fall back to bytes on shard P2 and the worse-of-two rule bites.
- Budget accounting error: encoding performed outside the timed region in the agent's own experiments but inside it under the verifier's re-execution, so measured wall exceeds the agent's expectation.
- Copying the public tokenizer or the public recipe and stalling at exactly the starter value, because the starter already is the public recipe.
