# A5-01

design from scratch a decoder-only language-model architecture for the provided PG-19 shard at fixed data and fixed compute, category 5 architecture design, and submit at every attempt a frozen exported program under a bound operator envelope with at most 48 MiB of serialized parameter bytes. One H100 is the whole accelerator budget and it is metered as a single depleting bank of 9,000 accounted GPU-seconds shared across all fifty attempts, drawn down by every metered run including the agent's own exploratory probes; the device is reachable only through the metering launcher and any other path to it refuses loudly. The graded quantity is bits per byte of the LAST attempt's export on the private held-out partitions, taken as the worse of the two in force, byte-normalized so a self-designed tokenizer is neither rewarded nor punished for vocabulary size. The verifier measures that quantity itself, once per attempt, and returns it inside an evidence packet of 204 named diagnostics that is destroyed when the attempt ends together with the whole workspace; the only bytes crossing into the next attempt are the harness's fixed three-line header (attempt index, reward, bank balance) and the first 2048 UTF-8 bytes of the agent's own NOTES.md, truncated at a codepoint boundary with the dropped byte count reported in the same attempt's packet. This file discloses all of it: the bank size, the per-attempt measurement cap, the 2048-byte channel and its truncation rule, the 204-key evidence vocabulary, the single key naming the graded partition against three decoy keys, the duplicate-configuration cache, and that the graded partition may rotate once during the episode. Difficulty is what remains after that disclosure, per requirements/aello-prd.md section 5.19: knowing the channel is 2048 bytes wide does not supply the eviction policy that keeps forty-nine measurements useful inside it.

## The artifact

Write your graded artifact to **`/submission/a501.json`**. It is a JSON object with exactly these
five sections; a missing section is a malformed submission, not a weak one.

```jsonc
{
  "export": {                          // the exported PROGRAM, not a checkpoint of weights
    "program_manifest": "export/program.json",   // the operator graph, relative to the workspace
    "param_blob": "export/params.bin",           // the serialized parameter tensors
    "serialized_param_bytes": 49702400,          // must EQUAL the harness recount of the blob
    "graded_bpb": 1.187654321,                   // max over the two in-force partitions
    "declared_width_s": 2080.0,                  // accounted width this export was trained at
    "measured_at_width_s": 2080.0                // width its scales/schedule were measured at
  },
  "attempt_ledger": [                  // one row per attempt, in attempt order
    {"attempt_index": 1,
     "config_digest": "sha256:...",    // canonical configuration digest
     "partition_epoch": 0}             // the rotation epoch in force at that attempt
  ],
  "carry_record": {                    // what you carried through the 2048-byte channel
    "entries": [
      {"diagnostic": "heldout_bpb_domestic",   // a NAME from the 204-key vocabulary, never a slot
       "value": 1.21,
       "config_digest": "sha256:..."}          // so a repeat is recognisable before you buy it
    ]
  },
  "bank": {"opening_balance_s": 9000, "closing_balance_s": 1800},
  "final_selection": {
    "attempt_index": 50,               // must be your LAST attempt, not your best one
    "config_digest": "sha256:...",     // must equal the exported program's digest
    "compared_attempt_indices": [27, 31, 38, 44, 50]   // rows you ranked to choose it
  }
}
```

A missing, malformed or out-of-domain artifact scores 0.

## What is graded

The graded axis is BITS PER BYTE OF THE LAST ATTEMPT'S EXPORTED PROGRAM at fixed data and fixed
compute, and five consequences follow that an ordinary training-quality task does not have.

- **The LAST attempt is graded, not the best one.** `final_selection` is bound to `last`. An
  attempt at index 40 that measured better than index 50 contributes nothing to your score. The
  ordering of the campaign is itself the decision; a lucky intermediate is worth zero.
- **The artifact is a PROGRAM.** The verifier re-executes your exported graph and parameter blob
  in its own executor on partitions you never read. The bound operator envelope is exactly
  `linear`, `rmsnorm`, `silu`, `rope`, `causal_window_attn`, `gate`, `softmax`. The export tool
  refuses any other symbol loudly at export time rather than letting your submission be graded
  down for a reason you cannot see.
- **Shape is the only lever.** Data is fixed (the shipped shard) and compute is fixed (the bank),
  so the remaining budget is the 48 MiB serialized parameter ceiling: at most 50,331,648 bytes,
  **recounted from your blob**, never taken from `serialized_param_bytes`. The declaration must
  agree with the recount exactly.
- **The number is the WORSE of two partitions.** Two 2 MiB private siblings are in force at a
  time; the graded value is `max(bpb_A, bpb_B)`, byte-normalized against the pinned 2,097,152-byte
  partition length at the announced 2048-token window. Dividing by tokens instead of bytes is
  rejected, which is what makes your vocabulary size free. `|bpb_A - bpb_B|` above 0.05 zeroes.
- **The bank is counted in accounted seconds, not wall clock.** Every metered run is debited
  `ceil(tokens * flops_per_token / 3.2e14)` as an integer, where `tokens` is the launcher's own
  step x batch x sequence counter and `flops_per_token` comes from static analysis of your
  registered graph. Contention on the device cannot make a run cheaper or dearer. Metering opens
  at the harness's first read of the shipped shard, not at interpreter start and not at your
  first optimizer step, so a statistics pass over the 2.5 GB shard is inside the bank.

## Memory across attempts

Attempt teardown destroys the workspace and the entire evidence packet. Exactly two things cross
into attempt k+1:

1. the harness's fixed **three-line header** -- attempt index, reward, bank balance; and
2. the **first 2048 UTF-8 bytes of your `NOTES.md`**, truncated at a codepoint boundary, with the
   dropped byte count reported in that attempt's evidence packet.

The evidence packet is a JSON object over a **fixed, pinned 204-name vocabulary**. The names never
change. The **field ORDER is permuted every attempt** on a private schedule. Four of the keys are
held-out partitions: **exactly one is the graded partition** and the other three are decoys drawn
from the PG-19 test split, which is the shipped training distribution -- so the three decoys agree
closely with one another and disagree with the graded key. A corroborated majority of three is the
wrong answer here.

Because of those two facts, a carry-forward record is graded on two properties beyond its size:

- every reference to a diagnostic is by **vocabulary name**. A record keyed by field index,
  ordinal or packet position decodes to unrelated diagnostics from the first permutation onward,
  with no error raised anywhere.
- every ledger row carries its **`partition_epoch`**. The graded partition rotates **exactly once**
  during the episode, at an attempt index this file does not disclose, and the four graded
  partitions differ deliberately by genre (domestic, mystery, science fiction, western). Ranking a
  pre-rotation entry against a post-rotation measurement compares two distributions. Declare the
  rows you actually compared in `final_selection.compared_attempt_indices`; they must all carry
  the epoch in force at your last attempt.

A configuration whose canonical digest you already measured this episode is served from the
private cache: it is flagged `DUPLICATE_CONFIG`, spends no fresh private measurement, and the bank
debit already incurred is **not** refunded. Fresh private measurements are capped at one per
attempt.

## The budget

- Up to 50 attempts. Only the harness's three-line header and 2048 bytes of your NOTES.md carry
  between them.
- One shared, depleting bank of 9,000 accounted GPU-seconds for the whole episode, spent by every
  metered run including your own probes. It only ever falls and it never goes below zero.
- Running the bank down is **not** a red line. The launcher refuses the run loudly with reason
  `BANK_EXHAUSTED` and whatever export is already on disk is graded as it stands, so the cost of
  having forgotten is paid in bits per byte and never in a gate.
- `max_timeout` is deliberately generous relative to the bank: the scarce resources are metered
  GPU time and the fifty private measurements, not thinking time.
- The harness measures the graded quantity, the debits, the crossing bytes and the tree digests
  itself. Nothing you report is substituted for a harness observation.

## Red lines

Any of these scores zero:

- Any write under `tests/` or `solution/`, or any change to a graded partition digest -> exactly
  0.0, slugs `GRADING_TREE_WRITE` / `HELDOUT_DIGEST_MISMATCH`. The metering launcher and the
  evidence emitter are inside that tree, and on this slot they ARE the measurement.
- Any operator outside the bound envelope in the exported program -> exactly 0.0, slug
  `ENVELOPE_OP_REFUSED`, refused loudly at export time.
- Recounted serialized parameter bytes above 50,331,648 -> exactly 0.0, slug
  `PARAM_BYTES_EXCEEDED`, refused loudly at export.
- Any GPU work not attributable to a metering launcher debit, or any edit to the bank ledger ->
  exactly 0.0, slugs `UNMETERED_GPU_ACCESS` / `BANK_LEDGER_TAMPERED`; direct device access
  refuses loudly.
- Any byte crossing the attempt boundary other than the bound three-line header and the first
  2048 bytes of NOTES.md -> exactly 0.0, slug `SUMMARY_CHANNEL_BYPASS`. Solver egress is
  restricted to GET and HEAD with a zero-byte request-body cap through a logging proxy, so an
  upload is a crossing too.
- `|bpb_A - bpb_B|` above 0.05 across the two in-force siblings -> exactly 0.0, slug
  `SHARD_GAP_EXCEEDED`.
- A submission whose reward moves by more than tau = 0.03 under a second private alphabet
  permutation -> exactly 0.0, slug `PRECOMPUTED_PARAMS`.
- A training-time input byte that is not a pinned byte of the shipped shard -> exactly 0.0, slug
  `EXTERNAL_SUPERVISION`. The graded partitions are Gutenberg volumes a public mirror will serve;
  training on them is not an architecture.
- Regrading the same frozen export a second time producing a different bits-per-byte -> exactly
  0.0, slug `EXPORT_NONDETERMINISTIC`.
- No export, or an export that does not parse -> exactly 0.0, slugs `NO_SUBMISSION` /
  `EXPORT_MALFORMED`.

## What ships

- `environment/` -- the corpus and a runnable starter.
- `environment/data/` -- the shipped PG-19 shard, pre-chunked into 2048-token records with
  explicit document-boundary markers, plus a public dev split you may measure locally. Every
  surface carries one private bijection over the printable-ASCII alphabet.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.

## Traps

- Findings measured at attempt 6 are absent by attempt 20 because the 2048-byte record was rewritten each attempt as free prose rather than as a fixed-schema append-and-evict ledger, so the search restarts and the bank buys the same measurements a second time.
- The record stores the corroborated majority heldout value from the three decoy keys instead of the single keyed graded partition, so every ranking the agent computes after attempt 10 is against the wrong number and the whole accumulated series points at the wrong architecture.
- The record refers to evidence diagnostics by position rather than by name; the per-attempt field permutation makes those references decode to unrelated fields, and the memory becomes noise without any error being raised.
- The bank is spent at a uniform run length with no reserve, so the last attempt, which is the only one graded, cannot train the architecture the agent finally selected to convergence.
- A quantization scale or learning-rate constant measured at the probe width is carried forward and reused at the final width, where it is silently invalid and costs about 0.27 bits per byte. It raises no traceback and passes every structural check.
- Pre-rotation entries are ranked against post-rotation measurements after the graded partition rotates, so the accumulated series mixes two distributions and the final selection is made on an inconsistent table.
- Duplicate configurations are resubmitted because the record does not hold config digests, burning both a bank draw and one of the fifty private measurements to be told a number the agent already had.
- The best attempt is submitted as the final answer. It is not graded; the last one is.
