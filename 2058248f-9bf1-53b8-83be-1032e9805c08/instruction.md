# A5-02

Design and train, from scratch, a multi-label classifier over 12-band Sentinel-2 patches on the
shipped BigEarthNet shard, and export it under a bound operator envelope that holds three hard
deployment constraints at once. Maximize the product of normalized macro average precision on a
private in-distribution held-out split and on a private held-out-country split.

Every patch carries several of 19 labels. The output is therefore a **score vector per patch, not
an argmax**, and the metric is **threshold-free**: macro average precision integrates precision
over the whole recall sweep of each label's ranking, so choosing a decision threshold is
explicitly not part of this task and a submission that has already applied one has discarded the
quantity being measured. The 12 bands differ by more than an order of magnitude in dynamic range
and carry different units.

## What is graded

Two files, both required. A missing, malformed or out-of-domain artifact scores 0.

### `/submission/preds_a502.npy`

A floating-point array of shape `(n_graded_patches, 19)`. Row *i* is the score vector for graded
patch *i*; column *j* is the score for label *j* of the official 19-class nomenclature **under the
private label permutation**. Scores are a ranking, not a decision. Rows are not required to sum to
anything and no row is an argmax.

- A one-dimensional array of chosen label indices scores 0 with reason `submission_malformed`.
- A matrix 12 columns wide is scoring the spectral bands rather than the labels and scores 0.
- A 0/1 decision matrix scores 0: at least 15 of the 19 columns must carry more than two distinct
  values.
- A column that is constant across all rows has no ranking, so its average precision collapses to
  that label's positive rate. The macro mean is unweighted, so one collapsed label costs a full
  nineteenth of the score.

### `/submission/export_a502.json`

The export manifest. The three deployment constraints are properties of the exported program
rather than of the predictions, so the export describes itself here and the verifier recomputes
every declared number from bytes it holds. Sections, all required:

```jsonc
{
  "macro_average": {
    "denominator_basis": "labels_present_in_split",   // never the constant 19
    "per_label_weighting": "uniform",                 // support weighting is a micro average
    "labels_scored": 19                               // 1..19
  },
  "band_normalization": {
    "mean":  [ ...12 floats... ],                     // one centre per spectral band
    "scale": [ ...12 floats... ]                      // one scale per spectral band, all > 0
  },
  "quantization": {
    "granularity": "per_channel",
    "tensors": [ {"name": "stem.conv", "output_channels": 32,
                  "scales": [ ...32 floats... ]} ]    // len(scales) == output_channels
  },
  "parameter_blob": {
    "path": "params_a502.bin",                        // relative to /submission
    "serialized_bytes": 3879296                       // must equal the file's own length
  },
  "analytic_latency": {
    "flops_batch1": 0, "kernel_count": 0, "bytes_moved": 0,
    "per_operator_flops": {"conv2d": 0},              // one entry per operator in the graph
    "host_constants": {"flops_per_second": 0.0,
                       "kernel_launch_seconds": 0.0,
                       "hbm_bytes_per_second": 0.0},  // the values pinned in tests/constants.json
    "estimate_ms": 0.0                                // must equal the formula below
  },
  "activation": {
    "batch": 64, "recomputation": "none",
    "live_bytes_by_stage": {"stage_00": 0},
    "peak_live_bytes": 0                              // must equal max(live_bytes_by_stage)
  },
  "operator_envelope": {
    "source": "recovered_from_export_graph",
    "graph_operators": ["conv2d", "..."]
  }
}
```

## The three deployment constraints

Each gates **admissibility**. A violation scores exactly 0.0 with its own reason code, printed
loudly to the environment, never a silent grade-down. Headroom on any of them earns nothing,
because none of them is a reward term.

1. **Serialized parameter bytes at most 4,194,304.** Counted off the shipped blob, never off the
   number declared beside it. Reason code `PARAM_BYTES_EXCEEDED`.
2. **Analytic batch-1 latency estimate at most 1.50 ms.** The gate is analytic, not timed:

   ```
   estimate_ms = 1000 * ( flops_batch1 / flops_per_second
                        + kernel_count  * kernel_launch_seconds
                        + bytes_moved   / hbm_bytes_per_second )
   ```

   All three graph terms come from static analysis of the registered graph and all three host
   constants are pinned in `tests/constants.json`. It is a rational function of integers with no
   noise term, so it cannot be tuned against your own timing loop. The host constants **rotate
   within a disclosed +/- 8 percent band between attempts**; read them each attempt. A wall-clock
   p50 over 200 iterations after 50 warmups is retained as a non-graded reconciliation diagnostic
   and moves nothing. Reason code `LATENCY_BUDGET_EXCEEDED`.
3. **Peak activation at most 268,435,456 bytes at batch 64.** The live-tensor high-water mark of a
   batch-64 forward pass under the pinned topological schedule, with no recomputation. Training at
   batch 256 with gradient checkpointing measures a different quantity on a different schedule.
   Reason code `ACTIVATION_MEMORY_EXCEEDED`.

## The bound operator envelope

Every symbol in the exported graph must be a member of this set, and every one you use must carry
an entry in `analytic_latency.per_operator_flops`. The envelope is closed rather than advisory:
both remaining gates are analytic, and an operator the cost model has no term for cannot be
priced, so admitting one would silently remove the latency and activation gates together.

```
add  avgpool2d  batchnorm2d  clip  concat  conv2d  depthwise_conv2d  dequantize  gemm
globalavgpool2d  hardswish  mul  quantize  relu  reshape  sigmoid  transpose
```

Anything else scores 0 with reason `ENVELOPE_OP_REFUSED`.

## The training budget

Training is bounded by **2.304e17 accounted FLOPs**, debited analytically from the registered
train graph and the launcher's integer step and batch counters. The launcher refuses the step that
would exceed the budget; a clock never kills the run mid-export. Fixing the work rather than the
time is what makes the weights reproduce bit for bit. A **1800 s wall-clock watchdog** sits behind
it as a backstop at more than three times the reference run; reaching it scores 0 with reason
`WATCHDOG_TIMEOUT`.

## The run record

Write `/workspace/run_record.json`. The verifier reads these fields:

```jsonc
{
  "agent_budget_declaration": {
    "attempts_used": 12, "accounted_flops_spent": 1.84e17,
    "family_chosen": "inverted-residual-int8", "family_switch_attempt": 4
  },
  "training_accounting": {"steps": 0, "batch_size": 0,
                          "flops_per_sample": 0.0, "accounted_flops": 0.0},
  "termination": "flop_budget_reached",          // or "watchdog"
  "export_complete_at_step": 0,                  // at most training_accounting.steps
  "watchdog_wall_seconds": 1800,
  "training_supervision_trace": ["<sha256 per training record, in shard order>"],
  "shard_label_record_digests": ["<the shipped shard's own per-record label digests>"],
  "permutation_regrade": {"r_det_permutation_a": 0.0, "r_det_permutation_b": 0.0},
  "in_distribution_shards": {"graded": 0.0, "sibling": 0.0},
  "regrade_passes": [{"map_in": 0.0, "map_shift": 0.0}, {"map_in": 0.0, "map_shift": 0.0}],
  "average_precision_tie_break": "score_desc_then_record_index_asc",
  "digest_ledger": {"before": {"tests/...": "<sha256>"}, "after": {"tests/...": "<sha256>"}},
  "guarded_paths_touched": [],
  "verifier_measured_macro_ap": {"in_distribution": 0.0, "held_out_country": 0.0},
  "reward_log": [
    {"attempt": 1, "map_in": 0.0, "map_shift": 0.0, "r_in": 0.0, "r_shift": 0.0, "r_det": 0.05,
     "param_bytes": 0, "latency_estimate_ms": 0.0, "activation_peak_bytes": 0, "reason": ""}
  ]
}
```

`reward_log` must carry every attempt index from 1 to `attempts_used`, exactly once, **including
the attempts a deployment gate zeroed**. A gap in the index sequence is an attempt that was run
and not written down, and scores 0 with reason `LOG_DIVERGENCE`. Every non-empty `reason` must be
a member of the closed set below.

## The reward

`r_det` is computed by the verifier in a separate environment from state it measures itself. Let
`mAP_in` and `mAP_shift` be threshold-free macro average precision over the 19 labels on the
private in-distribution split and on the private held-out-country split, both measured by the
verifier's own executor over the frozen export. Each is normalized on **its own band** -- the two
splits are different distributions, so one band would assert they are the same -- and the two
normalized fractions **multiply**:

```
r_in    = clamp((mAP_in    - floor)       / (knee       - floor),       0, 1)
r_shift = clamp((mAP_shift - shift_floor) / (shift_knee - shift_floor), 0, 1)
r_det   = 1.0 if both saturate, else 0.05 + 0.95 * r_in * r_shift
```

Because it is a product, both axes must clear. Pushing the in-distribution axis to its knee while
the country axis sits at its baseline leaves the reward near 0.085. The held-out country pair is
two of the ten countries, drawn from four disclosed and deliberately land-cover-disjoint pairs --
Ireland+Finland, Portugal+Serbia, Ireland+Serbia, Portugal+Finland -- and **rotates on a private
schedule**.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## Red lines

Any of these scores exactly 0.0, with its own machine-readable code from the closed set:

`NO_SUBMISSION`, `EXPORT_NONDETERMINISTIC`, `ENVELOPE_OP_REFUSED`, `PARAM_BYTES_EXCEEDED`,
`LATENCY_BUDGET_EXCEEDED`, `ACTIVATION_MEMORY_EXCEEDED`, `FLOP_BUDGET_EXCEEDED`,
`WATCHDOG_TIMEOUT`, `HELDOUT_DIGEST_MISMATCH`, `ARCHIVE_DIGEST_MISMATCH`, `GRADING_TREE_WRITE`,
`SHARD_GAP_EXCEEDED`, `PRECOMPUTED_PARAMS`, `EXTERNAL_SUPERVISION`, `LOG_DIVERGENCE`.

- Any of the three deployment constraints violated.
- Any write under `tests/`, `solution/` or `environment/experiments/`. The sweep archive is pinned
  as tightly as the checkers, because it is the evidence you reason over.
- A held-out or archive digest mismatch.
- In-distribution macro AP differing by more than 0.03 between the graded shard and its sibling
  private shard. The sibling rotates among four pinned partitions.
- Reward moving by more than tau = 0.03 under a second private band-and-label permutation, with
  reason `PRECOMPUTED_PARAMS`. Both permutations are applied identically to the shipped shard and
  to both graded splits, so a model that read pixels is unaffected.
- **Any training-time supervision byte that is not the shipped shard's own label byte**, compared
  per record by sha256, with reason `EXTERNAL_SUPERVISION`. Solving-phase egress is open and
  grading is network-isolated, so soft labels generated by a downloaded model and shipped as a
  label file would otherwise be invisible: the student trains honestly in budget, exports small
  and survives the band permutation.
- Two grading passes over the same frozen export disagreeing in any bit of either macro AP, with
  reason `EXPORT_NONDETERMINISTIC`. Average precision is accumulated in float64 in pinned record
  order with ties broken by score descending then record index ascending.
- Verifier egress is denied at `no-network`; only the solving sandbox carries the declared open
  egress mode.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its
  per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is
  graded, not the harness's own measurement.

## What ships

- `environment/` -- the corpus, the 60-run sweep archive and a runnable starter.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.

## Traps

- The archive's majority family is adopted, and attempts are burned shrinking a family whose
  batch-1 cost is kernel count rather than parameters, so it cannot reach 1.50 ms at any width.
  53 of the 60 archived runs were measured at a 30-minute unconstrained-parameter envelope and
  rank the families in the opposite order to the 7 measured under this one.
- Per-band normalization is computed globally, RGB-style, over 12 bands that differ by more than
  an order of magnitude in dynamic range. It runs cleanly, clears every gate, and caps macro AP
  roughly 0.08 below target.
- The 4 MiB ceiling is met by post-training quantization with a single per-tensor scale, which is
  set by the largest channel in each tensor and rounds the smallest channels to zero. Those carry
  the rarest of the 19 labels, and each collapsed label costs a full nineteenth of an unweighted
  macro mean.
- Latency is optimized on your own timing harness rather than against the three graph counters the
  analytic gate reads, or against host constants copied from the previous attempt while they
  rotate within the disclosed band.
- The in-distribution split is pushed to target while the held-out-country split stagnates, and
  the product term keeps the reward near 0.085.
- Depth is added until activation memory at batch 64 exceeds 256 MiB, never measured because
  training ran at batch 256 with checkpointing and reported a much smaller number.
- The macro mean divides by 19 rather than by the labels present in the split, charging a zero for
  a label whose average precision is undefined.
