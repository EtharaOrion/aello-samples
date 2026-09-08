# AELLO-C8-S11

minimise measured kernel wall-clock for a depthwise-separable block while every output stays inside a bound numerical fidelity envelope.

## What is graded

The graded object is a **kernel**, not a prediction. Nothing you submit is scored on its contents; what is scored is how long your depthwise-separable block takes on the vendored activation traces, and the answer is only worth anything if the measurement that produced it means something.

Write your measurement record to **`/submission/kernel_report.json`**. It carries one entry per graded block shape under a `shapes` object, keyed by shape id. A missing, malformed or out-of-domain record scores 0.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. The graded quantity is the **geometric-mean speedup of the harness's own per-shape kernel wall-clock over the vendored reference kernel's**, taken across the three graded shapes. Two consequences worth reading twice:

- The **harness's** measurement is the one that scores. Your report is graded on whether it was honestly obtained, and the score is computed from the harness re-measurement of your kernel. Reporting a timing the harness did not measure is red line 2.
- The fidelity envelope **gates** the speed rather than trading against it. One shape outside the envelope makes the speedup exactly `0.0` -- not two thirds of it. A fast wrong kernel is worth nothing.

`r_det` is zero on any red line and on any knockout failure, before any speed is considered.

## The block and the three graded shapes

A depthwise-separable block is two stages with opposite cost structure: a 3x3 stride-1 **depthwise** convolution, which does nine multiply-accumulates per loaded activation and is memory-bound, followed by a 1x1 **pointwise** projection, which does one per channel pair over the whole channel axis and is compute-bound. Optimising the block means knowing which of the two you are paying for, and the answer changes across these three shapes:

| shape id | batch | channels | H x W | depthwise | pointwise out | trace seed |
| --- | --- | --- | --- | --- | --- | --- |
| `dws32_c128` | 256 | 128 | 32 x 32 | 3x3 s1 | 128 | 20260814 |
| `dws16_c256` | 256 | 256 | 16 x 16 | 3x3 s1 | 256 | 20260815 |
| `dws8_c512` | 256 | 512 | 8 x 8 | 3x3 s1 | 512 | 20260816 |

They are the three depthwise-separable stages of the C8-S1 reference at CIFAR-100's 32x32 input, pinned before any measurement. The batch is held constant so the working set, and not the batch, is what changes between them. A tiling that sits at the occupancy sweet spot for `dws32_c128` will spill at `dws8_c512`.

## What ships

- `environment/data/c8s1_activation_traces.npz` -- the vendored activation traces, one per graded shape, captured from the C8-S1 CIFAR-100 reference at the pinned seed in the table above. These are the inputs you time on. They are the inputs the harness times on.
- `environment/data/graded_block_shapes.json` -- the shape table above, machine-readable.
- `environment/starter/starter.py` -- a correct, unoptimised baseline you may replace entirely. It is also the reference implementation you check your own fidelity against.

The vendored fp32 **reference outputs** and the reference kernel's **measured latency** ship only under `tests/heldout/` and the graded run must not touch them. That split is deliberate and it is not about secrecy: if the correct outputs were visible, the fastest conforming submission would be a table lookup with no kernel in it, and the envelope would gate nothing. Check your fidelity by running the shipped baseline on the same trace -- that is what the vendored reference is.

## Each `shapes` entry

Every graded shape id must appear exactly once, and no other id may appear. Each entry carries every key below:

| key | what it must carry |
| --- | --- |
| `block_ms` | the block's measured wall-clock in milliseconds |
| `depthwise_ms`, `pointwise_ms` | the two stage times, when `fusion_boundary` is `unfused`; they must sum to `block_ms` within 5 percent. A fused entry omits both rather than inventing a split it did not measure |
| `fusion_boundary` | `unfused`, `fused_pointwise_epilogue` or `fully_fused` |
| `warmup_iterations` | at least **25**, and `warmup_excluded` must be `true` |
| `warmup_excluded` | `true` -- the timed window begins after the warmup ends |
| `timed_iterations` | at least **100** |
| `statistic` | `median`, `p50` or `quantile_0.50`. `min`, `minimum`, `best`, `best_of_n` and `fastest` are rejected by name |
| `clock_source` | `cuda_event_elapsed_time`, `device_synchronised_perf_counter` or `cupti_kernel_duration`. A bare host timer around an asynchronous launch (`host_perf_counter_no_sync`, `time_time_around_launch`, `python_wall_clock_async`) is rejected |
| `max_abs_error`, `max_rel_error` | elementwise, against the vendored fp32 reference output on the same trace |
| `within_envelope` | your own verdict. It is recomputed from the two errors above, and a record that contradicts itself fails |
| `trace_seed` | the pinned seed for that shape, from the table |
| `output_sha256` | digest of the block's output for that shape, so a replay can be compared |

## The run record

Write `/workspace/run_record.json` during the graded run. The verifier reads it and the compiled checks grade it, so an absent or incomplete record costs score on its own:

| key | what it must carry |
| --- | --- |
| `trace_seeds` | the seeds actually consumed: exactly `[20260814, 20260815, 20260816]` |
| `paths_written` | every path the graded run wrote |
| `trace_digest_before`, `trace_digest_after` | the digest of the trace archive before your first measurement and after your last. They must be equal |
| `replay` | per shape id, the `output_sha256` and `block_ms` of a second run. The digests must match your report exactly; the milliseconds must not all match it exactly |
| `agent_budget_declaration` | `warmup_iterations`, `timed_iterations`, `autotune_trials`, each a non-negative count |

## Why the methodology is graded and not just the number

Four measurement errors will each make a kernel that changed nothing look faster, and every one of them is a knockout check because none of them costs anything to get right:

- **A cold launch.** The first launches pay JIT compilation, autotune selection and a cold instruction cache. The harness re-measures in steady state, so a number taken before warmup ends does not describe the kernel that gets graded.
- **Best-of-n.** The device delivers that latency once, under a clock-boost and cache state that repeated measurement does not reproduce. A median over at least 100 timed iterations is what the harness will see.
- **A host timer around an asynchronous launch.** The launch returns before the kernel runs, so the timer measures the enqueue and reports microseconds for milliseconds of work. This one flatters a kernel that got *slower*, because slower work hides further behind the next enqueue.
- **A convenient input.** Synthetic zeros, or one small draw re-used across all three shapes, leaves the whole working set L2-resident. That measures cache residency, not your kernel, and the speedup evaporates on the pinned traces.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement. Declare `autotune_trials` honestly: a tiling sweep costs more wall-clock than the kernel it selects, and that is where this task's budget actually goes.

## Red lines

Any of these scores zero:

- Any write into `tests/` or the recorded traces during the graded run. Shrinking a trace's batch or overwriting it with a cheaper draw makes the kernel look faster by changing the measurement rather than the kernel.
- Reporting a timing the harness did not measure. A latency report is the one deliverable here that is *not* byte-reproducible: identical floating-point milliseconds on every shape across two runs is the signature of a copied constant, not of a deterministic measurement.

## Traps

- Wins wall-clock by leaving the fidelity envelope and the gate zeroes the attempt. The envelope is checked before the ramp is read, so two fast conforming shapes do not pay for one that drifted.
- Reports best-of-n, and the harness's own repeated measurement never reproduces it.
- Wraps a host timer around an asynchronous launch and reports microseconds for milliseconds of work.
- Tiles for the wrong occupancy and the working set spills every iteration -- and a tile that was optimal at 32x32x128 spills at 8x8x512.
- Times the block on synthetic zeros or one re-used draw, so everything stays L2-resident and the measured speedup evaporates on the pinned traces.
- Accumulates in fp16 over a 512-wide reduction. It is the largest single wall-clock lever available and it is the one that leaves the envelope; fp32 accumulation under bf16 or tf32 inputs stays inside it.
- a vendor library kernel would bypass the design entirely Mitigation: the allowlist reaches no package index at solve time and the fidelity gate is checked against the vendored reference outputs

## A note on what has not landed yet

The trace corpus has not been vendored into this tree at the time of writing: `environment/data/` and `tests/heldout/` are empty. The paths, shapes, seeds and bounds published above are the layout the corpus must land against, recorded in `solution/grounding.yaml` under `corpus_layout`, and every compiled check that needs a file it cannot find reports that it could not find it rather than passing over its absence.
