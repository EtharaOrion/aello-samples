# AELLO-C8-S11 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

minimise measured kernel wall-clock for a depthwise-separable block while every output stays inside a bound numerical fidelity envelope. The graded artifact is one measurement record per graded block shape, written to /submission/kernel_report.json. The graded quantity is the geometric-mean speedup of the harness's own per-shape kernel wall-clock over the vendored reference kernel's, across the three pinned shapes, forced to zero when any shape leaves the envelope.

## The single most important insight

Nothing here grades an array's contents. The graded object is a kernel, the graded quantity is how long it takes, and a duration is only as trustworthy as the method that produced it -- so the checks are about measurement methodology first and speed second. Warmup discarded, a repetition count met, a median rather than a best-of-n, a clock that waits for the device instead of for the launch call to return. Get any of those wrong and the number in the report describes something other than the kernel. And speed is gated, not traded: the fidelity envelope is checked before the ramp is read, so a kernel that wins wall-clock by leaving the envelope on one of three shapes scores zero rather than scoring well on two.

## The ideal solve, step by step

1. **Time the two stages apart before optimising either** A depthwise-separable block is two kernels with opposite cost structure: the 3x3 depthwise stage does nine multiply-accumulates per loaded activation and is memory-bound, the 1x1 pointwise stage does one per channel pair over a much larger reduction and is compute-bound. The block's total tells you nothing about which one you are paying for, and at 8x8x512 the answer is not the same as at 32x32x128. Lever L4, the fusion boundary, is the choice this measurement makes legible; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

2. **Fix the measurement before trusting any speedup** Warmup that ends before the timed window opens, a repetition count that makes the median stable, a device-synchronised clock, and the pinned traces rather than a convenient draw. A speedup measured with a bare host timer around an asynchronous launch is a measurement of the launch queue, and it improves whenever the kernel gets slower enough to hide behind the next enqueue. Every one of these is a knockout check because none of them costs anything to get right and each of them silently invalidates every number downstream.

3. **Tile against the activation working set, not against the register file** Lever L1. The three graded shapes differ by an order of magnitude in arithmetic intensity and by 16x in spatial extent at the same batch, so one tiling does not sit at the occupancy sweet spot for all three; a tile chosen for 32x32x128 spills at 8x8x512. Its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

4. **Stage through shared memory and widen the loads** Levers L2 and L3. The depthwise stage re-reads a 3x3 neighbourhood per output and the pointwise stage streams the whole channel axis; staging policy and vectorised load width are the two knobs that decide whether those reads coalesce. Their worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

5. **Spend precision only where the envelope has room** Lever L5. Accumulating in a narrower type is the largest single wall-clock lever available and it is the one the fidelity envelope bounds -- fp32 accumulation under bf16 or tf32 inputs stays inside 2e-3 absolute and 5e-3 relative on these traces, fp16 accumulation over a 512-wide reduction does not. Its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

## Traps that catch agents that are not thinking carefully

- Wins wall-clock by leaving the fidelity envelope and the gate zeroes the attempt -- the envelope is checked before the ramp is read, so two fast conforming shapes do not pay for one that drifted.
- Reports best-of-n. The device delivers that latency once, under a clock-boost state the harness's own repeated measurement never reproduces, and the harness is what the score is computed from.
- Wraps a host timer around an asynchronous launch and reports microseconds for milliseconds of work.
- Tiles for the wrong occupancy and the working set spills every iteration -- and at 8x8x512 it spills for a tile that was optimal at 32x32x128.
- Times the block on synthetic zeros or one re-used draw, so the whole working set stays L2-resident and the measured speedup evaporates on the pinned traces.
- a vendor library kernel would bypass the design entirely Mitigation: the allowlist reaches no package index at solve time and the fidelity gate is checked against the vendored reference outputs

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `measurement_budget_declared` (compiled, weight 3) -- The agent declares the measurement budget it spent, in the terms this task spends it.
2. `latency_report_lands` (compiled, weight 5, knockout) -- The kernel's measurement record lands at the declared submission path and parses as a per-shape record.
3. `every_graded_block_shape_measured` (compiled, weight 5, knockout) -- Every pinned depthwise-separable block shape carries a complete record.
4. `pinned_trace_seeds_are_the_measured_inputs` (compiled, weight 3) -- The kernel was timed on the vendored activation traces at the pinned seeds.
5. `warmup_iterations_excluded_from_the_timed_window` (compiled, weight 5, knockout) -- The timed window begins after the declared warmup.
6. `reported_latency_is_a_median_not_a_best_of_n` (compiled, weight 5, knockout) -- The reported latency is a median over a declared repetition count.
7. `device_synchronised_clock_not_async_launch_return` (compiled, weight 5, knockout) -- The clock source measures the kernel, not the launch call.
8. `depthwise_and_pointwise_timed_as_separate_stages` (compiled, weight 3) -- Each record separates the depthwise stage from the pointwise stage, or declares the fusion.
9. `fidelity_envelope_holds_on_every_graded_shape` (compiled, weight 5, knockout) -- Every graded shape's output stays inside the published numerical fidelity envelope.
10. `replay_reproduces_outputs_but_not_timings` (compiled, weight 3) -- A second run reproduces the outputs exactly and the timings only approximately.
11. `vendored_traces_and_checker_tree_unwritten` (compiled, weight 5, knockout) -- The graded run writes nothing into the checker tree or the recorded traces.
12. `envelope_gated_speedup_ramp` (compiled, weight 10, knockout) -- The measured speedup over the vendored reference kernel clears the published bar through the ramp.

Rejected routes, each bound to the control that measures it wrong:

- a vendor library kernel would bypass the design entirely -- controlled by: the allowlist reaches no package index at solve time and the fidelity gate is checked against the vendored reference outputs

Deliverable manifest: /submission/kernel_report.json (json), one measurement record per graded depthwise-separable block shape, keyed by shape id under `shapes`.

Control obligations recorded in seed/contract.yaml:

- `CTL-NOOP` targets the shipped starter kernel verbatim; recorded score None.
- `CTL-UNFAITHFUL` targets a fast kernel outside the fidelity envelope; must score 0; recorded score None.
- `CTL-STALE` targets a kernel tuned for a different block shape; recorded score None.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-741a9ceccad662e7022448e7f797fc44
  slot1: AELLO-CANARY-521001e5368bb237f8f227a575ae6e33
  slot2: AELLO-CANARY-b2e27e5644d2d284c7954bf9baaca2d1
  slot3: AELLO-CANARY-492407b078d8e0821be18962ad1f325f
-->
