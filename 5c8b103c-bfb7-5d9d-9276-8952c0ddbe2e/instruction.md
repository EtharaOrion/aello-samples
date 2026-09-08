# AELLO-C6-S2

Build a pipeline that, in one graded run on one H100, emits for every solution in a private held-out set BOTH a per-step label sequence over `{good, neutral, bad}` AND an integer first-error index, using the exact sentinel `-1` when the solution contains no bad step.

The two outputs are coupled. The first error is the first step you labelled `bad`, so your index must equal the first `bad` position of the very vector you shipped beside it. A submission whose index contradicts its own labels is rejected outright, not graded down.

Score rises as the graded run gets **faster**, once the joint quality gate clears: macro-F1 over the three step classes at or above `T_F1` and exact first-error index match rate at or above `T_LOC`, both 0.55.

## What is graded

Write your graded artifact to **`/submission/c6s2.json`**. A missing, malformed or out-of-domain artifact scores 0.

```json
{
  "ids":     ["sol_00001", "sol_00002"],
  "outputs": [{"labels": ["good", "neutral", "bad", "good"], "first_error_index": 2},
              {"labels": ["good", "good"],                   "first_error_index": -1}]
}
```

- `ids` and `outputs` are the same length and in the same order. Emit for **every** solution in `environment/data/graded_solutions.jsonl` **and** in `environment/data/sibling_solutions.jsonl`. You are not told which shard is graded, and the sibling comparison below needs both.
- `labels[k]` is one of the three strings `"good"`, `"neutral"`, `"bad"` — one per step, in step order. Nothing else is a label: not a score, not a probability, not `null`.
- `first_error_index` is a JSON integer: either a valid index into that same `labels` list, or exactly `-1` when that list holds no `"bad"` entry. `-1` is a class on the localisation axis, matched by the human `-1` and by nothing else. It is never a wildcard and never a way to abstain.
- No output record may carry a positional key (`row`, `row_index`, `position`, `order`, `rank`, `presentation_index`). The graded set is presented to your run in a private per-run permutation, and every graded count is taken in canonical sorted id order.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. It reads no number you report.

## The corpus

- `environment/data/graded_solutions.jsonl` — one record per graded solution: `solution_id`, `problem_id`, `subject`, `level`, `steps` (in the **shipped** byte form), `final_answer`, `gold_answer`.
- `environment/data/sibling_solutions.jsonl` — the second private shard's features, same fields, same stratification.
- `environment/data/phase2_train_labelled.jsonl` and `environment/data/phase2_local_fold.jsonl` — labelled phase-2 material: `solution_id`, `steps`, `labels`, `first_error_index`, `level`. Fit your head on the first and your per-stratum thresholds on the second.
- `environment/data/step_normalisation.md` — the whitespace and LaTeX-spacing canonicalisation under which the graded shard's step boundaries are **re-derived**. Applied privately, disclosed here: a segmentation fitted to the shipped byte form drifts, and your label vector must be exactly as long as the re-segmented step count or the run fails loudly.

The disclosed difficulty strata are `level_1` through `level_5`. The graded shard carries a difficulty-stratum **mixture** that is not the shipped split's, so a threshold fitted to match the aggregate neutral rate will miss inside the levels.

The private human label vectors, the private step-count manifest and the private stratum map live under `tests/heldout/` and the graded run must not read or write them.

## The hard constraints

Any of these makes `r_det` exactly zero, with one named reason, before any quality is considered:

| quantity | bound |
| --- | --- |
| `nb` — largest per-stratum gap between your emitted neutral rate and the human neutral rate | at most **0.03**, inside **every** stratum |
| `pc` — solutions whose emitted index is not the first `bad` of their own vector | exactly **0** |
| `sib` — gap in joint quality (the mean of macro-F1 and the localisation rate) between the graded shard and its sibling | at most **0.04** |
| `ca` — first-error exact match on solutions whose final answer matches gold while a human-labelled bad step is present | at least **0.48** |
| `ls` — first-error exact match on solutions of at least **12** steps whose first error is at index **8** or later | at least **0.44** |
| `t` — harness-measured integer seconds from first data touch to artifact write | at most the slow end of the ramp |

Above those, `r_det` is staged: below the joint quality gate it is `0.10 * min(clamp((f1 - 0.33) / 0.22), clamp((loc - 0.20) / 0.35))`, so abandoning one axis to raise the other buys nothing; above it, it is `0.12 + 0.88 *` the log speed ramp in `t`.

## The run record

Write `/workspace/run_record.json` during the graded run. The verifier reads it and the compiled checks grade it, so an absent or incomplete record costs score on its own:

| key | what it must carry |
| --- | --- |
| `first_data_touch_ns` | harness nanosecond reading at the run's first read of the corpus |
| `artefacts_written_ns` | harness nanosecond reading at the write of the coupled artifact |
| `elapsed_seconds` | the graded integer seconds, `ceil` of the interval between those two readings — it is checked against them |
| `stage_spans` | one `[start_ns, end_ns]` pair per stage, for all six of `segmentation`, `encoding`, `three_class_fitting`, `per_stratum_calibration`, `localization_decode`, `export`. Every span must lie inside the timed interval: calibrating before the timer starts is not a faster pipeline |
| `replay_artifact_sha256` | sha256 of the artifact your own second run under the recorded seed produced. The whole artifact, both outputs together |
| `reward_log` | path of your `reward_log.jsonl`, relative to the workspace |
| `guarded_paths_touched` | every path under `tests/` or the verifier's output directory the run wrote. Empty, or the run scores zero |
| `agent_budget_declaration` | your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement |

`reward_log.jsonl` is one JSON object per line. Its last entry must carry `macro_f1`, `localization_rate` and `elapsed_seconds`, and each is compared against the verifier's own recomputation — the two rates within 0.005 absolute, the seconds within 2 percent. A larger gap is a fabrication, not a rounding.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts, and the workspace is wiped between them.

## Red lines

Any of these scores zero:

- Any read of, or write to, `tests/heldout/` bytes from the agent container — including holding a re-serialised copy of the private index vector, step-count map or stratum map anywhere in your source.
- Any write into the grading tree or verifier environment.
- Cross-attempt persistence of fitted thresholds or caches outside the fed-back summary.
- A label vector whose length or ids do not match the held-out manifest. This fails loudly rather than being silently graded down.
- Fabricated `reward_log.jsonl` values that disagree with host-observed metrics.
- Verifier-side network egress of any kind.

## What ships

- `environment/` — the corpus and a runnable starter.
- `environment/starter/` — a correct, unoptimised baseline you may replace entirely. It writes a well-formed coupled artifact and a complete run record, so the delivery path is exercised before you change a line of it.

## Traps

- The neutral class is collapsed into good or bad. Binary step accuracy stays respectable, the per-stratum neutral band is violated outright, and the score is exactly zero. Note that in a stratum whose human neutral rate is already under 0.03, emitting no neutrals at all passes the band while having thrown the class away — the checks look for neutral support in every stratum, not only for the band.
- Steps are scored independently and the first-error index is argmaxed separately, so the emitted index contradicts the emitted vector.
- A public scalar process reward model is thresholded globally. It passes aggregate macro-F1 and fails the per-stratum band, because a scalar has nowhere to put an annotated intermediate state.
- The sentinel is treated as a wildcard, so every no-error solution reads as a match.
- Indices are emitted one-based against a zero-based manifest. Macro-F1 is untouched and exact match collapses; a whole-shard offset is detected and named.
- Localisation is fitted on short solutions and collapses where the error sits behind eight or more correct distractor steps.
- The final answer is read instead of the process, so every correct-answer-with-a-bad-step solution is emitted as `-1`.
- The pipeline is accurate but unbudgeted and overruns the slow end of the ramp, or moves calibration outside the timed region and loses stage closure instead.
