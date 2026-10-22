# `bdb3bb73-f7b8-5922-8b12-f8303e59b13c`: `cifar10-94pct-1xh100`

Three model cohorts (`claude-opus-5`, `openai/gpt-5.6-sol` and `meta/muse-spark-1.2`), one sequential attempt series each, 85 recorded rollouts packaged. The two Claude/GPT series ran to their full 30; muse-spark is packaged at 25 of 30 with the series still in flight, so its figures will move.

The task is to train CIFAR-10 from scratch to 94% top-1 on the test split as fast as possible on a single H100. The score keys on *when* the target is first crossed, never on how high accuracy finally lands: the agent writes one TorchScript checkpoint per epoch, and the verifier independently re-evaluates that ladder on the real test split to find the first crossing. A ResNet9 reference crosses at 69.9 s and a whitened-stem speedrun at 28.1 s; those two anchors define the scale. `instruction.md` is the statement the agent receives.

```
task                    bdb3bb73-f7b8-5922-8b12-f8303e59b13c
reward protocol         v2_normalized_time_to_accuracy
scoring engine          aello_eval 0.1.0 (harness/packages/aello-eval/)
image                   aello-bdb3bb73-env:v4, built from environment/Dockerfile and
                        pinned by tag so the agent and the separate verifier share it
hardware                1 x H100-SXM5-80GB, agent and verifier alike
calibration             PILOT_CALIBRATED_2026-08-14 on the target GPU, median of 3 runs
dataset                 CIFAR-10, fast-ai S3 tarball, imagefolder PNG, pre-staged at
                        /workspace/data/cifar10/ -- the agent never downloads it
```

## Session result

| | `claude-opus-5` | `openai/gpt-5.6-sol` | `meta/muse-spark-1.2` |
|---|---|---|---|
| rollouts packaged | **30 of 30** | **30 of 30** | 25 of 25|
| best score | **0.9845**, iteration 8 | **0.9564**, iteration 15 | 0.8600, iteration 20 |
| mean / median / final | 0.9554 / 0.9588 / 0.9448 | 0.8738 / 0.9065 / 0.9167 | 0.2215 / 0.0000 / 0.0000 |
| mean over scoring runs | 0.9554 (30) | 0.9039 (29) | 0.7909 (7) |
| worst score | 0.9172 | **0.0000**, iteration 3 (gated) | **0.0000**, 18 of 25 |
| t\* to 94%, min / median / max | 6.55 / 15.61 / 27.85 s | **1.28 / 2.46 / 36.21 s** | 0.49 / 184.07 / 479.43 s |
| crossing epoch, min / median / max | 16 / 23 / 74 | 8 / 8 / 37 | 23 / 37 / 68 |
| re-eval accuracy at crossing | 94.02 / 94.25 / 94.80 % | 94.01 / 94.14 / 94.37 % | 94.02 / 94.22 / 94.83 % |
| S_outcome = 1.0 | 30 of 30 | 29 of 30 | **6 of 25** |
| tokens, input / output | 125,365,532 / 2,145,948 | 38,925,433 / 357,678 | 68,233,317 / 2,087,313 |
| wall clock per run | 19.4 / 36.0 / 53.2 min | 6.0 / 8.8 / 17.8 min | not preserved by packaging |
| total wall clock, cost | 17.74 h, $143.31 | 4.83 h, $41.81 | 26.39 h, $50.27 |

muse-spark quantiles for t\* and crossing epoch cover the 21 of 25 rollouts that produced a ladder at all. Its
`cost_usd` is null in every `reward.json` — the agent runs through OpenRouter — so the cost above is summed from the
per-call `cost` fields in the packaged OpenHands session events, de-duplicated for the action/observation pairing
that records each call twice. Per-run wall clock is not recoverable from the packaged bytes for this cohort.

Iteration 8 produced opus's best submission: a 6.08M-parameter convnet with frozen ZCA whitening initialised from train-split patch covariance, GPU-resident augmentation, and six-way test-time augmentation baked into `forward` so the scripted checkpoint carries it. The agent quantified the noise floor first — 0.46 pp attributable to cuDNN autotuning — then ranked finalists on a 3-seed mean before promoting the graded ladder. gpt's best, iteration 15, is the AirBench94 + Muon recipe reconstructed locally, crossing at epoch 8 in 8.90 s; its fastest crossing anywhere in the series is 1.28 s. The two completed cohorts sit at opposite ends of one trade: gpt reaches the target roughly six times faster in training time and at a third of the epochs, opus scores higher on eight of the ten council rubrics, gpt on the two CIFAR pipeline rubrics. muse-spark is not on that frontier: 18 of its 25 rollouts score 0 because training never crosses 94% inside the scale, and its best run (0.8600, iteration 20) trails gpt's worst ungated result. When it does cross it crosses fast — its 0.49 s minimum is the quickest t\* recorded on this task — so the series reads as high variance rather than uniformly slow.

## Conformance to the client specification

| requirement | state |
|---|---|
| `instruction.md` states an objective with headroom | pass: "reach 94% top-1 accuracy on the CIFAR-10 test split as fast as possible… within a 1800-second wall-clock budget", headroomed against a measured reference at 28.1 s rather than the frontier |
| a suboptimal implementation to improve on | pass: `solution/solve.sh` is a ResNet9 / 30-epoch weak reference crossing at 69.9 s; `environment/airbench_strong_reference.py` is the 28.1 s strong anchor. Both measured on the target GPU |
| score is a float in [0,1], never binary | pass: `score = GATE × S_outcome × RubricFactor`, exactly 0.0 when a knockout fails; recorded runs span 0.0000 to 0.9845 |
| one H100 | pass: `gpus = 1`, `gpu_types = ["H100"]`, for the agent and the separate verifier alike |
| `task.toml` tags `optimization` and `bounded-continuous` | pass: both present in `keywords`, alongside `image_classification`, `cifar10`, `time_to_accuracy` |
| a rubrics file plus an LLM trajectory grader | pass: `tests/rubrics.json` carries 10 council rubrics; a 3-judge panel reads the agent's real code and trajectory and returns per-rubric evidence in every `council_results.json` |
| only a solution where all rubrics pass counts as correct | narrower than the prose: 5 knockouts are veto and zero the score outright, but the other 15 rubrics are *scored*, folding into `RubricFactor = 0.5 + 0.5 × Σws/Σw`, so a partial failure discounts the outcome instead of invalidating it. Rank on `score`; read the rubric vector in `reward_detail.json` for per-rubric state |
| refinement loop: summarise, reattempt, fixed count, pick best | pass: every iteration after the first receives `iter_context.md`, the best attempt so far plus a bounded digest of the top 3 and last 3 with their measured scores. Fixed target of 30 per cohort; best selected on score. muse-spark is packaged at 25 of that 30 |
| an HTML file to inspect each task | pass: `inspector.html`, covering all three cohorts, all 85 packaged rollouts and every dataset file |
| two plots, x axis iterations and x axis total tokens | pass: `plots/iterations_{opus,gpt,muse}.svg`, `plots/tokens_{opus,gpt,muse}.svg` |
| full internet for setup, agent installs what it needs | narrower than the prose: `network_mode = "allowlist"` in both phases. The agent reaches its own model endpoint and the package mirrors, nothing else; CIFAR-10 is pre-staged rather than downloaded. The bundle copies differ only in this list, so each cohort can reach its own model endpoint — muse-spark's adds the OpenRouter host |
| `max_timeout` in hours, capped at N attempts | partial: the agent phase is `timeout_sec = 1800` with a ×2 multiplier, so a 3600 s hard kill per attempt, stated explicitly in the instruction. The cap is expressed as an iteration target on the refinement chain (30) rather than an `attempt_cap` key in `task.toml` |

## Caveats you should read before quoting a number

**S_outcome is saturated; the score is effectively RubricFactor.** `S_outcome = clamp((69.901 − t*) / (69.901 − 28.138), 0, 1)`, so any run crossing 94% in under 28.1 s clamps to 1.0. The two completed cohorts clear that on essentially every attempt, and `S_outcome = 1.0` in 59 of their 60 rollouts, the sole exception being gpt iteration 27 at t\* 36.21 s. **muse-spark does not clear it:** only 6 of its 25 rollouts reach `S_outcome = 1.0`, with a median t\* of 184 s against a 69.9 s baseline, so for that cohort the outcome term rather than the rubric vector drives the score. Saturation is a property of the two strong cohorts, not of the scale — the headroom above 28.1 s is real and this third series occupies it. With `GATE = 1` everywhere except one gpt run, `score` and `rubric_factor` are numerically identical in those 59. The outcome axis this task is named for does not currently separate the two models; the process rubrics do. Part of the cause is that `t*` is now the training-run clock, rebuilt from checkpoint mtimes and excluding the agent's authoring time, while both anchors still carry their pre-training startup. `tests/grounding.yaml` records this and names the fix: re-measure both references under the current definition.

**Two checks are not implemented.** `no_test_set_leakage` (a knockout) and `full_training_set_used` (weight 3) both self-report `"enforced": false` in all 85 packaged rollouts, with the remedy named in their own evidence blob, "needs dataloader trace or alternate eval set". Both pass by default, so a run that trained on the test split and a run that did not score identically here. A clean sheet on those two axes is not evidence of anything.

**The cohorts were graded by different councils.** opus was judged by `gpt-5.6-sol`, `claude-sonnet-4-5` and `claude-haiku-4-5`; gpt by `claude-opus-5`, `claude-sonnet-5` and `claude-haiku-4-5`; muse-spark by that same second panel, so it is directly comparable to gpt on judging and only partly to opus. Only haiku is common to all three. `claude-sonnet-4-5` returned no verdict in 15 of opus's 30 rollouts, which were then graded by a 2-judge panel; the panel and the no-shows are recorded per run. Re-scoring the two completed cohorts on the shared judge alone reproduces the ranking and the margin to within 0.002, so the result survives the one control available, but a fixed panel per campaign would remove the exposure.

**Two rubrics are scored on evidence the judges do not receive.** The council context is capped at 16,025 characters. opus authored a median of 48 KB per iteration, so the panel sees roughly a third of the code before any trajectory competes for the same budget; gpt authored a median of 10 KB and fits; muse-spark authored a median of 104 KB, more than twice opus's, making it the most truncation-exposed of the three. `iterated_genuinely` additionally asks the judge to inspect behaviour *across iterations*, but the council is invoked once per trial and receives one trial. Both rubrics score low across all three cohorts, and the recorded justifications say why in plain terms: "the excerpt is too truncated", "defaults to the insufficient-evidence anchor". Neither is a capability signal.

**gpt iteration 3 is a gated zero, not a training failure.** The run reached 94.15% at epoch 30 and would have scored on the outcome, but wrote its time-to-94% (11.503 s) into `agent_reported_metrics.json` as `wall_clock_sec` against a 15.812 s checkpoint span. The instruction requires that field be the total training wall clock and forbids a partial time there; under-reporting tripped the `timer_honesty` knockout. The field is optional. That one rollout moves gpt's mean by 0.0301, about a third of the gap between the cohorts, so quote the mean over ungated runs (0.9039) alongside it.

**The three series are conditioned, not independent.** Each iteration after the first was shown the measured scores of its predecessors, so a cohort's rollouts are one sequence in which each attempt saw the outcomes of the earlier ones. No pooled mean over them is a population statistic. This bites hardest on muse-spark, whose mean is dragged to 0.2215 by 18 gated zeros; quote the mean over its scoring runs (0.7909, n=7) alongside it, and treat both as provisional until the series reaches 30.

## Provenance

Scores are computed host-side by `aello_eval` from the verifier's own re-evaluation of the checkpoint ladder; agent-reported metrics are advisory and never enter the score. Every number in the inspector and the plots is read from the recorded artefacts under `trajectories/`, not restated from a model-written summary — one directory per cohort, `opus-5`, `gpt-5-6-sol` and `muse-spark-1-2`. The audit record for the underlying bundle is held separately and available on request.

## Files

```
README.md                   this file
inspector.html              browsable view of the task and all 85 recorded runs
plots/                      the client plots: score vs iteration, one per cohort
                            (iterations_{opus,gpt,muse}.svg) plus superimposed.svg
instruction.md              the statement handed to the agent
task.toml                   manifest: image, GPU, budgets, network allowlist, artefacts
environment/
  Dockerfile                builds the image shared by agent and verifier
tests/
  test.sh                   verifier entry point
  verifier.py               re-evaluates the ladder; owns the clock, t* and the anchors
  test_output.py            the 10 deterministic checks, 5 of them knockouts
  rubrics.json              the 10 council rubrics (rubrics.jsonl: legacy line format)
  fixtures/                 frozen inputs proving both halves of the enforced checks
solution/
  TRUTH.md                  the golden trajectory and the rejected score-hacking routes

trajectories/opus-5/         one directory per cohort, named for the model
trajectories/gpt-5-6-sol/
trajectories/muse-spark-1-2/   packaged at 25 of 30; series still in flight
  history.jsonl             one line per iteration: score, source, tokens
  iter_NNN/
    reward.json             the score for this iteration and the token account
    reward_detail.json      GATE, S_outcome, RubricFactor, floor, formula, failed
                            knockouts, the full rubric vector, council panel and status
    council_results.json    all 10 council rubrics, per judge, with evidence
    score.txt               the bare score, one float
    summary.md              the loop's own account of this iteration
    iter_context.md         the history block this iteration was shown
    <task-uuid>/            the trial as recorded
      result.json           task_checksum, phase timings, token account
      config.json, lock.json
      agent/agent_code/     every file the agent left in /workspace, verbatim
      verifier/             outcome.json (t*, crossing epoch, the re-eval ladder walk)
                            and check_results.json (the deterministic verdicts)
```
