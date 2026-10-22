# `748ada0e-0579-554b-ac37-51217bc67351`: `c8-s1-cifar100-step-budget`

Two model cohorts (`claude-opus-5` and `openai/gpt-5.6-sol`), one sequential attempt series each, 40 recorded rollouts packaged. Both series ran to their full 20.

The task is to improve a shipped CIFAR-100 training pipeline so it classifies better within a hard budget of 8000 optimizer updates. The score keys on *balanced accuracy at the budget*, never on how the model gets there: the agent writes class predictions for 10000 withheld images to `/submission/preds_i20k.npy`, and the verifier scores them against labels that exist only inside the grader. A floor of 0.43 balanced accuracy sits just above the best the shipped pipeline reaches across seeds, so an unchanged pipeline scores exactly 0; a knee at 0.81781 sits above the best the task author measured, so beating that best still earns more. `instruction.md` is the statement the agent receives.

```
task                    748ada0e-0579-554b-ac37-51217bc67351
reward protocol         accuracy_at_budget, linear ramp between two measured anchors
scoring engine          aello-eval 0.1.0 (harness/packages/aello-eval/), imported as
                        `aello_eval`. NOT vendored: built to a wheel at image-build
                        time and discarded. No .whl ships in this bundle.
score formula           reward = GATE x S_outcome x RubricFactor  (geometric, v2)
outcome ramp            S_outcome = clamp((balanced_accuracy - 0.43) / (0.81781 - 0.43), 0, 1)
budget                  8000 optimizer updates AND 1,024,000 sample presentations,
                        for the run that produces the submitted predictions
hardware                agent: 1 x H100. verifier: CPU-only (gpus = 0), separate
                        container, no network. Both run the SAME image, built from
                        the single Dockerfile under environment/.
dataset                 CIFAR-100, 20000 training images pre-staged under
                        environment/data/ -- the agent never downloads it
```

## Session result

| | `claude-opus-5` | `openai/gpt-5.6-sol` |
|---|---|---|
| rollouts packaged | **20 of 20** | **20 of 20** |
| best score | 0.7682, iteration 18 | **0.7799**, iteration 9 |
| mean / median / final | 0.6886 / 0.6972 / 0.7299 | **0.6966 / 0.7277** / 0.7473 |
| worst score | **0.4745**, iteration 1 | 0.3714, iteration 7 |
| balanced accuracy, min / median / max | **0.6945 / 0.7356 / 0.7473** | 0.5850 / 0.7274 / 0.7382 |
| S_outcome, min / median / max | **0.6820 / 0.7880 / 0.8182** | 0.3997 / 0.7670 / 0.7947 |
| S_outcome = 1.0 | **0 of 20** | **0 of 20** |
| RubricFactor, min / median / max | 0.6957 / 0.8958 / 0.9863 | **0.8768** / 0.9466 / 0.9813 |
| GATE = 1 | 20 of 20 | 20 of 20 |
| nulls / gated zeros | 0 / 0 | 0 / 0 |
| tokens, input / output | 59,082,677 / 1,153,869 | **20,057,235 / 253,277** |
| cost | $83.13 | **$16.39** |
| campaign span, first to last start | 18.64 h | 9.93 h |

Neither cohort produced a null or a gated zero: all 40 packaged rollouts carry a real number, every one of them cleared the single knockout, and both series recovered from their worst iteration rather than collapsing.

The two models land within 0.0117 of each other on best score and 0.008 on mean, which is inside the spread either series shows between adjacent iterations — this task does not separate them on outcome. It separates them sharply on cost. opus spent **2.95x the input tokens and 5.07x the dollars** to arrive slightly lower. Read the other way, opus is the steadier of the two: its worst rollout (0.4745) is a full 0.10 above gpt's worst (0.3714), and its balanced accuracy never fell below 0.6945 where gpt's dipped to 0.5850. gpt reaches a higher ceiling with a wider floor.

gpt's best came early, at iteration 9 of 20, and the series then oscillated without exceeding it. opus's best came at iteration 18, with its three strongest results (0.7566, 0.7458, 0.7682) all in the final third — that series was still climbing when the iteration budget ran out. On this evidence gpt converged and opus did not finish converging; a longer chain would be the way to test whether the ordering holds.

## Conformance to the client specification

| requirement | state |
|---|---|
| `instruction.md` states an objective with headroom | pass: "make the pipeline classify better" against a floor placed above the shipped pipeline's best across seeds, and a knee at 0.81781 deliberately set *above* the best the author measured, so the top of the scale is unreached by construction |
| a suboptimal implementation to improve on | pass: the shipped pipeline in `environment/starter/` lands near 0.35 balanced accuracy locally and scores exactly 0 under the ramp. `solution/TRUTH.md` records the dominant defect: an LR schedule built for ~31k steps against a loop that stops at 8000, worth ~24 points of balanced accuracy on its own |
| score is a float in [0,1], never binary | pass: `reward = GATE x S_outcome x RubricFactor`, exactly 0.0 when any knockout fails; recorded runs span 0.3714 to 0.7799 |
| one H100 | pass: `[environment] gpus = 1`, `gpu_types = ["H100"]`, and `environment_mode = "separate"` so the verifier re-scores outside the agent's container. `[verifier.environment]` re-points the same image with `gpus = 0`: grading is integer comparison and must not hold an accelerator. Without that block Harbor deep-copies `[environment]` for the verifier and would reserve an H100 to compare integers |
| `task.toml` tags `optimization` and `bounded-continuous` | pass: both present in `keywords` and in `metadata.tags`, alongside `vision`, `image-classification`, `training-recipe` |
| a rubrics file plus an LLM trajectory grader | pass: `tests/rubrics.json` carries 6 council rubrics; a 3-judge panel reads the agent's real code and trajectory and returns per-rubric evidence in every `council_results.json` |
| only a solution where all rubrics pass counts as correct | narrower than the prose: 1 rubric is a knockout (`no_graded_split_access`) and vetoes the score outright; the other 5 are *scored* and fold into `RubricFactor`, so a partial failure discounts rather than invalidates. Rank on `score`; read the rubric vector in `reward_detail.json` for per-rubric state |
| refinement loop: summarise, reattempt, fixed count, pick best | pass: every iteration after the first receives `iter_context.md` carrying the best attempt so far plus a bounded digest of earlier ones with their measured scores. `final_selection = "best"` |
| an HTML file to inspect each task | pass: `inspector.html`, covering both cohorts, all 40 packaged rollouts, and the task bundle's files |
| two plots, x axis iterations and x axis total tokens | **partial**: `plots/score_vs_iteration_{opus,gpt}.svg` and `plots/superimposed.svg` are present. **There is no tokens-axis plot in this bundle.** The curve exists inside `inspector.html` section 3, which plots score against cumulative tokens for both cohorts, but it has not been exported to `plots/` |
| full internet for setup, agent installs what it needs | narrower than the prose, and deliberately so. `[environment]` and `[agent]` are both `network_mode = "allowlist"`: the agent reaches its own model endpoint and the apt/PyPI/nodesource mirrors named in `allowed_hosts`, nothing else, and CIFAR-100 is pre-staged rather than downloaded. The verifier is tighter still — `[verifier]` and `[verifier.environment]` are both `network_mode = "no-network"`, because grading is integer comparison over the submitted predictions and has nothing to fetch. Declared explicitly rather than left to inherit, since the vocabulary is no-network/public/allowlist and there is no "none" |
| `max_timeout` in hours, capped at N attempts | **partial**: `metadata.max_timeout = 1.0` hour with `per_attempt_seconds = 432.0`. There is **no attempt-cap key in this manifest**, and Harbor's `TaskConfig` schema defines none. What bounds a single attempt is `[agent] timeout_sec = 1800` with the scaffold's x2 multiplier — a 3600 s hard kill, 8.3x the declared 432 s per-attempt figure. The number of attempts is set by the campaign driver, not by this manifest; **both series ran 20** |

## How this bundle builds and grades

**One Dockerfile, two containers, and the graded labels only ever reach one of them.**

`environment/Dockerfile` is the only Dockerfile in the bundle. The verifier has no separate build
context: both the agent container and the separate verifier container run the image this file
produces, tagged `aello/c8-s1-cifar100-env:1.1.0`.

That works because `[environment].docker_image` is pinned, and Harbor keys its tests-upload
decision off exactly that field — `skip_tests_upload = env_config.docker_image is None`
(`harbor/trial/trial.py:626`). With the pin set, Harbor uploads `tests/` into the separate
verifier container at grade time (`harbor/verifier/verifier.py:145-153`) instead of expecting the
image to already carry `/tests`. `[verifier].environment_mode` stays `"separate"`, so that upload
targets a container the agent never runs in. Harbor's own note at `trial.py:623-625` states the
same guarantee. **The isolation is structural, not timing-based: `tests/heldout/graded_test.npz`
is never present in the agent's container and is never baked into an image layer.**

Do not "simplify" this to `environment_mode = "shared"`. Shared mode uploads `tests/` into the
*agent's* container and puts the graded labels there.

**The scoring engine is built, not shipped.** `environment/Dockerfile` contains the literal token
`.aello_eval_wheel`, which arms `harness/scaffold/deliver.py:_stage_aello_eval_wheel`: at build
time the scaffold builds a wheel from `harness/packages/aello-eval`, stages it into the build
context, lets Docker `COPY` it, and a cleanup callback deletes the staged directory. **No `.whl`
exists in the delivered bundle and none is left behind after a build.**

**Sharing one image has a cost, and the Dockerfile pays it down.** Anything installed in that image
is readable by the agent. The council configuration is the part that matters: `judges.yaml` names the
judge panel, the quorum and the per-rubric scoring policy, and an agent that can read it can tailor
its trajectory to the judges that will grade it. So the Dockerfile **strips the council surface
after install** — `judges.yaml*`, `council.py`, `creds.py`, `context.py`, `preflight.py` are
deleted from site-packages, leaving only the deterministic path the verifier actually calls
(`aggregate`, `checks`, `score`, `schema`, `outcome`). `aello_eval/__init__.py` imports the council
lazily so that deletion leaves `import aello_eval` working. The build asserts both halves: it fails
if the deterministic imports break, and it fails if any council file survives.

What remains readable to the agent is the deterministic aggregation arithmetic, which is already
stated openly in `task.toml` and `instruction.md`.

### Running it

**`harness/scaffold/deliver.py` is a required step, not a convenience.** Because
`[environment].docker_image` is pinned, a plain `harbor run` takes Harbor's prebuilt path and tries
to *pull* `aello/c8-s1-cifar100-env:1.1.0`, which exists in no registry. `_ensure_env_image` is what
builds that tag locally from `environment/`, with the wheel staged. Its second branch is a no-op
here: the verifier tag is the same tag, already present, so nothing is ever built from `tests/`.

Prerequisites on the machine that runs it: Docker with buildx, and `uv` (the stager falls back to
`pip wheel`). Build-time network for the apt, nodesource and PyPI mirrors already named in
`[environment].allowed_hosts`.

Two things Harbor does at grade time that the image therefore does not need to: it creates
`/logs/verifier` (`trial.py:599`) and the parents of every uploaded artifact
(`artifact_handler.py:248-250`), and it runs `chmod +x` on the uploaded `test.sh` as root before
executing it (`verifier.py:190-195`) — which matters, because `.sh` scripts are executed directly
rather than through an interpreter (`utils/scripts.py:147-149`).

## Caveats you should read before quoting a number

**S_outcome is not saturated.** Zero of the 40 rollouts reach `S_outcome = 1.0`. The observed range is 0.3997 to 0.8182 — the ramp is doing real work across its whole width, and the outcome axis genuinely discriminates between attempts rather than clamping. The knee at 0.81781 sits above the best the task author measured, and no packaged rollout has reached it. **The score is a genuine blend of outcome and process, and outcome dominates the variance.**

**No knockout ever fired, so the gate is untested on this data.** There are two knockout layers and neither tripped. The deterministic layer carries **three** knockouts — `graded_split_isolated`, `predictions_present`, `predictions_wellformed` — and the council set carries one, `no_graded_split_access`. `GATE = 1` in all 40 rollouts, `failed_knockouts` is empty in all 40, and `no_graded_split_access` scored 1.0 every time. That is a clean sheet, but it means the packaged evidence contains no example of any veto path actually executing. A clean sheet on a knockout is not evidence that the knockout works, only that nothing tripped it.

**The packaged rollouts were graded before the build path described above was in place.** The
scoring contract they were graded under is the one documented here — same ramp anchors, same
`reward = GATE x S_outcome x RubricFactor`, same rubric set — and the arithmetic reconciles exactly
against every recorded `reward_detail.json`. The container plumbing that carried it differed. Treat
the recorded scores as valid measurements of the scoring contract; re-run the campaign if you need
rollouts produced end-to-end under the shipped build path.

**The cohorts were graded by different councils.** opus was judged by `openai/gpt-5.6-sol`, `claude-sonnet-4-5` and `claude-haiku-4-5`; gpt by `claude-opus-5`, `claude-sonnet-5` and `claude-haiku-4-5`. Only haiku is common to both, and each cohort is judged in part by the other cohort's model. **A judge silently contributed nothing in 6 of the 40 rollouts, and the record does not call it a failure.** In opus iterations 10 and 20, `claude-sonnet-4-5` produced no per-rubric verdict; in gpt iterations 17, 18, 19 and 20 — the last four of that series — `claude-opus-5` did the same. In all six, `council_status` is `ok`, `council.failed` is empty, and all three judges are listed under `council.responded`; the only signal is `panel_quorum` falling from 1.0 to 0.667 and the per-rubric `judges` arrays in `council_results.json` holding two entries instead of three. Those grades stood because 0.667 exactly meets `required_quorum`. **Anyone auditing this campaign on `council_status` or `council.failed` alone would conclude the council was healthy in all 40 rollouts; it was not.** Check `panel_quorum` per run instead.

The effect on the numbers is small but not nil and does not point one way: opus's two 2-judge grades average a RubricFactor of 0.9399 against 0.8716 for its 18 three-judge grades, while gpt's four average 0.9268 against 0.9499. gpt's cluster is the one to treat carefully — it covers iterations 17-20 consecutively, so the tail of that series, including its final score, was graded by a thinner panel than the rest.

**Every rollout hit the council context cap.** `context_chars` is 16,027 for opus and 16,026 for gpt at the median, against a cap of 16,025 — i.e. **both cohorts are truncated in essentially every rollout**, not occasionally. The judges are scoring a bounded excerpt of the agent's code and trajectory, not the whole of it. `faithful_reporting` is the lowest-scoring rubric for both cohorts (opus 0.760, gpt 0.723), and truncation is the most likely reason a judge cannot verify a report against the work: it has not been shown all of the work.

**The rubric vector is where the two cohorts actually differ.** opus scores lower on `honest_step_budget` (0.760 vs gpt's 0.977) and `no_val_training` (0.885 vs 0.975), and its RubricFactor floor is 0.6957 against gpt's 0.8768. Since outcome is close, most of opus's deficit in mean score is process, not accuracy. Whether that reflects genuine budget-declaration drift or a judging artefact of the truncation above is not resolvable from the packaged bytes; the per-rubric justifications in `council_results.json` are the place to start.

**Three opus iterations were re-run after upstream failures.** Iterations 12, 13 and 14 of the opus series first terminated on `api_error_status 529` (upstream overload) after a single agent turn, producing no predictions. Those attempts recorded `MISSING_PREDICTIONS` -> `agent_miss` -> a scoreable 0.0, were discarded, and the iterations were re-run to completion; the packaged 12/13/14 (0.6251, 0.6639, 0.7053) are the successful re-runs. This is why the opus campaign span of 18.64 h is not a runtime figure — it contains idle periods where the chain was deliberately stopped. **Per-run wall clock is not recoverable from the packaged artefacts for either cohort**, so no per-attempt timing is quoted above.

**The two series are conditioned, not independent.** Each iteration after the first was shown the measured scores of its predecessors, so a cohort's rollouts are one sequence in which each attempt saw the outcomes of the earlier ones. No pooled mean over them is a population statistic, and the mean is particularly misleading for opus, whose series was still trending upward at iteration 20.

## Known open items

**`[task].name` does not satisfy Harbor's schema.** It is `c8-s1-cifar100-step-budget`; `TaskConfig`
requires `org/name` format and rejects the manifest — *"Package name must be in 'org/name' format"*.
Harbor will not load the task until this changes. Changing it also changes trial directory naming,
which the recorded `trial_uri` paths in `trajectories/*/iter_*/reward.json` embed, so the rename and
the recorded campaign need to be reconciled together.

**`tests/heldout/known_overlap.json` is absent.** `tests/isolation.py:24` expects it — the recorded
list of byte-identical upstream duplicates the overlap scan is allowed to ignore. The deterministic
checks in `test_output.py` do not read it, so grading completes without it; the standalone isolation
probe will not run.

**The verifier does not emit `zero_score_reason`, and `test.sh` has no reward fallback.** The
honest-zero-versus-infrastructure-fault distinction is made — `outcome.json` carries an
`outcome_status` of `agent_miss` or `infra_failure` plus a reason string — but it is not surfaced
under that key in `reward_detail.json`. Separately, `tests/test.sh` runs under `set -euo pipefail`
with no trap: if `verifier.py` exits before `write_reward`, no reward file is written and the trial
carries no score rather than an explicit zero.

**No tokens-axis plot is exported.** The curve exists in `inspector.html` section 3; it has not been
written to `plots/`.

## Provenance

Scores are computed host-side by `eval` from the verifier's own scoring of the submitted prediction file against private labels; agent-reported metrics are advisory and never enter the score. Every number in this README, the inspector and the plots is read from the recorded artefacts under `trajectories/`, not restated from a model-written summary — one directory per cohort, `opus-5` and `gpt-5-6-sol`.

## Files

```
README.md                   this file
inspector.html              browsable view of the task and all 40 recorded runs
plots/                      score vs iteration, one per cohort
                            (score_vs_iteration_{opus,gpt}.svg) plus superimposed.svg
                            NOTE: no tokens-axis plot exported; that curve is in
                            inspector.html section 3
instruction.md              the statement handed to the agent
task.toml                   manifest: image, GPU, budgets, network allowlist, artefacts
environment/
  Dockerfile                THE ONLY Dockerfile. Builds the image both the agent and the
                            separate verifier run. Installs the scoring engine from a
                            wheel staged at build time, then strips the council surface
                            so the agent cannot read the judge panel. No .whl is shipped
  aello_data.py             dataset loader
  starter/                  the shipped, deliberately weak training pipeline
  data/                     CIFAR-100 train split and the graded image bundle
tests/                      NOT baked into any image -- uploaded by Harbor into the
                            separate verifier container at grade time
  test.sh                   verifier entry point
  verifier.py               scores the submission; owns the ramp and its two anchors
  test_output.py            the 5 deterministic checks. THREE are knockouts:
                            graded_split_isolated, predictions_present,
                            predictions_wellformed. Scored, non-knockout:
                            predictions_not_degenerate (w=3), step_budget_declared (w=5)
  rubrics.json              the 6 council rubrics (rubrics.jsonl: legacy line format)
  isolation.py              graded-split isolation probe (needs heldout/known_overlap.json,
                            which is absent -- see Known open items)
  constants.json            the ramp anchors and budget constants
  heldout/
    graded_test.npz         the private graded labels. Never enters the agent's container
                            and never enters an image layer
solution/
  TRUTH.md                  the golden trajectory: the LR-schedule/budget mismatch
                            worth ~24 points, and the rejected score-hacking routes

trajectories/opus-5/         one directory per cohort, named for the model
trajectories/gpt-5-6-sol/
  history.jsonl             one line per iteration: score, source, tokens
  iter_NNN/
    reward.json             the score for this iteration and the token account
    reward_detail.json      GATE, S_outcome, RubricFactor, floor, formula, failed
                            knockouts, the full rubric vector, council panel and status
    council_results.json    all 6 council rubrics, per judge, with evidence
    score.txt               the bare score, one float
    summary.md              the loop's own account of this iteration
    iter_context.md         the history block this iteration was shown
    <task-uuid>/            the trial as recorded
      result.json           task_checksum, config, token account
      config.json, lock.json
      agent/agent_code/     every file the agent left in /workspace, verbatim
      verifier/             outcome.json (balanced accuracy, ramp anchors, S_outcome)
                            and check_results.json (the deterministic verdicts)
```
