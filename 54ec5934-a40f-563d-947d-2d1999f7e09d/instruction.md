# C7-S2

your deliverable is agent/policy.py and any artifacts it builds at graded-run time. The grader launches a fresh graded run of your scaffold against a private held-out split of 300 WebArena instances on locally hosted sites, driving a pinned open-weights policy checkpoint on the single H100. Reach >= 0.26 task success on that private split in the least graded-phase wall-clock, with no episode exceeding 240 s or 30 actions.

## Who scores you

Not this harness, and not your scaffold. Success on an instance is decided by WebArena's own programmatic evaluators, vendored at a pinned upstream commit and run by the verifier over site fixtures it owns. You cannot read them while the run is graded and you may not change a byte of them. Each instance is scored by exactly one evaluator family, and each family reads exactly one field out of the episode you leave behind:

| evaluator family | the field it reads |
| --- | --- |
| `exact_match`, `must_include`, `fuzzy_match` | `answer` -- the answer string |
| `url_match` | `final_url` -- the page the episode ended on |
| `program_html` | `site_end_state_hash` -- the post-episode row-set hash of the affected tables |

An episode that reached the goal and then terminated without that field is not a near miss. It is unscoreable, and it costs exactly what never running the episode would have cost.

## What is graded

Write your run ledger to **`/submission/c7s2.json`**. It must be an object with an ids list and a matching outputs list:

- `ids` -- the pod-minted instance ids the graded run hands you, all 300 of them, no duplicates and none invented. The private split is the upstream's own 241 intent templates re-bound to different real entities in the shipped site databases, so the public instances and their circulated answer keys name entities that are not your targets. Two graded instances that share a template were re-bound to *different* entities; one answer reused across such a group is the memorised-answer-key route showing itself.
- `outputs` -- one terminal record per id, aligned position by position. Each record carries the field its instance's evaluator family reads, plus `actions`, `elapsed_s`, and `terminated_by` (`goal_confirmed`, `action_budget` or `time_budget`).

A missing, malformed or out-of-domain artifact scores 0.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. The graded quantity is task success rate on the 300-instance private held-out split under WebArena's own programmatic evaluators, gated by a per-episode latency and action-count SLO, with graded-phase wall-clock seconds as the optimized axis.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The per-instance budget

The budget gate binds on **one episode**, not on the run. A comfortable mean buys nothing.

- Every episode: at most **30 actions** and at most **240 s** of wall clock, with **p95 <= 180 s** across the 300. Measured host-side, per episode, even under concurrency.
- An episode you stop at its ceiling still owes the evaluator a terminal record. Mark it `terminated_by: action_budget` or `time_budget`, carry `actions` and `elapsed_s`, and carry the evaluator's field with whatever the episode reached. A recorded budget stop scores zero on that instance; an unrecorded one is a hole in the ledger.
- You must declare your own budget accounting in the run record as `agent_budget_declaration`, naming `per_instance_action_ceiling`, `per_instance_latency_ceiling_s` and `episode_concurrency`, each at or inside the published gate. The declaration is what is graded, not the harness's own measurement, and a run total divided by 300 is not a per-instance budget.
- Per-episode site restore from the pinned per-site dump is metered separately and **excluded** from the optimized clock. Skipping it buys you nothing on the axis and leaves the next episode inheriting the previous one's state.
- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget. Up to 50 attempts, reward history carried between them.

## Scope

Five sites ship and are graded: `shopping` (OneStopMarket), `shopping_admin`, `gitlab`, `map` and `wikipedia`. The Postmill site is excluded by name because its Pushshift-derived content carries no redistributable licence. An episode mutates its own target site and no other.

## Red lines

Any of these scores zero:

- Read or write under tests/heldout/, the evaluator tree or the grading constants -> 0.0, redline:holdout-access.
- Any outbound request to a host outside the local site allowlist during the graded run -> 0.0, redline:graded-run-network.
- Served checkpoint digest differs from the pin -> 0.0, redline:model-substitution.
- Any byte change to WebArena evaluators, task JSON or reward code -> 0.0, redline:harness-tamper.
- Site snapshot digest mismatch at episode start (fixture tampering) -> 0.0, redline:fixture-tamper.
- Any episode > 240 s or > 30 actions, or p95 > 180 s -> 0.0, redline:slo-violation.
- |success(private) - success(sibling)| > 0.08 -> 0.0, redline:shard-overfit.
- Unchanged starter or absent agent/policy.py -> exactly 0.0, redline:empty-submission.

## What ships

- `environment/` -- the corpus and a runnable starter.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely.
- `environment/data/dev/` -- the public dev split for the retained sites. Induce whatever you learn from it **inside the graded run**; a workflow memory that arrives as a committed file beside `policy.py` was built somewhere the grader cannot replay.

## Traps

- Goal displacement across a 30-action horizon: the scaffold re-navigates and terminates with a partially configured state that the site end-state evaluator rejects.
- The unscoreable episode: navigation worked, verification confirmed the goal, and the terminal record went out without the field that instance's evaluator reads.
- Observation overflow: unpruned accessibility trees push the goal out of the attended window on the distractor-heavy instances, and naive truncation drops the correct element in silence.
- Answering the intent template rather than the instance, so one answer repeats across a re-bound group.
- Order violation in an induced workflow: the recipe's steps are applied in the wrong order (checkout before coupon), so the final total is wrong even though every step ran.
- Cross-site collateral writes when a multi-site task is mis-decomposed.
- SLO collapse: raising per-episode search depth or concurrency to lift success pushes p95 past 180 s and zeroes the run.
- Dev-split overfit: site-specific hardcoded selectors tuned on the public split break on re-bound private instances and open a sibling gap above 0.08.
