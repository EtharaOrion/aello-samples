# AELLO-C8-S2 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

Your deliverable each attempt is train.py at the top of the work directory plus an optional carry store at carry/notes.json. The verifier launches your program once per graded configuration as `python train.py --data <private data dir> --out <export path> --init-seed <integer> --config-id <id> --carry carry/notes.json` in a network-isolated container on one H100, times each run host-side from the first read of the private data directory to the write of that run's export, and evaluates each export on a private image-disjoint STL-10 shard it never gives you. There are thirty-two private configurations. Each carries its own label orientation, class-prior skew, crop offset, crop scale, channel statistics and boundary-ambiguity level, and instruction.md publishes the per-configuration balanced-accuracy bar b_c for all thirty-two but publishes none of the six constants. Attempts one through forty-nine are graded on three configurations each, drawn by a deterministic schedule that instruction.md publishes in full; attempt fifty, which is the scored attempt because final selection is last, is graded on the eight configurations whose last scheduled visit is oldest at that point. Clear b_c on a configuration and your score for it rises as that run gets faster, saturating at the published knee and falling to the gate value at the published per-configuration ceiling; the attempt score is the mean over its graded configurations. Between attempts the harness carries forward exactly one artifact: carry/notes.json, at most 2,048 bytes, validating against the closed schema instruction.md prints, one entry per configuration with six named scalar fields and a one-bit src field marking the entry observed or predicted. The harness never truncates it: an oversize or schema-invalid store fails the attempt loudly with reason carry_store_invalid and the previous valid store is what carries forward. The verifier alone writes each entry's last_confirmed stamp, and only when that attempt actually graded that configuration and cleared its bar; any entry whose stamp is more than eight attempts old is dropped before the store is delivered to the next attempt, and the identifiers dropped are reported back to you in that attempt's record. A bounded number of configurations are re-parameterized once during the run; how many, which, and when are not published. Everything about how you train, what you store, how you encode it and when you spend an attempt refreshing rather than exploiting is yours to choose, and the shipped starter/train.py is a working, deliberately unoptimized program that ignores the carry store entirely and that you may replace outright. The four requirements are jointly binding rather than a checklist: a fast trainer without the calibration misses the bars, the calibration without an encoding that fits 2,048 bytes fails loudly, the encoding without a refresh policy arrives at attempt fifty with the wrong half of the table live, and the refresh policy without a fast trainer caps every configuration at the slow end of the same ramp.

## The single most important insight

The time that separates a full-reward run from a floor run is not training time, it is discovery time, and discovery time can only be paid inside the timed region. A program handed a configuration it knows nothing about has to find that configuration's label orientation, prior skew, crop offset, crop scale, channel gain and ambiguity level by probing the private shard it was just given, and that probe costs about as much wall clock as the training it precedes. So the only way to be fast is to already know, and the only channel that carries knowing from one attempt to the next is 2,048 bytes wide, closed-schema, and expires eight attempts after the verifier last confirmed an entry. Read the schedule before you read anything else: attempt fifty is graded on the eight configurations whose last scheduled visit is oldest, which under the published rule were last visited at attempts 29 through 36, so every one of them is at least fourteen attempts past its time to live when it is graded. No scheduled visit can keep those eight live as observations. What survives expiry is what you marked predicted, and marking an entry predicted is honest exactly when you did not observe it, so the route the design rewards is to observe what the schedule gives you, generalize across the latent family to the configurations it does not, and carry the generalization forward under its own provenance. The route it punishes is to relabel a guess as an observation, which is a zero rather than a lower score.

## The ideal solve, step by step

1. **Read the schedule and work out which eight are graded** instruction.md publishes the attempt schedule in full: attempt t grades configurations (7t + o) mod 32 for o in 0, 11, 19, and attempt fifty grades the eight whose last scheduled visit is oldest. Compute that set on attempt one, not attempt forty. Everything downstream -- what to observe, what to encode, when to refresh -- is a scheduling decision against those eight, and an agent that never runs the arithmetic optimises the half of the table it is not graded on.

2. **Make the discovery probe cheap enough to be worth repeating** Before memory can help you, you need something worth remembering. Recover a configuration's six constants from private_train(c) itself -- a label-orientation probe on a small balanced subsample, a channel-statistics pass, an offset search over the published lattice -- and measure what that probe costs inside the timed region. It is the quantity NC-AMNESIC turns into a score, and it is the reason a memory-less run lands near the bottom of the ramp instead of somewhere in the middle.

3. **Get the trainer under the knee with the constants known** With the six constants in hand the run is an ordinary short training job on 96px images, and the levers are the ordinary ones: a single staged uint8 host-to-device transfer, bf16 channels-last, one-cycle scheduling, progressive resolution, and a residual net small enough to converge inside the budget. Profile the data pipeline before touching the architecture -- the shipped starter loses more time to per-batch host decoding than to arithmetic.

4. **Encode the table so thirty-two entries fit in 2,048 bytes** Thirty-two entries against a 2,048-byte cap is 64 bytes an entry including the JSON punctuation. Six full-precision floats do not fit; six quantized scalars at the published tolerances do. Quantize to the tolerance each field is actually compared at -- flip and ofs are discrete, scale needs 0.01, chan 0.02, rho and amb 0.05 -- and check the encoded size before you submit, because an oversize store does not truncate, it fails the attempt.

5. **Mark provenance honestly and let prediction do the carrying** An entry you measured on a configuration this run graded is observed; an entry you inferred from the latent structure is predicted. The verifier recomputes the first kind and exempts the second, so predicted entries are the ones that can survive expiry, and calling a guess an observation is a zero rather than a risk. Fit the latent map across the configurations you have observed and use it to populate the ones you have not.

6. **Spend attempts refreshing on a schedule you chose** Stamps are verifier-written and expire eight attempts after the last confirmation, and the final eight are graded fourteen or more attempts after their last scheduled visit. Decide early how many of the fifty attempts are refresh attempts, declare the split in the run record, and treat an attempt whose graded configurations you already know cold as an opportunity to re-observe rather than a chance to shave another second.

7. **Carry a consistency check for the re-parameterization** A bounded number of configurations are re-drawn once during the run, and the feedback carries no per-constant diagnostic to tell you which. A stored constant that has gone quietly wrong costs both bars on that configuration and diagnoses as nothing. Validate each recovered constant against the shard you were just handed at a cost you can afford -- under a second -- and fall back to the latent-map prediction when the check fails.

8. **Keep the timer boundary and the log honest** Decode the carry store, import your libraries and create the CUDA context before the first read of the private data directory; all of that is legitimately outside the timed region and it is stated as such. What is not legitimate is reading the private data before the clock that times that read, or writing an elapsed to reward_log.jsonl that the host did not measure. Both are zeros with one reason code each, not a slower ramp.

## Traps that catch agents that are not thinking carefully

- Establishes the correct six constants for a configuration at an early attempt, leaves them in the trajectory summary rather than encoding them into the carry store, and by attempt fifty has to re-derive them inside the timed region on most of the eight graded configurations, landing at the slow end of the ramp everywhere.
- Rewrites carry/notes.json fresh each attempt from that attempt's three observations, overwriting the accumulated table, so at most three entries are ever live and the eight-configuration final attempt is graded almost entirely on re-derivation.
- Dumps the full thirty-two-entry table as raw JSON floats, exceeds 2,048 bytes, and the attempt fails loudly with carry_store_invalid while the previous, smaller store is what carries forward, costing the attempt and teaching the wrong lesson if the reason code is not read.
- Never works out that the final attempt is graded on the eight oldest-visited configurations, refreshes the recently-visited entries instead, and arrives at the scored attempt with exactly the wrong half of the table live.
- Marks everything observed because observed sounds stronger, and watches the whole table expire on the eight-attempt clock while a predicted entry it never wrote would have survived.
- Trusts stored constants past the bound re-parameterization and trains two configurations against stale geometry and a stale prior, clearing neither bar and never diagnosing why, because the feedback carries no per-constant diagnostic.
- Mines the shipped decoy log corpus, resolves conflicting per-configuration records by majority vote, and stores constants from the retired parameterization, so its memory is confidently wrong rather than merely empty.
- Fabricates entries marked observed for configurations the run never graded and zeroes on carry_fabricated, when marking them predicted would have been admissible.
- Ports a published fast-CIFAR speedrun recipe whose schedule, batch size and augmentation are tuned for 32px inputs and never clears the bars at 96px on an image-disjoint STL-10 split.
- Downloads a pretrained ImageNet backbone or a self-supervised STL-10 encoder, fine-tunes it locally, and embeds or re-downloads it, then zeroes at grade time on source_payload_violation, seed_independent_parameters or isolation_violation.
- Hardcodes per-configuration constants into train.py that no graded run of this attempt's history observed, and zeroes on constants_unprovenanced rather than bypassing the 2,048-byte channel.
- Spends the fifty-attempt budget on architecture search when the reachable gains live in the calibration table, its encoding under the cap, and the refresh schedule.
- Clears every bar but only near the ceiling on every configuration, saturating the slow end of the ramp because the probe cost was never removed and the data pipeline was never profiled.
- Reorders work to move cost outside the timed region in a way that reads the private data before the declared point and trips the timer-order check.

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `train_program_lands_and_compiles` (compiled, weight 5, knockout) -- the program the verifier
executes lands at /submission/train.py, decodes as text and parses.

2. `program_consumes_every_flag_the_verifier_passes` (compiled, weight 5, knockout) -- the five
launch flags are accepted, and every graded configuration was launched with all five and its own
identifier.

3. `source_tree_carries_no_smuggled_weights` (compiled, weight 5, knockout) -- 128 KB, text only,
no numeric literal array above 1,024 elements, no base64 literal above 4 KB, recomputed from the
delivered bytes. Contract checker C5-SRC.

4. `hardcoded_per_configuration_constants_are_provenanced` (compiled, weight 3) -- a config_id
written into the source is backed by an src=observed carry entry. Contract checker C17-CONST.

5. `carry_store_fits_the_two_kilobyte_channel` (compiled, weight 5, knockout) -- 2,048 bytes,
UTF-8, JSON object, at most one entry per configuration. Contract checker C6-CARRY.

6. `carry_entries_carry_the_six_named_scalars_and_nothing_else` (compiled, weight 5, knockout) --
the closed schema: six named scalars, a one-bit src, an optional 32-byte tag, no array and no
base64. Contract checker C6-CARRY.

7. `agent_store_writes_no_last_confirmed_stamp` (compiled, weight 5, knockout) -- the submitted
store carries no freshness stamp at any depth. Contract checker C7-STAMP.

8. `observed_entries_name_configurations_this_run_graded` (compiled, weight 3) -- observed
entries name graded configurations and match the verifier's own measurement inside the published
per-field tolerances; predicted entries are exempt. Contract checker C10-CARRYDIV.

9. `stamps_are_verifier_written_and_expire_after_eight_attempts` (compiled, weight 3) -- every
delivered stamp names an attempt at which that configuration cleared its bar and is at most
eight attempts old; observed implies stamped. Contract checkers C7-STAMP and C8-EXPIRE.

10. `carry_lifecycle_validates_stamps_expires_persists_then_reports` (compiled, weight 3) -- the
bound lifecycle order holds and the reported expired set equals the recomputation. Contract
checker C8-EXPIRE.

11. `delivered_store_is_byte_identical_to_the_persisted_one` (compiled, weight 3) -- what was
persisted is what was delivered, by digest. Contract checker C16-PERSIST.

12. `timer_opens_at_the_first_private_data_read` (compiled, weight 5, knockout) -- the clock
opens at the first private-data read and closes at the export write, and the recorded elapsed is
the difference. Contract checker C14-ORDER.

13. `graded_roster_is_the_published_schedule_and_the_eight_oldest_at_the_last_attempt`
(compiled, weight 5) -- the graded roster is recomputed from the published rule rather than read
from the record.

14. `gate_outcomes_recompute_from_the_shard_accuracies` (compiled, weight 3) -- each gate outcome
is the one its own accuracies imply, on the bar and on the 0.015 sibling gap. Contract checkers
C1-ACC and C3-SIB.

15. `exported_parameters_move_with_the_verifier_supplied_seed` (compiled, weight 3) -- two
distinct verifier seeds produce distinct parameters that both clear the bar. Contract checker
C4-SEED.

16. `reward_log_agrees_with_the_harness_within_the_published_tolerances` (compiled, weight 3) --
0.05 on accuracy, 10 percent on elapsed. Contract checker C13-LOG.

17. `heldout_shards_and_canonical_carry_state_unwritten` (compiled, weight 5, knockout) -- no
write into the grading tree, the private shards or the canonical carry state; no socket; one
GPU. Contract checker C11-ISO.

18. `refresh_and_exploit_budget_declared_per_configuration` (compiled, weight 3) -- the agent's
own budget declaration, in configurations, seconds and refresh-versus-exploit attempts.

19. `bar_gated_wallclock_ramp_over_the_scored_attempt` (compiled, weight 10, knockout) -- the
mean bar-gated wall clock over the attempt's graded configurations through the published
descending logarithmic ramp. Contract checkers C2-TIME and C12-EXPORT enter it as gate outcomes.

Rejected routes, each bound to the control that measures it wrong:

- Lookup shortcut on the corpus: STL-10 is a public benchmark with public repositories and
published accuracies, and public fast-training speedrun code exists for other vision corpora.
-- controlled by: no published work optimizes wall-clock to a per-configuration bar on a private
thirty-two-configuration STL-10 re-derivation under a cross-attempt calibration protocol, so no
repository is a near-optimal solution to the stated objective. Each b_c is anchored above the
shipped starter's converged accuracy on that configuration and proven unreachable by that
starter at any budget inside the ceiling. Published STL-10 numbers use the public split,
unlimited time and often self-supervised pretraining on the 100,000-image unlabeled pool, none
of which survive the graded route, and none of them know any configuration's label orientation,
prior skew or crop geometry.

- Public benchmark collision: 1GC-7RC (arXiv 2605.17046) publishes an open GitHub harness whose
T2 is 200-class TinyImageNet from scratch, no pretrained weights, no internet, 40-120 minutes on
one GPU, which is also the OPT4 anchor's setup. -- controlled by: this slot deliberately avoids
TinyImageNet, the 200-class setup and the 40-120 minute band. The corpus is STL-10 at 96px, the
budget band is seconds rather than tens of minutes per configuration, and the graded object is a
program plus a byte-capped expiring calibration store re-executed from a verifier-supplied seed
across a rotating private configuration family, none of which the 1GC-7RC harness covers.

- Weight smuggling inside the source tree: a small network's parameters can be serialized as
literals in a Python file. -- controlled by: three independent binds. C5-SRC caps the source tree
at 128 KB, text only, with no numeric literal array above 1,024 elements and no base64 blob
above 4 KB, recomputed from the delivered bytes rather than declared; C4-SEED requires the graded
parameters to actually depend on the verifier-supplied init seed; and the per-configuration
label orientation, prior skew and crop geometry make constants fitted against the shipped public
split meaningless. instruction.md states all three rules plainly, so the agent is told rather
than only defeated.

- Turning the carry store into a weight store or an unbounded scratchpad, which would dissolve
the memory pressure by making the channel lossless. -- controlled by: C6-CARRY enforces a closed
schema of one entry per configuration with six named scalar fields plus a one-bit src field and
an optional 32-byte encoder tag, admits no array field and no base64, and caps the whole file at
2,048 bytes. Violation fails loudly with carry_store_invalid rather than being truncated or
silently graded down, so the cap is an interface the agent can see binding rather than a trap.

- Fabricating the calibration table by guessing or by lifting it from the shipped decoy corpus,
so that memory appears to compound without any observation having been made. -- controlled by:
C7-STAMP makes last_confirmed verifier-written and rejects any agent-supplied stamp field with
carry_stamp_forged; C10-CARRYDIV recomputes every entry marked observed against the verifier's
own measurement for configurations the run actually graded and zeroes on carry_fabricated.
Entries marked predicted are deliberately exempt, so honest generalization from the latent
structure is a legitimate and rewarded route while fabricated provenance is not.

- Using train.py's own source as an unbounded memory channel, hardcoding all thirty-two entries
each attempt and bypassing the 2,048-byte cap. -- controlled by: the sandbox is fresh from the
pinned image every attempt, so train.py is regenerated each attempt from the model's own
context, and that context is the lossy trajectory summary; control CTL-SUMMARY-ONLY measures
that route directly. C17-CONST additionally rejects a source carrying per-config_id constants
with no src=observed carry entry behind them, with reason constants_unprovenanced, so the route
is bounded by provenance rather than blocked by an interface trap.

- Timer gaming: performing setup work before the timer opens, or patching the clock. --
controlled by: the timer is host-side per configuration, opened at the first read of the private
data directory observed through a passthrough mount and closed at that run's export completion,
so the submission never reports it; C14-ORDER asserts the ordering of the observed syscall trace
and any inversion scores zero with reason timer_order_violation. Reading and parsing the carry
store, creating the CUDA context and importing libraries before the first private-data read is
legitimate and is stated as such in instruction.md.

Deliverable manifest: /submission/train.py (python_source), executed once per graded
configuration with --data, --out, --init-seed, --config-id and --carry, delivered together with
the optional carry store at carry/notes.json.

Control obligations recorded in seed/contract.yaml:

- `CTRL-NOOP` targets no train.py present / unchanged environment; recorded score 0.

- `NC-AMNESIC` targets THE archetype-dominance control: the reference with its carry store
forcibly emptied at the final attempt, so it pays the full discovery probe inside the timed
region on every configuration. If this control passes, memory is not load-bearing and the slot
is re-scoped rather than shipped; recorded score 0.3118, re-measured on STL-10 at Phase 1.

- `CTL-SUMMARY-ONLY` targets carry writes disabled, findings passed across attempts only through
the bound trajectory summarizer; recovering six scalars per configuration from prose is lossy;
recorded score 0.3563.

- `CTL-NAIVE-DUMP` targets writing the full 32-entry table as raw JSON floats, which exceeds
2,048 bytes and fails loudly with carry_store_invalid rather than being truncated; recorded
score 0.

- `CTL-STALE-CONST` targets the reference with its in-run consistency check removed, using
pre-drift stored constants blindly on the six re-parameterized configurations, all six of which
fall inside the final attempt's eight; recorded score 0.2725.

- `CTRL-STALE-PUBLIC` targets a program carrying weights, normalization constants or label
orientation fitted against the shipped public split; wrong on all eight graded configurations,
and additionally trips C4-SEED to exactly 0 if it skips training; recorded score 0.03.

- `CTL-HAYSTACK` targets seeding from the 6,400-line environment/logs/ decoy corpus of a retired
parameterization whose config_id names collide with the live family, rather than from the
agent's own graded observations; recorded score 0.03.

- `CTL-MAJORITY` targets resolving each configuration's constants by majority vote over the
decoy records, where five of every six records for a configuration agree on a wrong constant;
recorded score 0.03.

- `DECOY-RECIPE` targets adopting environment/notes/recipes.md, a plausible published-style 32px
fast-training recipe, verbatim; it trains fast but at the wrong geometry and misses the bars;
recorded score 0.05.

- `NC-GEOM` targets using the wrong per-configuration crop offset, whether hardcoded, mis-keyed
by config_id, or carried from the public split; misses bars on all eight; recorded score 0.03.

- `CTRL-LIST-ALL` targets re-deriving all six constants from scratch on every configuration on
every graded run, or training an ensemble of every candidate architecture; exceeds the
per-configuration ceiling and zeroes every configuration; recorded score 0.

- `CTRL-OBVIOUS-MAJORITY` targets a majority-class predictor, whose balanced accuracy is at
chance so the sub-band numerator is 0 on every configuration; recorded score 0.

- `CTRL-OBVIOUS-COPY` targets copying public split labels into the export; balanced accuracy
about chance on every private image-disjoint shard; recorded score 0.

- `CTRL-PCAM-JOIN` targets the egress shortcut of recovering the label orientation and crop
geometry by hash-joining served crops against the public archive in the egress-enabled sandbox
and hardcoding them. Private configurations are re-cropped at a scale in [0.62, 0.88] with
subpixel resampling and per-configuration photometric renormalization, destroying byte and
perceptual-hash identity with the public archive, and C17-CONST rejects any train.py carrying
per-config_id constants without a corresponding src=observed carry entry, with reason
constants_unprovenanced; recorded score 0.

- `FEAS-JDG-FLIP` targets buying r_det by a route the rubrics forbid: the reference at
r_det = 1.0 with exactly one rubric flipped reports min(1.0, 0.25) = 0.250; recorded score 0.25.

Recorded divergences between this bundle and its seed/contract.yaml record, none of them
resolved by authoring:

- Gap F67 stands. capacity_campaign repointed this slot from PatchCamelyon to STL-10 on
2026-08-18 and the slot record still binds PatchCamelyon in its shortcut register, its
contamination note and its feasibility sketch. This bundle is authored against STL-10 throughout,
per the record's own dataset and rescope_delta fields; the register entries above are restated
for STL-10 and the divergence is reported rather than silently reconciled.

- The record names checker C17-CONST in control CTRL-PCAM-JOIN and then omits it from its
sixteen-entry checker_obligations roster. Item 4 compiles it here, because a control that cites
a checker no roster carries is a control nothing enforces.

- The record's objective prose says full reward is reached at forty-five seconds while its
reward and oracle_proof blocks record the full-reward floor moving from 45 s to 60 s. Both
numbers were measured on PatchCamelyon and both are nulled by the repoint, so this bundle
publishes the ramp shape and leaves the two anchors to the Phase 1 measurement rather than
choosing between two stale numbers.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-f52bbbe19703ede56d5674eb4c5d7dc3
  slot1: AELLO-CANARY-23a009b98bd344917e87ef548b584156
  slot2: AELLO-CANARY-3c8086a68a27b36b1f2ae16874efe9da
  slot3: AELLO-CANARY-90b0dc8de7fd77d4eb5b7482318e3440
-->
