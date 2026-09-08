# AELLO-C8-S7

maximise keyword accuracy on a private split while every inference stays inside a bound per-utterance latency.

## What is graded

Write your keyword decisions to **`/submission/preds_c8s7.npy`**: a one-dimensional array of **integer** dtype, **3500 rows**, one per graded utterance, in the order the graded roster ships them. Every value is a class index in `0..34`. A missing, malformed, misshapen or out-of-domain artifact scores 0.

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself. The graded quantity is **top-1 keyword accuracy over the private split**, and it is **forced to exactly `0.0` if any single one of the 3500 inferences the harness times exceeds the per-utterance latency bound**.

Read that second clause twice. The gate is a **maximum**, not a mean and not a percentile. 3499 fast inferences do not pay for one slow one, and the gate is applied before the labels are compared, so accuracy bought by exceeding the bound is worth nothing rather than worth most of what it scored.

`r_det` is zero on any red line and on any knockout failure, before any accuracy is considered.

## The latency bound

| | |
| --- | --- |
| per-utterance bound | **20.0 ms** of wall clock for one inference over one 1000 ms utterance |
| statistic | **maximum** over all 3500 graded utterances |
| clock | the **harness's**. Yours is red line 2 |
| host class | eight CPU cores, no GPU -- `[verifier.environment]` in `task.toml` |

The bound is a published specification, fixed before any measurement on this slot, not a number measured from a baseline. It binds where this task's levers live: a raw-waveform branch at 16 kHz, 80 mel bands at a 64-sample hop, and a fused two-branch stack each cost real milliseconds on that class.

## The corpus

Google Speech Commands v0.02 (Warden 2018). One second of 16 kHz audio per utterance -- exactly **16000 samples**, shorter clips zero-padded at landing.

**The graded split is balanced 1.00:1**: 100 utterances for each of the 35 keywords, 3500 in total. Accuracy is therefore plain accuracy and needs no per-class weight; it also means no keyword can be won by prior alone.

**Class index 0 is `backward` and class index 34 is `zero`.** The roster is the 35 keyword folders in ascending lexicographic order:

```
 0 backward   1 bed        2 bird       3 cat        4 dog
 5 down       6 eight      7 five       8 follow     9 forward
10 four      11 go        12 happy     13 house     14 learn
15 left      16 marvin    17 nine      18 no        19 off
20 on        21 one       22 right     23 seven     24 sheila
25 six       26 stop      27 three     28 tree      29 two
30 up        31 visual    32 wow       33 yes       34 zero
```

Declare this mapping in your run record. It is the one mistake on this corpus that hides completely: a model trained against its own label ordering emits a perfectly well-formed vector of in-range integers that scores near 1/35, and neither the array nor the metric can tell a permuted mapping from a model that did not learn. The twelve-class formulation of this corpus -- ten commands plus `unknown` plus `silence` -- is **not** what is graded here; an index of 35 or of -1 names no keyword.

## What ships

- `environment/data/speech_commands_v002_train.npz` -- the training audio and its labels.
- `environment/data/graded_utterances.npz` -- the 3500 graded utterances, **unlabelled**, with `utterance_id` in the graded order. These are the clips you predict on and the clips the harness times you on.
- `environment/data/keyword_index.json` -- the roster above, machine-readable.
- `environment/data/background_noise/` -- the six background recordings the release ships: `doing_the_dishes`, `dude_miaowing`, `exercise_bike`, `pink_noise`, `running_tap`, `white_noise`.
- `environment/starter/starter.py` -- a correct, unoptimised baseline you may replace entirely.

The keyword **labels** for the graded split ship only under `tests/heldout/` and the graded run must not touch them. The audio cannot be hidden -- latency is a property of running your model over it -- and the labels cannot be shown.

## The front end is a declarable lever

Window, hop and mel band count set both your accuracy and most of your cost. Declare them, inside this envelope:

| lever | admitted range | in milliseconds at 16 kHz |
| --- | --- | --- |
| `window_samples` | 128 to 1024 | 8 ms to 64 ms |
| `hop_samples` | 32 to 512 | 2 ms to 32 ms |
| `mel_bands` | 10 to 80 | -- |

Three joint constraints are checked, not just the box:

- `frames_per_utterance` must equal `1 + (16000 - window_samples) // hop_samples`. Halving the hop doubles the frames and roughly doubles the per-utterance convolution cost.
- `hop_samples` must not exceed `window_samples`. Audio between the end of one frame and the start of the next is never read, and a keyword onset can land entirely in that gap.
- `mel_bands` must not exceed `window_samples // 2 + 1`, the positive-frequency bin count. More mel filters than bins produces empty filters that cost latency and carry no signal.

`sample_rate_hz` is the corpus's 16000.

## The run record

Write `/workspace/run_record.json` during the graded run. The verifier reads it and the compiled checks grade it, so an absent or incomplete record costs score on its own. Two of its keys are written by the **harness**, not by you.

| key | what it must carry |
| --- | --- |
| `keyword_index` | the ordered 35-word roster your class indices mean |
| `front_end` | `sample_rate_hz`, `window_samples`, `hop_samples`, `mel_bands`, `frames_per_utterance` |
| `front_end_digest_at_timing` | sha256 over the canonical JSON of that block (sorted keys, `,`/`:` separators), taken on the timed pass |
| `front_end_digest_at_prediction` | the same digest, taken on the scored pass. The two must be equal and must be the digest of the block you declared |
| `latency_source` | `harness_per_utterance_timer`. `agent_reported_wall_clock`, `self_timed_batch_mean`, `python_time_around_forward` and `amortised_throughput_estimate` are rejected by name |
| `latency_gate_passed` | your **claim**, formed from your own development profiling, that every graded inference lands inside the bound. It is recomputed from the harness maximum, and a claim the measurement contradicts fails -- including the case where your claim is true of the mean and false of the maximum |
| `harness_utterance_latency_ms` | **written by the harness**: one measured millisecond figure per graded `utterance_id` |
| `graded_audio_digest_before`, `graded_audio_digest_after` | the digest of the graded utterance archive before your first inference and after your last. They must be equal |
| `paths_written` | every path the graded run wrote |
| `guarded_paths_touched` | **written by the harness** |
| `augmentation` | `applied_to` (`train`, or `none` with an empty clip list), `background_clips` drawn from the vendored six, `snr_db_range` inside 0 to 20 dB |
| `replay` | `artifact_sha256` and `front_end_digest` of a second run under the recorded seed. The artifact digest must match your submission byte-for-byte; the front-end digest must match the graded run's |
| `agent_budget_declaration` | `train_seconds`, `inference_utterances_timed`, `attempts_used`, each a non-negative number |

Do **not** write `agent_utterance_latency_ms`, `self_timed_latency_ms` or `reported_latency_ms` into the record. Offering your own milliseconds for grading is red line 2 whatever the harness does with them.

## Why the gate is per utterance and not per batch

A deployed keyword spotter wakes, decides, and sleeps. It pays the per-utterance cost every time, and it has no batch to amortise across. So:

- Batching all 3500 utterances into one forward pass and dividing by 3500 amortises away exactly the cost the bound constrains.
- A mean over 3500 utterances hides an arbitrarily slow inference; a 99th-percentile gate passes 35 of them by construction.
- One configuration has to serve both measurements. Timing a 20-band, hop-320 front end and then scoring an 80-band, hop-64 one is two measurements of two models, and the two digests above are what closes it.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement. Declare `inference_utterances_timed` honestly: latency here is a property of the design, and a front end tuned against a dozen clips has not been measured against a gate it must clear on 3500.

## Red lines

Any of these scores zero:

- Any write into `tests/` or into the graded utterance archive during the graded run. Re-mixing the graded clips at a friendlier signal-to-noise ratio, or trimming them below the second, raises accuracy and lowers latency at once by changing the input rather than the model -- and the archive digest moves.
- Measuring latency with a clock the agent controls. A duration reported by the process being timed is not a measurement of that process: a batch mean is not the per-utterance cost, and a timer opened after the front end has already run charges nothing for the front end. Profile your design during development however you like -- that is how you form the `latency_gate_passed` claim -- but the graded number is the harness's, and an agent-timed latency field in the graded run record is the red line whatever the harness does with it.

## Traps

- Fuses at the logit level, so the two branches never share a representation. The waveform branch then pays its full inference cost against the gate while contributing only a vote.
- Meets accuracy by exceeding the latency bound, and the gate zeroes the attempt.
- Reads the gate as an average or a high percentile.
- Times a batched forward over the whole split and divides by the row count.
- Times a cheap front end and scores an expensive one.
- Trains against its own label ordering, and reports a modelling failure that was a mapping failure.
- Declares a hop wider than its window, or more mel bands than the window has positive-frequency bins.
- Mixes background noise into the graded utterances rather than the training split.
- a pretrained audio encoder would shortcut the fusion design Mitigation: no model-zoo host is reachable at solve time; any backbone must be vendored

## A note on what has not landed yet

The audio has not been vendored into this tree at the time of writing: `environment/data/` and `tests/heldout/` do not exist. The paths, the roster, the counts, the sample rate and the bounds published above are the layout the corpus must land against, recorded in `solution/grounding.yaml` under `corpus_layout`, and every compiled check that needs a file it cannot find reports that it could not find it rather than passing over its absence.
