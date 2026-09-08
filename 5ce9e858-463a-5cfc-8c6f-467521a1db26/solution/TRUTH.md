# AELLO-C8-S9 -- what this task actually tests

GENERATED SECTION. DO NOT HAND-EDIT.

minimise normalised forecast error on a private future window at a fixed prediction horizon and a fixed lookback budget. The graded artifact is float array (n_windows, horizon, n_targets), written to /submission/forecast.npy. The graded quantity is Normalised mean absolute error over the private future window, mapped to [0,1] higher-better.

## The single most important insight

The reward is deterministic under frozen bytes and is zero on any red line before any quality is considered, so the first thing that matters is that the artifact exists, has the declared shape, and was produced without touching the checker tree or the graded fold. Only then does Normalised mean absolute error over the private future window, mapped to [0,1] higher-better move the score.

## The ideal solve, step by step

1. **Normalisation scheme against distribution drift** A lever the contract records as separately ablatable for this slot; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

2. **Lookback allocation across the seven covariates** A lever the contract records as separately ablatable for this slot; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

3. **Seasonal decomposition before the learned component** A lever the contract records as separately ablatable for this slot; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

4. **Loss shaping for the tail of the horizon** A lever the contract records as separately ablatable for this slot; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

5. **Validation-split policy that respects time order** A lever the contract records as separately ablatable for this slot; its worth is measured on the Phase 2 ladder over the full lever power set, never asserted here.

## Traps that catch agents that are not thinking carefully

- Shuffles the split and reports an error the future window cannot reproduce.
- Normalises with statistics computed over the whole series including the future.
- time-order leakage is the dominant way to inflate this metric Mitigation: CTL-LEAK is a compiled red-line control and a shuffled split scores exactly 0

---

## Contract record (annex; not part of the narrative body)

Retained under FORGE.md item 10e and Phase 2 item 7, which require this file to carry the ordered path through instruction.md with each satisfied checker identifier, and each rejected route bound to a measured known-wrong control. standards/truth-md-authoring-v1.md section 3 admits no fifth section, so this annex is a recorded deviation rather than an omission.

Ordered path through instruction.md, with the checker identifier each step satisfies:

1. `budget_declared` (compiled, weight 3) -- The agent declares its own budget accounting for the graded run.
2. `deliverable_present` (compiled, weight 5, knockout) -- The agent writes the graded artifact at the declared submission path.
3. `deliverable_shape` (compiled, weight 5, knockout) -- The graded artifact has the declared shape and domain.
4. `deterministic_replay` (compiled, weight 3) -- The graded artifact is reproducible from the recorded seed.
5. `graded_metric` (compiled, weight 10, knockout) -- The graded metric clears the published bar and is mapped through the ramp.
6. `no_graded_writes` (compiled, weight 5, knockout) -- The graded run writes nothing into the checker tree or the graded fold.

Rejected routes, each bound to the control that measures it wrong:

- time-order leakage is the dominant way to inflate this metric -- controlled by: CTL-LEAK is a compiled red-line control and a shuffled split scores exactly 0

Deliverable manifest: /submission/forecast.npy (npy), float array (n_windows, horizon, n_targets).

Control obligations recorded in seed/contract.yaml:

- `CTL-NOOP` targets the shipped starter verbatim; recorded score None.
- `CTL-PERSIST` targets persistence forecast, repeat the last observation; recorded score None.
- `CTL-LEAK` targets a split that shuffles across time; must be rejected; recorded score None.

<!-- AELLO-CANARY-BLOCK
  slot0: AELLO-CANARY-3338d48cf6aaaab9fed70d3999c4eef3
  slot1: AELLO-CANARY-59f56b16c9be9274bfe9c2f55513d4ee
  slot2: AELLO-CANARY-3f73f40ec5070d2d02b1d437e606d45b
  slot3: AELLO-CANARY-d1e3166e80845b74480c81931491d895
-->
