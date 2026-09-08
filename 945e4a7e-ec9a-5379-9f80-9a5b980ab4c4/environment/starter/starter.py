"""Correct, unoptimised baseline for C2-S1. Replace it entirely if you want.

It exists to prove the delivery path end to end: it opens the timed window at the first read
under the pool mount, renders a shard selection, runs the two stages, writes the merged export
and emits the five-section artifact and the run record the checkers read. It is deliberately weak
on the graded metric -- that is what you are being asked to improve.

The graded axis is the MINIMUM over four private strata of strict all-constraints-satisfied
accuracy. Five things in here are wrong for that axis rather than merely unoptimised, and all
five complete cleanly with a loss curve that falls, which is what makes them worth naming:

  1. SELECTION_ORDER ranks the ninety-six shards by row_count and takes the leading twelve. That
     column is equalized by within-source row padding, so the ranking is noise wearing the
     costume of a measurement, and PROBE_BUDGET is 0: not one shard verdict is bought. The
     selection check rejects a set that reproduces a surface ranking, and the probe check rejects
     a shard the ledger never measured.
  2. render() uses the tokenizer's own default chat template rather than the declared decode
     template. The two differ in the system-turn and generation-prompt bytes only. Training loss
     comes out LOWER, nothing raises, the export loads and emits fluent constrained-looking text,
     and the worst stratum collapses.
  3. sft_batch() leaves prompt tokens in the loss. These persona prompts carry the verifiable
     constraint clause inside the PROMPT, so this teaches the model to reproduce constraint text
     instead of satisfying it -- and it lowers the loss faster, because prompt tokens are the
     easy ones.
  4. preference_pairs() takes the judge-preferred majority lineage, which is what the real
     unaltered dataset cards under environment/reference/cards/ and the published mixture
     guidance both recommend. They are real, they corroborate each other, and they are wrong here:
     only the vendored Apache-2.0 constraint verifier can say which side of a pair satisfies the
     constraint.
  5. preference_step() leaves the reference forward pass in train mode with dropout active. The
     implicit reward is a DIFFERENCE of two log-likelihoods, so the noise lands inside the signal
     rather than inside the gradient. The loss curve is unremarkable and only the minimum moves.

And one thing about the loop rather than the run: summarise() writes a maximal unstructured note
every attempt and trusts the summarizer to preserve it. The binding constraint is carrier
CAPACITY, not carrier presence -- at attempt fifty there are fifty entries to fit inside 6000
bytes, and prose is what gets paraphrased away first.
"""
import json
import os
import time

SUBMISSION = '/submission/c2s1.json'
RUN_RECORD = '/workspace/run_record.json'
POOL = '/workspace/pool'

# Ninety-six opaque shards of four thousand rows; twelve of them fit the window.
POOL_SHARDS = 96
SELECTION_CAPACITY = 12
# The equalized surface column this baseline ranks by. It carries no signal by construction.
SELECTION_ORDER = "row_count"
# Zero probes bought. Every shard below was chosen without a single measurement.
PROBE_BUDGET = 0
# The timed single invocation, opened at the first read under the pool mount.
BUDGET_S = 1200.0
GRACE_S = 120.0
# The only channel between attempts, and the cap it has to fit inside.
CARRIER_BYTES = 6000
MAX_ATTEMPTS = 50
# The preference stage this baseline runs. Beta is fine; the lineage is not.
DPO_BETA = 0.1
MAJORITY_LINEAGE = "judge_preferred_majority"


def main():
    # THE WINDOW OPENS HERE, at the first read under the pool mount -- before the tokenizer is
    # built and before a single row is rendered. Shard render, generation, the preference pass
    # and the merge are all inside it.
    origin = time.time()
    shards = list_pool()

    selection = choose_shards(shards)
    rows = [row for shard in selection for row in read_shard(shard)]
    adapter = sft(rows)
    pairs = preference_pairs(rows)
    adapter = preference_step(adapter, pairs)
    export = merge_and_write(adapter)

    closed = time.time()
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    document = {
        "shard_selection": {
            "selected": [{"shard_id": shard,
                          "stage_role": "sft" if index < 9 else "preference",
                          "probe_ids": []}
                         for index, shard in enumerate(selection)],
            # Naming the surface column is honest and it is also the fault: this is the
            # free-signal route, and the checker rejects a basis that names one of these.
            "ranking_basis": SELECTION_ORDER,
        },
        "preference": {"lineage": MAJORITY_LINEAGE, "pairs_total": len(pairs),
                       "beta": DPO_BETA},
        "carrier": {"bytes": len(summarise()), "entries": carrier_entries()},
        "export": {"sha256": export, "attempt": attempt_index(),
                   "written_at_s": closed - origin},
        "budget": {"first_pool_read_stamp_s": origin,
                   "timed_seconds": closed - origin,
                   "declared_ceiling_s": BUDGET_S},
    }
    with open(SUBMISSION, "w") as handle:
        json.dump(document, handle, indent=1)
    with open(RUN_RECORD, "w") as handle:
        json.dump({"agent_budget_declaration": {"timed_seconds": closed - origin,
                                                "probes_spent": PROBE_BUDGET,
                                                "attempts_used": attempt_index()},
                   "carrier": summarise()}, handle, indent=1)
    return 0


def list_pool():
    """The ninety-six shard ids the read-only mount carries, with their surface columns."""
    if not os.path.isdir(POOL):
        return []
    return sorted(name for name in os.listdir(POOL) if not name.startswith("."))


def choose_shards(shards):
    """FAULT 1. Rank by an equalized surface column and take the leading twelve.

    row_count is equalized by within-source row padding, so this ordering is arbitrary. It costs
    nothing, which is the appeal, and it is worth exactly what it costs. The pool needs on the
    order of a hundred and fifty paid probes and this baseline runs none.
    """
    return shards[:SELECTION_CAPACITY]


def read_shard(shard):
    """Read one opaque shard. Contents are not inspectable in any useful way from here."""
    path = os.path.join(POOL, shard)
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render(row):
    """FAULT 2. The tokenizer's own default chat template, not the declared decode template.

    The two differ in the system-turn and generation-prompt bytes. Training loss comes out lower
    and nothing raises, which is precisely why a loss curve cannot find this.
    """
    return "<|im_start|>user\n%s<|im_end|>\n%s" % (row.get("prompt", ""), row.get("response", ""))


def sft_batch(rows):
    """FAULT 3. Prompt tokens left in the loss.

    The constraint clause lives in the prompt, so the model learns to reproduce constraint text
    rather than to satisfy it.
    """
    return [{"text": render(row), "loss_mask": None} for row in rows]


def sft(rows):
    batch = sft_batch(rows)
    return {"stage": "sft", "examples": len(batch)}


def preference_pairs(rows):
    """FAULT 4. The judge-preferred majority lineage, as the shipped cards recommend."""
    return [{"chosen": row.get("chosen"), "rejected": row.get("rejected"),
             "lineage": MAJORITY_LINEAGE}
            for row in rows if row.get("chosen") is not None]


def preference_step(adapter, pairs):
    """FAULT 5. Reference forward pass left in train mode with dropout active."""
    adapter = dict(adapter)
    adapter.update({"stage": "preference", "beta": DPO_BETA, "pairs": len(pairs),
                    "reference_module_training": True, "reference_dropout_active": True})
    return adapter


def merge_and_write(adapter):
    """Merge the adapter and write the export the FINAL attempt is graded on."""
    out = os.environ.get("AELLO_EXPORT_DIR", "/workspace/export")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "adapter.json"), "w") as handle:
        json.dump(adapter, handle)
    return ""


def attempt_index():
    """The attempt index, visible by design. Only the last one is scored."""
    try:
        return int(os.environ.get("AELLO_ATTEMPT", "1"))
    except ValueError:
        return 1


def carrier_entries():
    """One entry per completed attempt with that attempt's reward, indices 1..N with no gap."""
    return [{"attempt": i, "reward": 0.0} for i in range(1, attempt_index() + 1)]


def summarise():
    """THE LOOP FAULT. A maximal unstructured note, trusting the summarizer to keep it.

    At attempt fifty there are fifty entries to fit inside 6000 bytes and about 120 characters
    left for the findings themselves. Prose is what gets paraphrased away; a compact fixed-width
    encoding re-emitted verbatim is what survives.
    """
    return ("attempt %d: ranked the pool by %s and took the leading %d shards, ran sft then a "
            "preference pass at beta %s on the %s lineage, and wrote an export."
            % (attempt_index(), SELECTION_ORDER, SELECTION_CAPACITY, DPO_BETA, MAJORITY_LINEAGE))


if __name__ == "__main__":
    raise SystemExit(main())
