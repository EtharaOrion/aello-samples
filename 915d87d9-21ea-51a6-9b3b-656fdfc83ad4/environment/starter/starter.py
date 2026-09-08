"""Correct, unoptimised baseline recipe for C4-S3. Replace it entirely if you want.

It exists to prove the delivery path end to end: it walks the pinned checkpoint's 196 linear
projections, assigns ONE uniform format to all of them, packs, and writes the manifest, the
recipe and the shards at the declared paths. The format it picks is int4_group256_sym, which
prices at 4 + 16/256 = 4.0625 bits per weight and therefore fits the 4.10 budget -- the point of
the starter is to be admissible, not to be good.

It is deliberately weak on the graded metric in two ways at once, and both are the thing you are
being asked to fix. It runs no clipping-range search, so its perplexity ratio sits at the 1.185
that defines the no-gain band; and it spends the same 4.0625 bits on every projection, so it
neither buys speed with INT3 where the network can afford it nor protects quality with FP8 where
it cannot. Finding out which projections are which is what the 8-minute calibration budget is for.

What this file writes and what the harness writes are different things. The manifest, the recipe,
the shards and the calibration accounting are yours. Every measured quantity in run_record.json --
the decode trial table, the perplexities on the two private slices, the traced stage order, the
per-layer dequant-invocation counters, the harness wall clock, the egress record and the
re-execution result -- is written by the harness on the grading host and is never read from
anything a recipe says about itself.
"""
import json
import math
import os

EXPORT_ROOT = "/submission/quant_export"
MANIFEST = os.path.join(EXPORT_ROOT, "manifest.json")
SHARD_DIR = os.path.join(EXPORT_ROOT, "shards")
RECIPE = os.path.join(EXPORT_ROOT, "recipe.py")
RUN_RECORD = "/workspace/run_record.json"

# The pinned checkpoint, by its own shapes. hidden 1536, intermediate 8960, 12 attention heads and
# 2 key-value heads at head_dim 128, over 28 blocks.
BLOCKS = 28
WEIGHT_COUNTS = {
    "self_attn.q_proj": 1536 * 1536,
    "self_attn.k_proj": 1536 * 2 * 128,
    "self_attn.v_proj": 1536 * 2 * 128,
    "self_attn.o_proj": 1536 * 1536,
    "mlp.gate_proj": 1536 * 8960,
    "mlp.up_proj": 1536 * 8960,
    "mlp.down_proj": 1536 * 8960,
}
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

UNIFORM_FORMAT = "int4_group256_sym"
UNIFORM_GROUP = 256
UNIFORM_BITS = 4 + 16.0 / UNIFORM_GROUP          # 4.0625, under the 4.10 budget

# The order this recipe declares IS the order it executes. Declaring an order you do not run, or
# an order that is decorative, is what the ordering and regeneration checks are looking for.
STAGES = ["clipping_range_search", "rounding", "group_folding", "packing"]
CALIBRATION_SECONDS = {"clipping_range_search": 0.0, "rounding": 0.0, "group_folding": 0.0,
                       "packing": 0.0, "total": 0.0}


def allocate():
    """One format for every projection. A knapsack over measured per-layer sensitivity goes here."""
    rows = []
    for block in range(BLOCKS):
        for projection in PROJECTIONS:
            weights = WEIGHT_COUNTS[projection]
            rows.append({
                "name": "model.layers.%d.%s" % (block, projection),
                "weight_count": weights,
                "format": UNIFORM_FORMAT,
                "group_size": UNIFORM_GROUP,
                "bits_per_weight": UNIFORM_BITS,
                "packed_bytes": int(math.ceil(weights * UNIFORM_BITS / 8.0)),
                "shard": "weights-%02d.bin" % block,
            })
    return rows


def pack(rows):
    """Write one shard per block and report the bytes the filesystem actually holds.

    The average bits per weight is recomputed from these files, not from the manifest, so the
    manifest is written FROM the packer's output rather than from the allocation that was intended.
    """
    os.makedirs(SHARD_DIR, exist_ok=True)
    sizes = {}
    for row in rows:
        sizes[row["shard"]] = sizes.get(row["shard"], 0) + row["packed_bytes"]
    shards = []
    for name in sorted(sizes):
        with open(os.path.join(SHARD_DIR, name), "wb") as handle:
            handle.write(b"\0" * sizes[name])
        shards.append({"file": name, "bytes": os.path.getsize(os.path.join(SHARD_DIR, name))})
    return shards


def main():
    os.makedirs(EXPORT_ROOT, exist_ok=True)
    rows = allocate()
    shards = pack(rows)
    packed_bits = sum(shard["bytes"] for shard in shards) * 8.0
    weights = sum(row["weight_count"] for row in rows)
    with open(MANIFEST, "w") as handle:
        json.dump({
            "recipe_stages": list(STAGES),
            "layers": rows,
            "shards": shards,
            "declared_bits_per_weight": packed_bits / weights,
            "calibration_seconds": dict(CALIBRATION_SECONDS),
            "upstream_checkpoint": {"revision": None, "weight_digest": None},
        }, handle, indent=1)
    with open(RECIPE, "w") as handle:
        handle.write("# The recipe the harness re-executes. This baseline is reproducible because\n"
                     "# it makes no measurement at all: one format, every layer, no search.\n"
                     "from starter import main\n\nif __name__ == '__main__':\n    raise "
                     "SystemExit(main())\n")
    os.makedirs(os.path.dirname(RUN_RECORD), exist_ok=True)
    with open(RUN_RECORD, "w") as handle:
        json.dump({"agent_budget_declaration": {"attempts_used": 1,
                                                "calibration_seconds_declared": 0.0}},
                  handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
