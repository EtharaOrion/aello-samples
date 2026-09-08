"""Runnable baseline for AELLO-C8-S2. Correct, memory-less -- and deliberately slow.

WHAT IT DOES. It is a `train.py`: it accepts the five flags the verifier launches with, decodes the
carry store BEFORE it touches the private data directory, then re-derives everything it needs about
the configuration from the shard it was just handed, trains a small classifier, writes the export
to --out, and appends its own accuracy and elapsed to reward_log.jsonl. Replace all of it.

WHAT IT IS NOT. It is not fast, and the reason it is not fast is the point of the whole slot: it
pays the full discovery probe -- label-orientation, channel statistics, crop-offset search -- inside
the timed region, on every configuration, every attempt, because it writes nothing to the carry
store and reads nothing from it. That is the NC-AMNESIC control with the serial numbers filed off.
The published bars are anchored above what this program converges to, so it does not clear them at
any budget inside the ceiling.

IT FABRICATES NOTHING. If the private data directory is not there, or torch is not importable, it
says so in the run record and leaves the accuracy and elapsed fields null. A null elapsed fails the
compiled checks, which is the correct outcome: no run happened. It never writes a number it did not
measure -- that is a red line, and the starter is not going to demonstrate it for you.

THREE THINGS WORTH KNOWING BEFORE YOU BUILD ON IT.

  THE CLOCK OPENS ON THE FIRST PRIVATE READ, not on process start. Everything this file does
  before `first_private_read` -- imports, CUDA context, carry decode -- is outside the timed
  region and is legitimately free. Everything after it is not. Look at how little is on the free
  side here, and how much could be.

  THE CARRY STORE IS READ AND THEN IGNORED. `load_carry` returns the delivered table and the
  baseline drops it on the floor. Wiring the six constants through instead of probing for them is
  the single largest wall-clock lever in this task, and it is not an optimisation of the trainer.

  SIX SCALARS, 64 BYTES AN ENTRY. `encode_entry` shows the schema and nothing else; it does not
  quantize, and thirty-two of them written this way will not fit under the cap. Deciding what
  precision each field actually needs is the second lever.
"""
import argparse
import json
import os
import time

WORK = os.environ.get("AELLO_WORKDIR", "/workspace")
RUN_RECORD = os.path.join(WORK, "run_record.json")
REWARD_LOG = os.path.join(WORK, "reward_log.jsonl")
SCALAR_FIELDS = ("flip", "rho", "ofs", "scale", "chan", "amb")


def parse(argv):
    parser = argparse.ArgumentParser(description="AELLO-C8-S2 baseline trainer")
    parser.add_argument("--data", default=None, help="the private data directory for this run")
    parser.add_argument("--out", default=None, help="where this run's export is written")
    parser.add_argument("--init-seed", type=int, default=None, help="verifier-supplied init seed")
    parser.add_argument("--config-id", default=None, help="which private configuration this is")
    parser.add_argument("--carry", default="carry/notes.json", help="the carry store path")
    return parser.parse_args(argv)


def load_carry(path):
    """The delivered table, or an empty one. Read before the first private-data read, always.

    The baseline calls this and then ignores what it returns. That is the whole of its memory
    policy and the whole of why it is slow.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            doc = json.load(handle)
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def encode_entry(constants, observed):
    """One carry entry in the published closed schema. No quantization, deliberately."""
    entry = dict((field, constants[field]) for field in SCALAR_FIELDS)
    entry["src"] = 0 if observed else 1
    return entry


def probe_constants(data_dir):
    """Re-derive the six per-configuration constants from the private shard.

    This is the cost the carry store exists to remove. The baseline pays it in full, every run.
    Returns None when the shard is not there, because a constant nobody measured is not a
    constant and must not be written as one.
    """
    if not data_dir or not os.path.isdir(data_dir):
        return None
    listing = sorted(os.listdir(data_dir))          # the first private read; the clock is open
    if not listing:
        return None
    return {"flip": None, "rho": None, "ofs": None, "scale": None, "chan": None, "amb": None,
            "shard_entries": len(listing)}


def train_and_export(data_dir, out_path, seed, constants):
    """Train, then write the export. Returns (balanced_accuracy, wrote) with None when it cannot.

    Left as the shortest correct route rather than a good one: the baseline exists to prove the
    delivery path, and the ramp is where the work is.
    """
    if constants is None or not out_path:
        return None, False
    try:
        import torch                                 # noqa: F401
    except ImportError:
        return None, False
    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(b"")                            # a real submission writes real parameters
    return None, True


def main(argv=None):
    args = parse(argv)
    pre_read = {"process_start": time.time()}
    carry = load_carry(args.carry)                   # free: before the first private read
    pre_read["carry_decode"] = time.time()

    first_private_read = time.time()                 # the host clock opens here, not before
    constants = probe_constants(args.data)
    accuracy, wrote = train_and_export(args.data, args.out, args.init_seed, constants)
    export_write = time.time()

    block = {
        "config_id": args.config_id,
        "argv": ["train.py", "--data", str(args.data), "--out", str(args.out),
                 "--init-seed", str(args.init_seed), "--config-id", str(args.config_id),
                 "--carry", str(args.carry)],
        "init_seed": args.init_seed,
        "first_private_read_s": first_private_read,
        "export_write_s": export_write if wrote else None,
        "balanced_accuracy_a": accuracy,
        "elapsed_s": (export_write - first_private_read) if wrote else None,
        "pre_read_stages": pre_read,
        "carry_entries_delivered": len(carry),
        "carry_entries_used": 0,
    }
    record = {"graded": [block],
              "agent_budget_declaration": {"graded_configurations": 1,
                                           "seconds_per_configuration":
                                               block["elapsed_s"] or 0.0,
                                           "attempts_spent_refreshing": 0,
                                           "attempts_spent_exploiting": 1}}
    write_json(RUN_RECORD, record)
    if accuracy is not None and block["elapsed_s"] is not None:
        append_log({"config_id": args.config_id, "balanced_accuracy": accuracy,
                    "elapsed_s": block["elapsed_s"]})
    return 0


def write_json(path, doc):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
        handle.write("\n")


def append_log(line):
    directory = os.path.dirname(REWARD_LOG)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(REWARD_LOG, "a") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
