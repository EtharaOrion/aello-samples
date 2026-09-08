"""Suboptimal starter: a length-margin pairwise judge that ANSWERS EVERY ITEM.

It clears no constraint that matters. Answering everything makes the risk cap the binding
condition and the measured selective risk of a length prior on these dialogues sits far
above alpha, so the run scores exactly zero with reason risk-cap-exceeded. Turning this
into a judge that knows WHEN NOT TO ANSWER, with a risk cap that holds on two shards whose
sub-distribution mixtures you are not told, is the task.

Reads the graded items, writes one verdict per item to /submission/verdicts.json.
"""
import json, os, sys, time


def read(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    env = ("/workspace/environment" if os.path.isdir("/workspace/environment")
           else os.path.join(os.path.dirname(__file__), ".."))
    out = sys.argv[1] if len(sys.argv) > 1 else "/submission"
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    ids, verdicts = [], []
    for shard in ("p1", "p2"):
        for item in read(os.path.join(env, "data", "%s_items.jsonl" % shard)):
            ids.append(item["id"])
            # A length prior, applied to every item: no abstention, no calibration.
            verdicts.append("A" if len(item["response_a"]) >= len(item["response_b"]) else "B")
    json.dump({"ids": ids, "verdicts": verdicts, "elapsed_s": time.perf_counter() - t0},
              open(os.path.join(out, "verdicts.json"), "w"))
    print("starter: wrote %d verdicts, 0 abstentions" % len(ids))


if __name__ == "__main__":
    main()
