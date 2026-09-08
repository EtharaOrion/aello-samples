"""Suboptimal starter: a length-heuristic pairwise judge. It clears neither the adversarial
stratum floor (it prefers the more verbose response, which is exactly the fluency prior the
task penalizes) nor, without care, the swap cap. Improving it into a fluency-decoupled scorer
is the task. Reads the graded items, writes one verdict per item to /submission/verdicts.json.
"""
import json, os, sys, time

def read(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def main():
    env = "/workspace/environment" if os.path.isdir("/workspace/environment") else os.path.join(os.path.dirname(__file__), "..")
    out = sys.argv[1] if len(sys.argv) > 1 else "/submission"
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    items = read(os.path.join(env, "data", "heldout_items.jsonl")) + read(os.path.join(env, "data", "sibling_items.jsonl"))
    ids, verdicts = [], []
    for it in items:
        ids.append(it["id"])
        verdicts.append(1 if len(it["response_1"]) > len(it["response_2"]) else 0)
    json.dump({"ids": ids, "verdicts": verdicts, "elapsed_s": time.perf_counter() - t0},
              open(os.path.join(out, "verdicts.json"), "w"))

if __name__ == "__main__":
    main()
