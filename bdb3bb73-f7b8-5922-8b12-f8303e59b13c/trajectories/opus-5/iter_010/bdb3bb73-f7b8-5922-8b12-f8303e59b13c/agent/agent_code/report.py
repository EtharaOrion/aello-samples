"""Aggregate sweep runs: mean/std of final holdout accuracy per config tag."""
import json
import sys
from collections import defaultdict

import statistics as st

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/logs/sweep.jsonl"
prefix = sys.argv[2] if len(sys.argv) > 2 else ""

groups = defaultdict(list)
best_ep = defaultdict(list)
for line in open(path):
    r = json.loads(line)
    tag = r["tag"].rsplit("_s", 1)[0]
    if prefix and not tag.startswith(prefix):
        continue
    groups[tag].append(r["final_acc"])
    best_ep[tag].append(max(h["acc"] for h in r["history"]))

rows = []
for tag, accs in groups.items():
    m = st.mean(accs)
    s = st.stdev(accs) if len(accs) > 1 else 0.0
    rows.append((m, s, len(accs), st.mean(best_ep[tag]), tag))
rows.sort(reverse=True)
print(f"{'config':<34} {'n':>2} {'mean':>7} {'std':>6} {'sem':>6} {'best_ep':>7}")
for m, s, n, be, tag in rows:
    sem = s / (n ** 0.5) if n > 1 else 0.0
    print(f"{tag:<34} {n:>2} {m:>7.4f} {s:>6.4f} {sem:>6.4f} {be:>7.4f}")
