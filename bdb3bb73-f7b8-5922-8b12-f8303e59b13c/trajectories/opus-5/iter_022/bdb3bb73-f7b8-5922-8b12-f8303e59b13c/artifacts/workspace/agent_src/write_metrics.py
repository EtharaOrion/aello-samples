"""Write /workspace/agent_reported_metrics.json from the graded run.

`wall_clock_sec` is the total wall clock of the graded training run measured
conservatively: from process start (data load + GPU warm-up included) through the
last checkpoint save.  It is therefore strictly larger than the span between the
first and last checkpoint.  `accuracy` is the final checkpoint's top-1 as measured
by verify.py through the verifier's own pipeline.
"""
import json
import os
import sys

CKPT = "/workspace/periodic_checkpoints"

run_log = json.load(open("/workspace/run_log.json"))
rows = json.load(open("/workspace/verify_report.json"))

span = max(r["mtime"] for r in rows) - min(r["mtime"] for r in rows)
wall = run_log["wall_clock_sec"]
total = run_log.get("total_wall_clock_sec", wall)
assert total >= span, f"reported wall {total} < checkpoint span {span}"

final_acc = rows[-1]["acc"]
first = next((r for r in rows if r["acc"] >= 0.94), None)

out = dict(accuracy=final_acc, wall_clock_sec=total)
with open("/workspace/agent_reported_metrics.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
print(f"checkpoint span            {span:.3f}s")
print(f"train loop only            {wall:.3f}s")
print(f"reported (>= span)         {total:.3f}s")
if first:
    N = first["epoch"]
    t_star = (first["mtime"] - rows[0]["mtime"]) * N / (N - 1)
    print(f"first >= 0.94              epoch {N}  acc {first['acc']:.4f}")
    print(f"t_star (verifier formula)  {t_star:.2f}s -> S_outcome "
          f"{max(0.0, min(1.0, (70.0 - t_star) / 42.0)):.4f}")
else:
    print("NO CROSSING", file=sys.stderr)
    sys.exit(1)
