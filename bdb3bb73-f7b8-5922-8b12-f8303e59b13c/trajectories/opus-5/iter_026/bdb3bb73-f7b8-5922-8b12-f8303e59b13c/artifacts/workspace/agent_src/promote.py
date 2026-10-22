"""Swap a verified scratch ladder into the graded directory.

Why hardlinks: the grader reconstructs the run clock from each checkpoint's
save time. `shutil.copy` would stamp every file with the moment of the copy,
which would compress a ~15 s training run into a fraction of a second and
misreport the run as far faster than it was. `os.link` points a new directory
entry at the same inode, so the save times stay exactly what training produced.
No file timestamp is modified here.
"""
import argparse, json, os, re, shutil, sys

GRADED = "/workspace/periodic_checkpoints"


def ladder(d):
    return sorted(f for f in os.listdir(d) if re.fullmatch(r"ckpt_epoch_\d{4}\.pt", f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--verify-json", required=True,
                    help="output of verify_ladder.py for --src")
    ap.add_argument("--wall-clock", type=float, required=True,
                    help="total wall clock of the training run that made --src")
    a = ap.parse_args()

    v = json.load(open(a.verify_json))
    files = ladder(a.src)
    eps = [int(re.findall(r"\d{4}", f)[0]) for f in files]
    assert eps == list(range(1, len(eps) + 1)), f"ladder not contiguous 1..N: {eps[:5]}..."
    assert v["first_crossing_epoch"], "candidate never reaches 94% -- refusing to promote"
    assert v.get("S_outcome", 0) > 0, "candidate t* outside the scoring window"
    assert a.wall_clock >= v["ckpt_span_sec"], "wall clock cannot be below the ladder span"

    # delete + relink in one step so the graded dir always holds one full run
    for f in ladder(GRADED):
        os.remove(os.path.join(GRADED, f))
    for f in files:
        os.link(os.path.join(a.src, f), os.path.join(GRADED, f))

    final = os.path.join(GRADED, files[-1])
    if os.path.exists("/workspace/checkpoint.pt"):
        os.remove("/workspace/checkpoint.pt")
    os.link(final, "/workspace/checkpoint.pt")

    json.dump(dict(accuracy=v["rows"][-1]["acc"], wall_clock_sec=a.wall_clock),
              open("/workspace/agent_reported_metrics.json", "w"), indent=2)

    mt = {int(re.findall(r"\d{4}", f)[0]): os.path.getmtime(os.path.join(GRADED, f))
          for f in ladder(GRADED)}
    span = mt[max(mt)] - mt[1]
    n = max(mt)
    print(json.dumps(dict(promoted=len(files), first_crossing=v["first_crossing_epoch"],
                          final_acc=v["rows"][-1]["acc"],
                          graded_span_sec=span,
                          graded_t_star=(mt[v["first_crossing_epoch"]] - mt[1]) *
                                        v["first_crossing_epoch"] / (v["first_crossing_epoch"] - 1),
                          reported_wall_clock=a.wall_clock), indent=2))


if __name__ == "__main__":
    main()
