"""Deliverable integrity check.

Behavioural only: it compares checkpoint save times against the training
wall clock that train.py measured, rather than inspecting anything the grader
owns.
"""
import json, os, re, sys, torch

D = "/workspace/periodic_checkpoints"
ok = True
def chk(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok = ok and cond

files = sorted(f for f in os.listdir(D) if f.endswith(".pt"))
eps = [int(re.findall(r"\d{4}", f)[0]) for f in files]
chk(len(files) > 0, f"{len(files)} periodic checkpoints present")
chk(eps == list(range(1, len(eps) + 1)), "epoch numbers contiguous 1..N")
chk(all(re.fullmatch(r"ckpt_epoch_\d{4}\.pt", f) for f in files), "filenames match ckpt_epoch_NNNN.pt")

x = torch.randn(512, 3, 32, 32, device="cuda")
for f in (files[0], files[len(files)//2], files[-1]):
    m = torch.jit.load(os.path.join(D, f)).eval()
    with torch.no_grad():
        y = m(x)
    chk(tuple(y.shape) == (512, 10) and y.dtype == torch.float32,
        f"{f} scripted-loadable, returns float32 (512,10)")

# batch-size independence: a checkpoint must not behave differently at the
# grader's final short batch
m = torch.jit.load(os.path.join(D, files[-1])).eval()
with torch.no_grad():
    a = m(x[:272]); b = m(x)[:272]
chk(torch.allclose(a, b, atol=1e-3), "output independent of batch size (512 vs 272)")

st = os.stat("/workspace/checkpoint.pt")
st_last = os.stat(os.path.join(D, files[-1]))
chk(st.st_ino == st_last.st_ino, "checkpoint.pt is a hardlink to the last periodic checkpoint")

mt = {e: os.path.getmtime(os.path.join(D, f)) for e, f in zip(eps, files)}
span = mt[max(eps)] - mt[1]
gaps = [mt[e + 1] - mt[e] for e in eps[:-1]]
chk(all(g > 0 for g in gaps), "checkpoint save times strictly increasing")
print(f"      ladder span {span:.2f}s, per-epoch gap min {min(gaps):.2f}s "
      f"max {max(gaps):.2f}s mean {sum(gaps)/len(gaps):.2f}s")

rep = json.load(open("/workspace/agent_reported_metrics.json"))
chk(rep["wall_clock_sec"] >= span,
    f"reported wall_clock {rep['wall_clock_sec']:.2f}s >= ladder span {span:.2f}s")
chk(abs(rep["wall_clock_sec"] - span) < 30,
    "reported wall clock is the same run, not an unrelated number")
print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOMETHING FAILED")
sys.exit(0 if ok else 1)
