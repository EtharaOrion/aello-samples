"""Repeatability of the exact-pipeline evaluation across fresh processes.

The score depends on whether the first crossing checkpoint re-evaluates >= 94% in the
verifier's process, so the run-to-run spread of that number is worth knowing.  Fresh
processes are used because cudnn algorithm autotuning is re-done per process.
"""
import sys
import verify_ckpt as V

x, y = V.load_split("test")
for ep in (19, 20):
    acc = V.evaluate(f"/workspace/periodic_checkpoints/ckpt_epoch_{ep:04d}.pt", x, y)
    print(f"rep={sys.argv[1]} epoch={ep} acc={acc:.4f}", flush=True)
