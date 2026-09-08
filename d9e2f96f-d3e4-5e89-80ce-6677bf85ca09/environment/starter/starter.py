"""Runnable baseline for AELLO-C8-S11. Correct, honestly measured -- and deliberately slow.

WHAT IT DOES. For each of the three pinned block shapes it runs an unfused depthwise-separable
block on that shape's vendored activation trace: a 3x3 stride-1 depthwise convolution, then a 1x1
pointwise projection, both in fp32. It times the two stages separately with CUDA events, discards
a warmup, takes a median over the repetition floor, and writes the measurement record and the run
record that instruction.md publishes. Replace all of it.

WHAT IT IS NOT. It is not a fast kernel and it is not trying to be. It is the REFERENCE
implementation -- the thing your fidelity is measured against -- so its own drift from the vendored
fp32 reference is zero by construction, and yours will not be. When you replace it, compute
max_abs_error and max_rel_error by running this baseline on the same trace and differencing.

IT FABRICATES NOTHING. If the trace corpus has not landed in your tree, or if no CUDA device is
present, it says so in the report and in the run record and leaves the latency fields null. A
record with a null latency fails the compiled checks, which is the correct outcome: no measurement
was taken. It never writes a number it did not measure -- that is red line 2, and the starter is
not going to demonstrate it for you.

THREE THINGS WORTH KNOWING BEFORE YOU BUILD ON IT.

  DETERMINISM IS SWITCHED ON. `replay` must reproduce every shape's output digest bit-for-bit
  while its milliseconds must NOT all reproduce exactly. Non-deterministic algorithm selection
  breaks the first half; a copied constant breaks the second. cuDNN benchmarking is disabled here
  for exactly that reason, and turning it back on is a legitimate lever -- but then the autotune
  cost belongs in your declared budget.

  block_ms IS THE MEDIAN OF THE PER-ITERATION SUM, not the sum of the two stage medians. The two
  differ by a fraction of a percent on stable timings and the checker allows five, but only one of
  them is the latency of the block.

  THE STAGES DO NOT COST ALIKE. Print the two stage times before you optimise anything. At
  32x32x128 the depthwise stage dominates and it is memory-bound; at 8x8x512 the pointwise stage
  dominates and it is compute-bound. A single blended millisecond will not tell you which lever to
  pull.
"""
import hashlib
import json
import os
import statistics
import time

SUBMISSION = os.environ.get("AELLO_SUBMISSION", "/submission/kernel_report.json")
RECORD = os.environ.get("AELLO_RUN_RECORD", "/workspace/run_record.json")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRACES = os.path.join(DATA, "c8s1_activation_traces.npz")
SHAPE_TABLE = os.path.join(DATA, "graded_block_shapes.json")

WARMUP = 30            # the published floor is 25
TIMED = 120            # the published floor is 100

# The pinned graded set, mirroring environment/data/graded_block_shapes.json. It is embedded so
# the starter runs before the corpus lands; the shipped table is authoritative when it is there.
PINNED = [
    {"id": "dws32_c128", "batch": 256, "channels": 128, "height": 32, "width": 32,
     "depthwise_kernel": 3, "stride": 1, "pointwise_out": 128, "trace_seed": 20260814},
    {"id": "dws16_c256", "batch": 256, "channels": 256, "height": 16, "width": 16,
     "depthwise_kernel": 3, "stride": 1, "pointwise_out": 256, "trace_seed": 20260815},
    {"id": "dws8_c512", "batch": 256, "channels": 512, "height": 8, "width": 8,
     "depthwise_kernel": 3, "stride": 1, "pointwise_out": 512, "trace_seed": 20260816},
]


def graded_shapes():
    """The shipped shape table when it has landed, the embedded pin otherwise."""
    if not os.path.exists(SHAPE_TABLE):
        return PINNED
    try:
        with open(SHAPE_TABLE) as handle:
            doc = json.load(handle)
    except Exception:
        return PINNED
    shapes = doc.get("shapes", doc) if isinstance(doc, dict) else doc
    return shapes if isinstance(shapes, list) and shapes else PINNED


def trace_digest():
    """sha256 of the vendored trace archive, or None when it has not landed.

    Recorded before the first measurement and after the last. Red line 1 is writing into the
    traces, and two equal digests either side of the run is what says it did not happen.
    """
    if not os.path.exists(TRACES):
        return None
    digest = hashlib.sha256()
    with open(TRACES, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_device():
    """(torch, device) when a CUDA device is present, (None, None) otherwise."""
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    torch.backends.cudnn.benchmark = False       # see DETERMINISM, above
    torch.backends.cudnn.deterministic = True
    return torch, torch.device("cuda")


def load_trace(shape):
    """The vendored activation trace for one shape, or None when the corpus has not landed."""
    if not os.path.exists(TRACES):
        return None
    import numpy as np
    with np.load(TRACES, allow_pickle=False) as archive:
        if shape["id"] not in archive.files:
            return None
        return archive[shape["id"]].astype("float32")


def time_block(torch, device, shape, activations):
    """Median depthwise, pointwise and block milliseconds over TIMED iterations after WARMUP.

    Both stages are timed with CUDA events inside the same iteration. A host timer here would
    measure the launch queue: conv2d returns to Python before the kernel has run.
    """
    x = torch.from_numpy(activations).to(device)
    channels, out_channels = int(shape["channels"]), int(shape["pointwise_out"])
    k, stride = int(shape["depthwise_kernel"]), int(shape["stride"])
    generator = torch.Generator(device="cpu").manual_seed(int(shape["trace_seed"]))
    dw_weight = torch.randn(channels, 1, k, k, generator=generator).to(device)
    pw_weight = torch.randn(out_channels, channels, 1, 1, generator=generator).to(device)
    pad = k // 2

    def block(inp):
        stage = torch.nn.functional.conv2d(inp, dw_weight, stride=stride, padding=pad,
                                           groups=channels)
        return stage, torch.nn.functional.conv2d(stage, pw_weight)

    for _ in range(WARMUP):
        block(x)
    torch.cuda.synchronize()

    start, mid, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    depthwise, pointwise, blocks = [], [], []
    out = None
    for _ in range(TIMED):
        start.record()
        stage = torch.nn.functional.conv2d(x, dw_weight, stride=stride, padding=pad,
                                           groups=channels)
        mid.record()
        out = torch.nn.functional.conv2d(stage, pw_weight)
        end.record()
        end.synchronize()
        one, two = start.elapsed_time(mid), mid.elapsed_time(end)
        depthwise.append(one)
        pointwise.append(two)
        blocks.append(one + two)
    digest = hashlib.sha256(out.detach().cpu().numpy().tobytes()).hexdigest()
    return {"depthwise_ms": statistics.median(depthwise),
            "pointwise_ms": statistics.median(pointwise),
            "block_ms": statistics.median(blocks), "output_sha256": digest}


def unmeasured_record(shape, why):
    """A record that says no measurement was taken, rather than one that invents a number."""
    return {"block_ms": None, "depthwise_ms": None, "pointwise_ms": None,
            "warmup_iterations": WARMUP, "warmup_excluded": True, "timed_iterations": TIMED,
            "statistic": "median", "clock_source": "cuda_event_elapsed_time",
            "fusion_boundary": "unfused", "max_abs_error": None, "max_rel_error": None,
            "within_envelope": None, "trace_seed": shape["trace_seed"], "output_sha256": None,
            "unmeasured_reason": why}


def measured_record(shape, timing):
    """The baseline IS the fp32 reference, so its drift from the reference is zero, not assumed."""
    return {"block_ms": timing["block_ms"], "depthwise_ms": timing["depthwise_ms"],
            "pointwise_ms": timing["pointwise_ms"], "warmup_iterations": WARMUP,
            "warmup_excluded": True, "timed_iterations": TIMED, "statistic": "median",
            "clock_source": "cuda_event_elapsed_time", "fusion_boundary": "unfused",
            "max_abs_error": 0.0, "max_rel_error": 0.0, "within_envelope": True,
            "trace_seed": shape["trace_seed"], "output_sha256": timing["output_sha256"]}


def main():
    started = time.time()
    shapes = graded_shapes()
    torch, device = torch_device()
    before = trace_digest()

    records, replay, harness_view = {}, {}, {}
    for shape in shapes:
        activations = load_trace(shape)
        if torch is None:
            records[shape["id"]] = unmeasured_record(shape, "no_cuda_device")
            continue
        if activations is None:
            records[shape["id"]] = unmeasured_record(shape, "trace_corpus_not_landed")
            continue
        first = time_block(torch, device, shape, activations)
        second = time_block(torch, device, shape, activations)
        records[shape["id"]] = measured_record(shape, first)
        # The digests must agree and the milliseconds must not all agree. Both halves are
        # measured here rather than asserted: the second pass is a real second measurement.
        replay[shape["id"]] = {"output_sha256": second["output_sha256"],
                               "block_ms": second["block_ms"]}
        harness_view[shape["id"]] = first["block_ms"]

    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    with open(SUBMISSION, "w") as handle:
        json.dump({"shapes": records,
                   "corpus_status": "landed" if before else "not-landed",
                   "device": "cuda" if torch is not None else "none"}, handle, indent=1)

    with open(RECORD, "w") as handle:
        json.dump({
            "trace_seeds": [s["trace_seed"] for s in shapes],
            "paths_written": [SUBMISSION, RECORD],
            "trace_digest_before": before,
            "trace_digest_after": trace_digest(),
            "replay": replay,
            # This baseline autotunes nothing: cuDNN benchmarking is off and no tiling is swept.
            # Zero is a declaration, not an omission -- and your solve's will not be zero.
            "agent_budget_declaration": {"warmup_iterations": WARMUP, "timed_iterations": TIMED,
                                         "autotune_trials": 0},
            "starter_elapsed_s": round(time.time() - started, 3),
        }, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
