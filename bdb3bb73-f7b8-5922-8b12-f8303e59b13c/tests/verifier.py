"""Verifier for cifar10-94. Wires the bundle to the aello_eval engine.

Flow:
  1. S_outcome = aello_eval.normalized_time_to_target from LOADABLE periodic
     checkpoints, each re-evaluated INDEPENDENTLY (batched, on GPU) on the real
     test split. Timing is verifier-owned: read from the bind-mounted /logs
     (P1.3), NOT from mtimes (which artifact-copy resets in separate mode).
  2. pytest deterministic layer (aello_eval @check plugin) -> check_results.json.
  3. 3-judge LLM council grades the behavioural rubrics on a SEALED trajectory
     excerpt (P1.6), self-excluding the agent's own model.
  4. aggregate() composes score = GATE * S_outcome * RubricFactor. Fail-closed.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from aello_eval import (aggregate, assemble, load_check_results,
                        normalized_time_to_target, write_reward)
from aello_eval.schema import RubricSet

LOG_DIR = Path("/logs/verifier")
LOG_DIR.mkdir(parents=True, exist_ok=True)
# VERIFIER-OWNED evidence, delivered as artifacts (the separate verifier only mounts
# /logs/verifier, so agent-side /logs/* is not visible). The [[verifier.collect]] hook
# (verifier config, not agent code) copies the ENTRYPOINT-stamped launch time and
# stamps real checkpoint mtimes -- neither is agent-writable, closing fabricated t*.
CONTAINER_LAUNCH_TS = Path("/workspace/verifier_container_launch.ts")
VERIFIER_CKPT_TIMES = Path("/workspace/verifier_checkpoint_mtimes.jsonl")
PERIODIC_GLOB = "/workspace/periodic_checkpoints/ckpt_epoch_*.pt"
ACC_TARGET = 0.94
BUDGET_SEC = 1800
FLOOR = 0.5
# A model trained FROM SCRATCH on CIFAR-10 cannot have its EARLIEST periodic
# checkpoint already near target; an early checkpoint at/above this accuracy is
# proof of pretrained init (no_pretrained_weights red-line).
PRETRAINED_CEILING = 0.85
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

T_BASELINE_SEC: float | None = None
T_REFERENCE_SEC: float | None = None
EVAL_HOLDOUT_FETCH_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
EVAL_HOLDOUT_EXTRACT_SUBDIR = "cifar10/test"
EVAL_HOLDOUT_LOCAL_ROOT = Path("/workspace/verifier_holdout")
EVAL_HOLDOUT_EXPECTED_COUNT = 10000

def _read_launch_ts() -> float:
    if not CONTAINER_LAUNCH_TS.is_file():
        raise RuntimeError("container_launch.ts missing; ENTRYPOINT defect")
    return float(CONTAINER_LAUNCH_TS.read_text().strip())


def _checkpoint_times() -> dict[str, float]:
    """P1.3: ckpt filename -> save timestamp from the bind-mounted /logs jsonl
    (survives the agent->verifier container hop; mtimes do not)."""
    times: dict[str, float] = {}
    if VERIFIER_CKPT_TIMES.is_file():
        for line in VERIFIER_CKPT_TIMES.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ts = d.get("mtime", d.get("ts"))
                if "ckpt" in d and ts is not None:
                    times[os.path.basename(d["ckpt"])] = float(ts)
            except Exception:
                continue
    return times


def _eval_root() -> Path | None:
    """The VERIFIER-ONLY held-out eval set. Returns the eval dir, or None on infra failure."""
    import shlex
    local_root = EVAL_HOLDOUT_LOCAL_ROOT
    subdir = EVAL_HOLDOUT_EXTRACT_SUBDIR
    eval_dir = (local_root / subdir) if subdir else local_root
    if not eval_dir.is_dir():
        url = EVAL_HOLDOUT_FETCH_URL
        if not url:
            return None
        try:
            local_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["bash", "-c",
                 f"curl -fsSL --retry 5 --connect-timeout 20 {shlex.quote(url)} "
                 f"| tar -xz -C {shlex.quote(str(local_root))}"],
                check=True, timeout=600, capture_output=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"holdout eval fetch failed: {exc!r}", file=sys.stderr)
            return None
    if not eval_dir.is_dir():
        return None
    exp = EVAL_HOLDOUT_EXPECTED_COUNT
    if isinstance(exp, int):
        n = sum(1 for _ in eval_dir.rglob("*.png"))
        if n != exp:
            print(f"holdout eval count {n} != expected {exp}", file=sys.stderr)
            return None
    return eval_dir


def _reeval_accuracy(ckpt_path: Path, eval_root: Path) -> float:
    """Independently re-evaluate a scripted checkpoint on the VERIFIER-ONLY held-out
    eval set. BATCHED on the GPU; num_workers=0 (small /dev/shm in the verifier)."""
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = torchvision.datasets.ImageFolder(str(eval_root), transform=tf)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    try:
        model = torch.jit.load(str(ckpt_path), map_location=device).eval()
    except Exception as exc:
        print(f"  reeval load failed for {ckpt_path.name}: {exc}", file=sys.stderr)
        return 0.0
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def _compute_s_outcome() -> tuple[float, dict]:

    # P1-E: fail-closed if the verifier ENV is inadequate. A CPU fallback or missing
    # data would emit a WRONG number; instead return an infra reason the host-side
    # gate withholds (INVALID), never a plausible score.
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return 0.0, {"reason": "verifier_env_missing_deps", "detail": f"torch import failed: {exc!r}"}
    if not torch.cuda.is_available():
        return 0.0, {"reason": "no_gpu"}
    eval_root = _eval_root()
    if eval_root is None:
        return 0.0, {"reason": "eval_holdout_unavailable"}
    try:
        launch = _read_launch_ts()
    except Exception as exc:  # noqa: BLE001
        return 0.0, {"reason": "no_container_launch_ts", "detail": repr(exc)}

    ckpts = [Path(p) for p in glob.glob(PERIODIC_GLOB)]
    if not ckpts:
        return 0.0, {"reason": "no_periodic_checkpoints"}

    # P1-B: t* comes ONLY from verifier-collected mtimes. If they are absent, the
    # only remaining source is agent-influenced -> withhold (fabricated timing).
    times = _checkpoint_times()
    if not times:
        return 0.0, {"reason": "no_verifier_timing"}
    ckpts = [p for p in ckpts if p.name in times]
    if not ckpts:
        return 0.0, {"reason": "no_verifier_timing"}
    ckpts.sort(key=lambda p: times[p.name])

    # t* (time-to-target) is the raw measurement of the first crossing checkpoint.
    walked = []
    crossing = None  # (t_star, acc, ckpt_name)
    for p in ckpts:
        t_star = times[p.name] - launch
        if t_star > BUDGET_SEC:
            walked.append({"ckpt": p.name, "t_star_sec": t_star, "reason": "past_budget"})
            continue
        acc = _reeval_accuracy(p, eval_root)
        walked.append({"ckpt": p.name, "t_star_sec": t_star, "reeval_accuracy": acc})
        if acc >= ACC_TARGET:
            crossing = (t_star, acc, p.name)
            break

    if crossing is None:
        return 0.0, {"reason": "no_checkpoint_reached_target", "walked": walked}

    t_star, acc, ckpt_name = crossing
    measured = {"t_star_sec": t_star, "reeval_accuracy": acc, "first_crossing_checkpoint": ckpt_name,
                "time_source": "verifier_collected_mtime", "walked": walked}

    tb, tr = T_BASELINE_SEC, T_REFERENCE_SEC
    if not isinstance(tb, (int, float)) or not isinstance(tr, (int, float)):
        return 0.0, {"reason": "anchors_not_configured", "t_baseline_sec": tb,
                     "t_reference_sec": tr, **measured}

    s = normalized_time_to_target(t_star_sec=t_star, t_baseline_sec=float(tb), t_reference_sec=float(tr))
    return s, measured


def _pretrained_ceiling() -> float:
    """Per-task plausibility ceiling for the no_pretrained_weights check."""
    return PRETRAINED_CEILING


def _override_check(results: list, check_id: str, *, passed: bool, evidence: dict) -> None:
    """Replace/insert a deterministic knockout result with a verifier-enforced verdict."""
    entry = {"id": check_id, "passed": bool(passed), "score": 1.0 if passed else 0.0,
             "weight": 0, "knockout": True, "kind": "process", "grader": "pytest",
             "evidence": evidence}
    for i, c in enumerate(results):
        if c.get("id") == check_id:
            results[i] = entry
            return
    results.append(entry)


def main() -> int:
    # Deterministic layer: pytest @check plugin flushes check_results.json.
    r = subprocess.run(["pytest", "-q", "--no-header", "-p", "aello_eval.checks",
                        str(Path(__file__).parent / "test_output.py")],
                       capture_output=True, text=True)
    print("=== pytest ===")
    print(r.stdout[-2000:])

    check_results = load_check_results()
    outcome_score, outcome_detail = _compute_s_outcome()

    # REAL no_pretrained_weights enforcement (replaces the pytest placeholder): the
    # verifier re-eval gives per-checkpoint accuracy; a from-scratch model's EARLIEST
    # checkpoint cannot already be near-target. High early accuracy -> pretrained
    # init (or memorized data) -> red-line fail. Override + persist the verdict so
    # the host-side gate sees the ENFORCED result.
    earliest_acc = next(
        (w.get("reeval_accuracy") for w in outcome_detail.get("walked", [])
         if "reeval_accuracy" in w),
        None,
    )
    if earliest_acc is not None:
        ceiling = _pretrained_ceiling()
        passed = earliest_acc < ceiling
        _override_check(check_results, "no_pretrained_weights", passed=passed,
                        evidence={"earliest_checkpoint_accuracy": earliest_acc,
                                  "ceiling": ceiling, "enforced": True})
        (LOG_DIR / "check_results.json").write_text(json.dumps(check_results, indent=2))

    # This SEPARATE verifier cannot see the agent's /workspace code or its reasoning
    # trajectory (Harbor writes the trajectory host-side). So it does NOT run the
    # council -- it emits the MEASURED base (S_outcome) + deterministic results, and
    # the scaffold runs the council host-side (scaffold/evaluate.py) over the real
    # agent code + trajectory, writing the FINAL score there.
    # Generic contract read by the host-side delivery gate: declare whether the
    # outcome is a real measurement ('ok'), a genuine agent miss ('agent_miss',
    # delivered as a valid 0), or an infra fault ('infra_failure', withheld).
    reason = outcome_detail.get("reason")
    outcome_detail["outcome_status"] = (
        "ok" if reason is None
        else "agent_miss" if reason == "no_checkpoint_reached_target"
        else "infra_failure"
    )
    (LOG_DIR / "outcome.json").write_text(
        json.dumps({"s_outcome": outcome_score, **outcome_detail}, indent=2)
    )

    # Provisional score = GATE * S_outcome * (pytest-only factor); an empty rubric
    # set means assemble() takes only the deterministic results from check_results.
    # Harbor needs a numeric reward.json; the host-side grade overrides it.
    provisional = assemble(RubricSet(rubrics=[]), check_results, {})
    score, detail = aggregate(outcome_score=outcome_score, rubrics=provisional, floor=FLOOR)
    detail["council_status"] = "pending_host_side"
    write_reward(score, detail, log_dir=str(LOG_DIR))

    out = Path("/workspace/verifier_output")
    out.mkdir(parents=True, exist_ok=True)
    (out / "outcome_detail.json").write_text(json.dumps(outcome_detail, indent=2))
    print(f"[provisional] score={score:.4f} s_outcome={outcome_score:.4f} "
          f"rubric_factor={detail.get('rubric_factor')} (council runs host-side)")
    return 0


if __name__ == "__main__":
    # P0.3 fail-closed: any exception -> numeric reward 0.0 + reason.
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as _exc:
        import json as _json
        import traceback as _tb
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "reward.json").write_text(_json.dumps({"reward": 0.0}))
        (LOG_DIR / "reward_detail.json").write_text(_json.dumps({
            "reward": 0.0,
            "zero_score_reason": f"verifier_crash: {type(_exc).__name__}: {_exc!r}",
            "traceback": _tb.format_exc(),
        }))
        raise SystemExit(1)
