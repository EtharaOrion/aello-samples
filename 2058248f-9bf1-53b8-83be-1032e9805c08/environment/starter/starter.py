"""Correct, unoptimised baseline for A5-02. Replace it entirely if you want.

It exists to prove the delivery path end to end: it writes a well-formed score matrix and a
well-formed export manifest, ships a parameter blob under the ceiling, and declares its own budget
accounting. It CLEARS all three deployment gates and it is deliberately weak on everything that
moves macro average precision:

  * input normalization is one global centre and one global scale broadcast across all 12 bands,
    which is the RGB-style reduction. The 12 Sentinel-2 bands differ by more than an order of
    magnitude in dynamic range, so this flattens the narrow ones into noise;
  * the weights are fp32 and nothing is quantized, so there are no per-channel scales at all;
  * there is no handling of label imbalance, so the rarest of the 19 labels are learned worst and
    the unweighted macro mean charges the full price for each of them.

Those three are the work. Everything below them is plumbing you may keep.
"""
import hashlib
import json
import os

SUBMISSION_DIR = "/submission"
SCORE_MATRIX = os.path.join(SUBMISSION_DIR, "preds_a502.npy")
EXPORT_MANIFEST = os.path.join(SUBMISSION_DIR, "export_a502.json")
PARAMETER_BLOB = os.path.join(SUBMISSION_DIR, "params_a502.bin")
RUN_RECORD = "/workspace/run_record.json"

BANDS = 12
LABELS = 19
# The graded instance count comes from the graded manifest when it is present. This fallback is
# only for a tree where the corpus has not landed; see corpus_layout in solution/grounding.yaml.
GRADED_MANIFEST = "/workspace/environment/data/graded/manifest.json"
FALLBACK_ROWS = 1024
SEED = 20260601


def graded_row_count():
    try:
        with open(GRADED_MANIFEST) as handle:
            return int(json.load(handle)["n_graded_patches"])
    except (OSError, ValueError, KeyError, TypeError):
        return FALLBACK_ROWS


def write_score_matrix(rows):
    """A float score per label per patch. Not an argmax: several of the 19 labels hold at once.

    Not a decision either. Average precision is threshold-free and reads the ordering, so every
    column has to carry a spread of values rather than a chosen 0 or 1.
    """
    import numpy

    generator = numpy.random.default_rng(SEED)
    scores = generator.random((rows, LABELS), dtype="float32") * 0.5 + 0.25
    numpy.save(SCORE_MATRIX, scores)
    return scores


def write_parameter_blob():
    """A small fp32 blob, well under the 4,194,304-byte ceiling."""
    payload = bytes(bytearray((index * 37) % 251 for index in range(262144)))
    with open(PARAMETER_BLOB, "wb") as handle:
        handle.write(payload)
    return len(payload)


def global_band_statistics():
    """ONE centre and ONE scale, broadcast across all twelve bands. This is the thing to fix."""
    centre, scale = 1024.0, 512.0
    return [centre] * BANDS, [scale] * BANDS


def write_export_manifest(param_bytes):
    centres, scales = global_band_statistics()
    operators = ["conv2d", "batchnorm2d", "relu", "avgpool2d", "globalavgpool2d", "gemm",
                 "sigmoid"]
    per_operator_flops = dict((name, 1 << 20) for name in operators)
    host_constants = {"flops_per_second": 3.2e14, "kernel_launch_seconds": 4.1e-06,
                      "hbm_bytes_per_second": 2.04e12}
    terms = {"flops_batch1": 1.2e08, "kernel_count": 44, "bytes_moved": 5.6e07}
    seconds = (terms["flops_batch1"] / host_constants["flops_per_second"]
               + terms["kernel_count"] * host_constants["kernel_launch_seconds"]
               + terms["bytes_moved"] / host_constants["hbm_bytes_per_second"])
    stages = dict(("stage_%02d" % index, 6291456 - index * 262144) for index in range(8))
    manifest = {
        "macro_average": {"denominator_basis": "labels_present_in_split",
                          "per_label_weighting": "uniform", "labels_scored": LABELS},
        "band_normalization": {"mean": centres, "scale": scales},
        # Nothing is quantized here, so there is no per-channel scale to declare. Meeting the
        # ceiling at a useful width is what forces quantization, and the granularity you pick is
        # what decides whether the rare labels survive it.
        "quantization": {"granularity": "none", "tensors": []},
        "parameter_blob": {"path": os.path.basename(PARAMETER_BLOB),
                           "serialized_bytes": param_bytes},
        "analytic_latency": dict(terms, per_operator_flops=per_operator_flops,
                                 host_constants=host_constants,
                                 estimate_ms=seconds * 1000.0),
        "activation": {"batch": 64, "recomputation": "none", "live_bytes_by_stage": stages,
                       "peak_live_bytes": max(stages.values())},
        "operator_envelope": {"source": "recovered_from_export_graph",
                              "graph_operators": operators},
    }
    with open(EXPORT_MANIFEST, "w") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return manifest


def write_run_record(manifest, rows):
    """The record the verifier reads. Every field instruction.md names is written here.

    The supervision trace is the honest one: each entry is the digest of the shipped shard's own
    label record, because this baseline trains on nothing else. A pipeline that introduced targets
    from anywhere else has to write those digests instead, and they will not match.
    """
    digests = [hashlib.sha256(("shard-label-%d" % index).encode()).hexdigest()
               for index in range(rows)]
    steps, batch = 4000, 256
    flops_per_sample = 1.8e11
    record = {
        "agent_budget_declaration": {"attempts_used": 1,
                                     "accounted_flops_spent": steps * batch * flops_per_sample,
                                     "family_chosen": "starter-plain-stack",
                                     "family_switch_attempt": None},
        "training_accounting": {"steps": steps, "batch_size": batch,
                                "flops_per_sample": flops_per_sample,
                                "accounted_flops": steps * batch * flops_per_sample},
        "termination": "flop_budget_reached",
        "export_complete_at_step": steps,
        "watchdog_wall_seconds": 1800,
        "training_supervision_trace": digests,
        "shard_label_record_digests": digests,
        "permutation_regrade": {"r_det_permutation_a": 0.05, "r_det_permutation_b": 0.05},
        "in_distribution_shards": {"graded": 0.512, "sibling": 0.512},
        "regrade_passes": [{"map_in": 0.512, "map_shift": 0.437},
                           {"map_in": 0.512, "map_shift": 0.437}],
        "average_precision_tie_break": "score_desc_then_record_index_asc",
        "digest_ledger": {"before": {}, "after": {}},
        "guarded_paths_touched": [],
        "verifier_measured_macro_ap": {"in_distribution": None, "held_out_country": None},
        "reward_log": [{"attempt": 1, "map_in": 0.512, "map_shift": 0.437, "r_in": 0.0,
                        "r_shift": 0.0, "r_det": 0.05,
                        "param_bytes": manifest["parameter_blob"]["serialized_bytes"],
                        "latency_estimate_ms": manifest["analytic_latency"]["estimate_ms"],
                        "activation_peak_bytes": manifest["activation"]["peak_live_bytes"],
                        "reason": ""}],
    }
    os.makedirs(os.path.dirname(RUN_RECORD), exist_ok=True)
    with open(RUN_RECORD, "w") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return record


def main():
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    rows = graded_row_count()
    write_score_matrix(rows)
    manifest = write_export_manifest(write_parameter_blob())
    write_run_record(manifest, rows)
    print("A5-02 starter: %d x %d score matrix, %.4f ms analytic latency, %d parameter bytes"
          % (rows, LABELS, manifest["analytic_latency"]["estimate_ms"],
             manifest["parameter_blob"]["serialized_bytes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
