"""Delivery-path baseline for C2-S2. Replace it entirely if you want.

It exists to prove the delivery path end to end before the corpus lands: it emits a well-formed
six-section artifact at the graded path -- a chain manifest over the eight pinned stage tools, the
twelve argument bindings, the pinned tool digests, the export record, the published composite
restated, and per-stage seconds. It composes the stages in one admissible linearization and binds
every argument correctly, because a baseline that emits a malformed manifest tests nothing.

It writes NOTHING into run_record.json. That record holds the quantities an orchestration must not
author -- the harness-measured wall clock, the private episode ledger, the pinned digests, the
twelve single-link ablations, the mount manifest -- and a starter that wrote them would be handing
the verifier the agent's own account of what the verifier is there to measure.

It is deliberately weak on the graded metric. Chain end-state accuracy over the 1200 private
episodes is measured verifier-side against a model this script never trains, so improving the
number means composing a real chain, not editing this file's literals.
"""
import hashlib
import json
import os

SUBMISSION = '/submission/c2s2.json'

STAGES = ("export", "merge", "render", "rollout", "score", "seed", "sft", "update")
BINDINGS = {
    "--mask-convention": "postshift",      # sft shifts labels internally
    "--generation-prompt": "on",           # one explicit value at render AND rollout
    "--pad-side": "left",                  # the side rollout emits on
    "--epoch-binding": "inherit",          # one seed epoch for the whole run
    "--score-input": "rollout-latest",     # never a previous iteration's file
    "--export-form": "merged",
    "--embed-resize": "tool-tokens",
    "--tokenizer-fingerprint": "propagate",
    "--advantage-baseline": "group",
    "--reference-dtype": "fp32",
    "--merge-before-rollout": "yes",
    "--decoy-downweight": "subset",
}
EPOCH = "epoch-starter-0"


STAGE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage")


def digest(tag):
    """A placeholder artifact digest. The real ones come from the stage tools themselves."""
    return hashlib.sha256(tag.encode()).hexdigest()


def stage_digest(stage):
    """The pinned per-file sha256 of one stage tool, read off the tool.

    Never authored: the manifest has to name the bytes that actually ran, and the grader compares
    it against its own pin. When environment/stage/ has not landed yet there is nothing to read,
    so this falls back to a placeholder and the digest check reports stage_tree_written -- a named
    absence rather than a silent pass.
    """
    path = os.path.join(STAGE_ROOT, stage)
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    return digest("stage/" + stage)


def artifact(name, tag):
    return {"artifact": name, "sha256": digest(tag)}


def invocation(stage, started, ended, consumed, produced):
    return {"stage": stage, "stage_digest": stage_digest(stage), "epoch_id": EPOCH,
            "started_s": started, "ended_s": ended, "bindings": dict(BINDINGS),
            "consumed": consumed, "produced": produced}


def chain():
    """One admissible linearization: seed first, render before sft, merge before rollout, one
    GRPO iteration, export last."""
    return [
        invocation("seed", 0.0, 30.0, [], [artifact("world_state", "world/0")]),
        invocation("render", 30.0, 150.0, [artifact("world_state", "world/0")],
                   [artifact("render_spec", "render/spec"),
                    artifact("rendered_episodes", "render/episodes")]),
        invocation("sft", 150.0, 500.0, [artifact("rendered_episodes", "render/episodes")],
                   [artifact("adapter", "sft/adapter")]),
        invocation("merge", 500.0, 540.0, [artifact("adapter", "sft/adapter")],
                   [artifact("merged_weights", "merge/0")]),
        invocation("rollout", 540.0, 660.0,
                   [artifact("merged_weights", "merge/0"), artifact("world_state", "world/0")],
                   [artifact("rollout_samples", "rollout/0")]),
        invocation("score", 660.0, 700.0, [artifact("rollout_samples", "rollout/0")],
                   [artifact("episode_rewards", "score/0")]),
        invocation("update", 700.0, 900.0,
                   [artifact("episode_rewards", "score/0"),
                    artifact("rollout_samples", "rollout/0")],
                   [artifact("merged_weights", "merge/1")]),
        invocation("export", 900.0, 960.0,
                   [artifact("merged_weights", "merge/1"), artifact("render_spec", "render/spec")],
                   [artifact("export_bundle", "export/final")]),
    ]


def main():
    os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
    baseline(SUBMISSION)
    return 0


def baseline(path):
    """Emit the six-section artifact instruction.md publishes."""
    records = chain()
    document = {
        "chain_manifest": records,
        "argument_bindings": dict(BINDINGS),
        "stage_digests": {stage: stage_digest(stage) for stage in STAGES},
        "export_record": {
            "export_form": "merged",
            "render_spec_digest": digest("render/spec"),
            "artifact_sha256": digest("export/final"),
            "validate_export_reason_codes": ["export_form_rejected", "export_unloadable",
                                             "manifest_absent", "manifest_malformed"],
        },
        "composite_declaration": {
            "quality_weight": 0.55, "speed_weight": 0.45,
            "quality_fraction": 0.0, "speed_fraction": 0.0, "composite": 0.0,
            "form": "quality_fraction * (0.55 + 0.45 * speed_fraction)",
        },
        "stage_seconds": {record["stage"]: round(record["ended_s"] - record["started_s"], 3)
                          for record in records},
    }
    with open(path, "w") as handle:
        json.dump(document, handle, indent=1, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
