# Provenance narrative for AELLO-C6-S3

Moved out of solution/provenance.yaml so the carrier holds only structured scalars,
enums, digests and identifiers (G-CON-RECORD, the closed schema of step 8i). The
reasoning is preserved here verbatim; the carrier references this file by digest as
narrative.sha256. This file is not the step-8i carrier and is not on the leak-gate
path: it is hash-bound as an ordinary solution-side note, never a private canary
carrier.

## anchors.measured_reference.margins.coverage_above_knee

0.0323 absolute above COV_TARGET 0.26, 11.1 percent of the reference coverage. Coverage is an integer answered count over an integer shard size taken from a bit-identical verdict artefact, so it carries EXACTLY zero run-to-run variance; this margin absorbs Phase-1 re-measurement drift on a rebuilt host image rather than run-to-run noise.

## anchors.measured_reference.note

Measured 2026-08-18 on the authoring host through the delivered tests/verifier.py over the two frozen private shards, 3000 items each. Both clamps saturate, the four compiled structural checks pass so GATE is 1, and r_det evaluates to binary64 1.0 exactly. The host carried other tenants throughout, so the 1214 s is an upper bound on an exclusive grading host.

## corpus.citation

Bai et al. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. Anthropic/hh-rlhf, MIT.

## corpus.exclusion_is_a_red_line

The harmless portion is excluded by construction from the shipped corpus and from both graded shards, so the graded quantity never becomes a safety-stance judgment.

## corpus.licence_chain_note

Data and code licences recorded separately. The two graded shards, their private mixture proportions, the private re-serialisation and the private per-shard side assignment are our own derivation and appear in no public corpus; the task-generating event is this derivation date, not hh-rlhf's release.

## screening.attestation_reason

The pod holds no signed platform attestation for a fork-ancestry snapshot or a task-generating-event timestamp. FORGE.md Phase 0 item 9a.1 caps at HOLD:PILOT_REQUIRED with this named reason rather than accepting the declaration on trust.

## screening.reason

hh-rlhf is a public MIT corpus very likely inside pretraining mixtures, so preference-label recall is a live exposure. It is bounded by the private two-shard mixture, the private re-serialisation, the coverage-gap check, and by the graded quantity being a calibrated selective decision rather than a per-item label, which recall alone does not supply. Whether hh-rlhf holds a freeze-table row is not exposed by the projection and must be resolved against the ENGRAM roots at Phase 2 item 7f. Until then this stands as a named screening-unverified exposure, never a clean result.

## shards.admission_rule

Only pairs whose chosen and rejected transcripts share every turn but the final assistant turn are admitted, so the decisive turn is well defined; duplicates are removed by content digest, and every private item is checked absent from the shipped split by exact context-plus-response match.

