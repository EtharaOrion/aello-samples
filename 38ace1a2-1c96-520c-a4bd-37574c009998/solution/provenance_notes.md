# Provenance narrative for AELLO-C8-S1

Moved out of solution/provenance.yaml so the carrier holds only structured scalars,
enums, digests and identifiers (G-CON-RECORD, the closed schema of step 8i). The
reasoning is preserved here verbatim; the carrier references this file by digest as
narrative.sha256. This file is not the step-8i carrier and is not on the leak-gate
path: it is hash-bound as an ordinary solution-side note, never a private canary
carrier.

## anchors.note

The harness imports the shipped code rather than reimplementing it, so a measurement cannot drift from what ships.

## corpus.citation

Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.

## corpus.licence_chain_note

Data licence and code licence are recorded separately and deliberately. Conflating them is how a bundle acquires an unenforceable claim.

## corpus.licence_data

Released for research use by the authors; no redistribution restriction asserted upstream.

## identity.leak_scan_strength

Recorded so the clean result is read for exactly what it is. The F64 defect in the shared instrument is repaired as of 2026-08-18: seed/identity.py leak_scan() now scans against the FROZEN token set, derived from the canonical_content_hash this file binds or from a caller-supplied frozen hash, never from the tree under scan, and it fails closed with a named finding when no bound identity resolves. Measured on this lineage's fixtures: a verbatim leak of solution/rubrics.json onto the agent-visible surface, which the pre-repair scan reported clean, now fires a canary match on all four token slots, a clean planted bundle stays silent, and the item 11a conformance probe identity.leak_gate proves both halves on every run. Containment additionally holds by Harbor assembly, since solution/ and tests/ are excluded from the assembled agent package; assembly is convenience and the repaired gate is the proof.

## identity.note

FORGE.md Phase 2 item 7 requires the recomputed hash to equal the hash this file binds and to derive the directory uuid. An earlier revision of this file bound no hash at all, so the obligation could not be discharged from bundle bytes; it is bound here. This file and solution/provenance.sig are excluded from the canonical hash by Phase 1 item 10i, so recording identity here does not move identity.

## signature_note

The carrier names NO signer and NO detached-signature path, and no solution/provenance.sig exists in this bundle. An earlier revision of the carrier declared both; declaring a signature that cannot be honoured is worse than declaring none, so the fields were removed and this note is corrected to match the bytes it describes. Whether a signature exists is knowable only outside these bytes and is never asserted here. solution/provenance.sig is excluded from the canonical bundle hash by FORGE.md Phase 1 item 10i, so its absence never moves identity, and an absent detached signature caps the disposition at HOLD:PILOT_REQUIRED at Phase 2, where that fact is measured rather than self-declared. The signature itself is the pilot-blocked half; this correction needs no signer.

## supersedes.chain[0].reason

An intermediate freeze of the remediated tree. Adversarial verification found that three files asserted the pristine-starter fallback in the step-budget check was fixed, and it was not: the collect hook copies the shipped starter into the scanned root where a sorted walk can let it shadow the agent's declaration. Measured both directions. The claims were corrected and the bundle re-frozen as 748ada0e; no grading behaviour changed, verified by the reward fixtures reproducing bit-identically across the freeze.

## supersedes.chain[1].reason

The originally delivered bundle, audited (c8s1-audit.md) NOT DELIVERABLE on four blockers and remediated (c8s1-adjudication.md); admitted as touchstone ts-02 (BLOCK) and withdrawn from dataset/ under G111 on 2026-08-18. Retained unmodified in touchstones/ as the negative half of the calibration pair.

## supersedes.reason

The delivered remediated bundle, admitted as touchstone ts-03 (HOLD). This freeze supersedes it under the 2026-08-18 re-scope reconcile: the F65-defective step_budget_declared body is replaced by the digest-exclusion form that grades only agent-authored declarations, the budget subsystem bindings (budget_hours, budget_envelope, budget_margin) land in task.toml with max_timeout de-conflated from budget_hours, instruction.md makes the budget declaration mandatory, the ciFAIR-100 duplicate share is measured and carried in tests/heldout/known_overlap.json, and the leak-scan honesty text reflects the repaired frozen-token instrument (F64). The reward map constants are unchanged; ts-03 remains the calibration touchstone and is not re-authored.

## supersedes.relationship

Same task design, same graded axis, same anchors across the whole chain; wiring and contract conformance repairs only. The 2026-08-18 freeze additionally carries the F64 and F65 instrument repairs, the budget-subsystem bindings, and the measured ciFAIR duplicate share.

