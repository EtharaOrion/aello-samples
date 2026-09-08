# Provenance narrative for AELLO-C6-S1

Moved out of solution/provenance.yaml so the carrier holds only structured scalars,
enums, digests and identifiers (G-CON-RECORD, the closed schema of step 8i). The
reasoning is preserved here verbatim; the carrier references this file by digest as
narrative.sha256. This file is not the step-8i carrier and is not on the leak-gate
path: it is hash-bound as an ordinary solution-side note, never a private canary
carrier.

## anchors.note

The reference reaches r_det = 1.0 on the host grading path: all five gates clear with a positive margin and the time ramp saturates because the reference at ~221 s sits below the T_MIN 400 s knee. Controls fail correctly: a length-only judge scores 0 on the adversarial stratum floor, a constant judge scores 0 on the swap cap.

## corpus.citation

Wang et al. (2024). HelpSteer2. nvidia/HelpSteer2, CC-BY-4.0.

## corpus.licence_chain_note

Data and code licences recorded separately. The graded assembly, the private stratified holdout, and the private A/B side assignment are our own derivation and appear in no public corpus; the task-generating event is this derivation date, not HelpSteer2's release.

## identity.leak_scan_strength

leak_scan() scans the FROZEN token set derived from the canonical_content_hash this file binds, never re-derived from the tree under scan (F64 repair of 2026-08-18), and fails closed when no bound identity resolves. Containment additionally holds by Harbor assembly, since solution/ and tests/ are excluded from the assembled agent package.

## signature_note

The carrier names NO signer and NO detached-signature path, and no solution/provenance.sig exists in this bundle. An earlier revision of the carrier declared both; declaring a signature that cannot be honoured is worse than declaring none, so the fields were removed and this note is corrected to match the bytes it describes. Whether a signature exists is knowable only outside these bytes and is never asserted here. solution/provenance.sig is excluded from the canonical bundle hash by FORGE.md Phase 1 item 10i, so its absence never moves identity, and an absent detached signature caps the disposition at HOLD:PILOT_REQUIRED at Phase 2, where that fact is measured rather than self-declared. The signature itself is the pilot-blocked half; this correction needs no signer.
