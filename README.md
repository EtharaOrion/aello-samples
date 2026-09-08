# Promoted sample index

GENERATED SECTION. DO NOT HAND-EDIT.

Regenerated every run by `seed/forge.py` from the promoted bundles in this directory. Drift between this file and those bundles fails closed. It publishes an absolute expiry rather than a current-overdue flag, because a flag goes stale where an absolute instant does not: readers compare the instant themselves.

No bundle is promoted. Promotion requires five conditions at transition and this project meets none of them: a SHIP disposition, a current signed pilot under the projected freshness horizon, freshly re-proven frozen-byte delivery conformance and ground truth, zero open gaps, and current contamination screening whose release root digests remain effective. No external signer exists and the screening roots are unresolved, so no delivered bundle can meet the promotion conditions and the honest state of this directory is empty. The delivered bundles live under `dataset/`; this index covers promotion only.

## Removal lag

Only a FORGE invocation removes bytes from this directory. A prompt-only contract cannot schedule work, cannot mutate bytes between invocations, and cannot remove a promoted unit without being invoked, so a unit that becomes overdue between invocations leaves on the next run. That bounded lag is named here and is never treated as a silent pass.
