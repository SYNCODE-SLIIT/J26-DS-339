# Reference-conditioned review flag: design (not implemented, not qualified)

4 November 2026. Design record only. No model, default, demo or threshold changed. Follows V70-V73.

## Why the repair target changes
V70-V73 (no-fit studies on existing predictions/features) show that the pitch trace cannot separate an injected
short detune from natural short-note movement (AUROC .62-.64 vs a declared .85 bar; abstain-by-default gate protects
originals but catches <1% of injections; oracle-span level estimate 18.97c with 54% natural false alarms).
A note off by ~25c is the same signal whether intended or not, so "intended pitch" is not recoverable from audio alone.
(Correction of an earlier suggestion: harmonic-ratio coherence would not help; natural drift moves all harmonics together.)

## Proposed behaviour
BPDR stops promising silent repair of short notes. Instead, when a reference is supplied, it:
1. Measures each note-span's pitch deviation from the supplied reference (acoustic evidence; owned by BPDR).
2. Flags spans whose deviation exceeds a registered amount for sustained support, with time ranges shown to the user.
3. Offers a bounded local repair only on the user's explicit accept, with the existing rollback, decoded-WAV
   verification and accepted-WAV provenance. Unflagged audio is byte-identical by construction.
4. Without a reference, no flag and no edit (abstain). Held-note repair keeps its existing path and scope.

Reference sources (owned elsewhere; BPDR only consumes): final notes from the notes/durations component, key/mode
grid from the key component, or an explicit user-chosen target. BPDR does not infer or own them.

## What can and cannot be measured
- Measurable now: that the flag reports deviation from the reference correctly (controlled shifts of known size),
  that repaired WAVs reach the reference within tolerance, and that unflagged/known-zero samples are unchanged.
- Not measurable now: whether the reference is musically right, whether users wanted the change, listening quality.
  Those need the review and collection plan (two reviewers, consent); none collected.

## Evaluation to declare before building
- Data: existing 445 pairs (known injected size, a true reference by construction) plus the 53 originals scored against
  their own unshifted reference. Held singers per existing folds. No new fit is required for the first stage.
- Primary checks (declared before scores): deviation-report error vs injected size (cents, singer-macro, per group);
  false flags on originals (should be ~0 when reference = original pitch); repaired-WAV error vs reference;
  unchanged-sample identity outside accepted spans; per-group/width/direction reporting; latency on the V62 path.
- Stopping rule: if deviation reporting error exceeds 10c singer-macro or false flags on originals exceed 5%,
  record failure and do not build the UI. Keep every attempt and unassessed group.

## Risks
- Reference quality bounds everything; a wrong reference produces confident wrong flags. The UI must show the
  reference source and let the user reject it.
- Controlled-shift success is not evidence about natural intended-pitch accuracy.
