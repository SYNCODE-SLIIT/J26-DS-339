# Centered ALWA arithmetic: fixed training-side WAV comparison

Declared1 November2026 before evaluating this change on the training benchmark.
The prior complete PCM24 diagnosis independently replays all11 development
candidates and33 source/float/encoded measurements. Tiny opposing coarse shifts
of about1e-12c remain with uncentered float64 arithmetic, including one encoded
WAV whose fine shift matches the requested positive correction.

Use the mathematically identical weighted mean around its existing argmax:
`mean_bin = center + sum(w * (index-center)) / sum(w)`.
Compute frequency directly from the A440-relative bin value:
`Hz = 440 * 2**((center - 207 + relative) / 36)`.
This avoids multiplying the dominant weight by a large absolute pitch and
cancelling it through division/subtraction. Keep the same encoder/distributions,
five-bin window, clipping-at-window-edge semantics and weights. This is not
peak-only/categorical decoding, a tolerance, quantization, round-to-zero rule,
changed model input, a fitted parameter or permission to omit opposing frames.

Tests cover constant pitch under changing local peak mass, edge-window behavior,
invalid inputs and agreement with a90-digit weighted reference computed from
exact binary weights. These establish arithmetic behavior, not delivered benefit.

Compare only the frozen reference Transformer on the full200 attempted/108
scoreable/92 excluded training pack and50 original controls. Preserve all648
prediction/curve arrays and216 permission decisions. Same median11/Praat,
YIN,strict per-frame sign checks and all other error/coverage limits. Recheck
actual encoded PCM24 WAVs. Selection must beat the V33 precision baseline's
19.824704787c remaining error with raw gates and all50 originals unchanged;
count alone cannot select and ties retain the baseline. Independently replay
all candidate/encoded measurements with Torch relative-bin arithmetic.

Known-development inspection motivated this engineering change. Reused CV and
development do not become fresh confirmation. Any training-side winner needs
a separate locked default339 known-development replay with the same practical
gates before untouched confirmation. Current V34 remains failed24.744%<25%,
two development groups unassessed, no activation and no independent listening.

Commands: `evaluate_centered_auditor.py`, then `verify_centered_auditor.py`.
Evidence: `results/transformer_usability/2026-11-01_v35_centered_auditor`.
No new model fits, selected/exported model, notebook or shared-demo changes.


## Independently verified training-side result

| Measurement | V33 float64 baseline | Centered float64 |
|---|---:|---:|
| Verified delivered repairs / all attempts | 27/200 | 29/200 |
| Singer-macro remaining delivered error, cents | 19.824705 | 19.585478 |
| Originals delivered unchanged | 50/50 | 50/50 |
| Prediction/curve arrays and permission decisions | Same | Same |

The older float32-auditor reference remains23/200 and20.342142c. All200/108/92
counts are retained. Independent Torch-centered replay verifies all30 candidate
audits and29 encoded WAV audits,648 identical prediction/curve arrays,216
identical decisions and all50 original raw predictions/plans. Six protected
artifacts match and44 focused tests pass. No new model fit or activation.
Training-side selection prefers the centered pipeline under the declared error
criterion; repaired count alone is not the selector. Full practical training
coverage remains limited, including female7's4 valid references among20 attempts.

Three new arithmetic tests pass, including a90-digit reference on exact binary
weights. A separate101-row one-peak scaling demonstration yields constant
frequencies for BOTH implementations; it does not reproduce the observed old
jitter by itself and is retained as such. Do not present that toy as evidence
that scaling alone caused the real failures. Actual delivery and independent
candidate/encoded replay establish the observed pipeline improvement.

The separately locked [known-development qualification](Centered_Development_Qualification_2026-11-01.md)
subsequently passes for its two scoreable groups at11/48 repairs and15.954963c,
12 originals unchanged. This does not validate its two unscoreable groups,
fresh voices or natural performer intention.
