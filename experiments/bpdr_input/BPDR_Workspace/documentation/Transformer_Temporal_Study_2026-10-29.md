# Adjacent-frame Transformer training experiment

**Status: eight CV fits and both actual-WAV comparisons complete; no usable candidate. MOD-03 remains active.**

The preceding [frozen-curve and renderer studies](Transformer_Curve_Study_2026-10-28.md)
raised known-development repairs from 6/48 to 8/48 but did not meet usability.
The alternate renderer lost its training-side comparison. This experiment
returns to model training, testing temporal supervision against a matched
baseline instead of adding more renderer changes.

Both arms retain the two-layer, width-32 joint-context Transformer, the same
audited real-WAV pairs, original anchors and feature-only training controls.
Both forward 108 windows per update: 12 source-balanced groups, each containing
an original/shifted/original-anchor triple, with three adjacent centers per
entry. Only the temporal arm adds the neighbor-difference loss. The baseline
still forwards the same neighbors, keeping computation and center sampling
matched. Initialization, deterministic sampling, optimizer, 1,200-update budget,
four singer/source folds and seed 339 are identical. Saved sampler hashes must
agree in every paired fold.

The temporal term supervises changes in the correction, relative to changes in
the known target. It does not force genuine labelled transitions flat. Both
adjacent labels must be valid; unknown ramps, unreliable observations and
clipped neighborhood boundaries cannot contribute. The fixed five-cent loss
scale and weight 0.1 are training hyperparameters, not relaxed audit limits.
The same original/outside, response, activity and amplitude losses remain.

## Selection and gates

After all eight CV fits finish, both arms undergo actual-WAV out-of-fold
evaluation with the existing median11/Praat condition and unchanged numerical
and encoded-WAV audits. Selection first requires the prediction gate, then
original identity, then minimizes singer-averaged delivered error. Ties prefer
the matched baseline. Jitter reduction alone cannot select a zero-output
model or establish repair benefit. No development recording enters fitting or
selection. Three 2,400-update refits, seeds 339/340/341 with default 339, are
allowed only after prediction and original-preservation gates pass.

Three targeted loss tests pass: unknown-ramp labels do not create gradients,
correct known transitions are not flattened, and incorrect jitter on a known
plateau creates a corrective gradient. These tests are separate from model
accuracy and delivered-audio usability.

- [Prospective declaration](../results/transformer_usability/2026-10-29_v11_temporal/declaration.json)
- [Training implementation](../source/bpdr_input/experiments/learning/temporal_transformer_training.py)
- [Reproducible execution command](../tools/train_temporal_transformer.py)

```powershell
& ..\.venv\Scripts\python.exe -X utf8 -B tools\train_temporal_transformer.py cv
& ..\.venv\Scripts\python.exe -X utf8 -B tools\train_temporal_transformer.py delivery
& ..\.venv\Scripts\python.exe -X utf8 -B tools\train_temporal_transformer.py refit
```

## Independent confirmation reservation

The new female-source reservation uses CSD1.1: one professional female singer,
according to the [dataset authors](https://github.com/emotiontts/emotiontts_open_db/tree/master/Dataset/CSD).
The current [official deposit](https://zenodo.org/records/4916302) identifies
CC BY-NC-SA 4.0 and archive MD5 `74d121dd8706fded26a15526a379f7a2`.
No full archive was downloaded or full-archive MD5 verified. Individual MIDI
members passed CRC checks during the metadata-only intake.

The fixed rule chooses the first three English a-key recordings whose
published score has a note at least 1.8 seconds long between 0.4 and 19.6
seconds in a fixed first-20-second crop. It selected `en006a.wav`, `en010a.wav`
and `en023a.wav`. This uses score duration only to match the declared held-note
scope; score pitch/timing never enters the predictor or repair permission.
Measured audio eligibility and intervention audits still apply later. A failed
chosen source stays in the denominator and cannot be replaced.

Only metadata and 63,853 bytes of MIDI archive ranges were transferred during
selection, in addition to the separately saved 174,901-byte ZIP directory and
official record metadata. No audio, F0 extraction, intervention, listening or
inference occurred. All three files represent **one singer**. The reservation
joins the already reserved male10/male11 VocalSet sources; it does not mean
the confirmation test has passed or that biometric identity independence has
been verified. Audio access remains gated on a locked model/policy and passing
development usability. Dataset attribution and derivative-use terms are kept
with the intake records; no model or data is published in this work.

- [Reservation declaration](../results/transformer_usability/confirmation_sources_2026-10-29/csd_confirmation_declaration.json)
- [Exact chosen members and MIDI checks](../results/transformer_usability/confirmation_sources_2026-10-29/csd_reservation.json)
- [Bounded metadata-intake script](../tools/reserve_csd_confirmation.py)

Other investigated corpora were not adopted: PJS is a male singer and cannot
fill the female-singer confirmation gap; Kiritan/OpenCpop and GTSinger require
different source/access/terms decisions. Those investigations supplied no
training examples or evaluation audio.

## Completed prediction comparison

Both paired samplers match in all four folds. Matched-baseline raw MAE is12.171 cents versus12.194 with temporal supervision. Mean raw-original request improves0.794 ->0.571 cents, and original p95 improves4.211 ->2.725 cents. Known-plateau adjacent-frame error rises0.556 ->0.570 cents, so the intended jitter-reduction benefit is not established. Both arms fail the weakest singer response gate (male7:0.24668 and0.24474, required>=0.25). The criteria remain unchanged. Actual-WAV comparison is retained for diagnosis; no final refit is unlocked by these raw results.

The current engineering objective is separate from the proposal-aligned BPDR/augmentation-only/frame-only comparison. Boundary invariance remains by input construction; no learned-method superiority or general singing accuracy is claimed.

## Completed actual-WAV comparison

The matched baseline verifies18/200 attempts and temporal supervision22/200, with all50 original WAV controls unchanged in both arms. Singer-averaged delivered error is21.499 versus21.553 cents; the declared rule retains the baseline as a diagnostic choice, not an adopted model. Neither arm passes the prediction gate. Full refits and development access remain blocked by those failed criteria; the refit command was actually tested and refused.

The new loss reduces raw-original requests but does not reduce measured jitter or establish better repair. Next investigate response calibration, keeping the current source/singer splits and output audits fixed, and distinguish learned activity from the separate shape-fit veto. No further refits of the current failed setting are justified.

[Full measured report](../results/transformer_usability/2026-10-29_v11_temporal/REPORT.md). Seventeen focused tests passed together; no full-suite, listener or general singing-accuracy claim is made.

[Artifact-integrity and gate verification](../results/transformer_usability/2026-10-29_v11_temporal/technical_execution/verification.json) binds the eight checkpoints, all paired sampling hashes, frozen inputs, actual-WAV evidence, preserved exports and metadata-only reservation. It verifies the failed outcome rather than declaring usability.
