# Transformer recovery: first matched VocalSet experiment

27 October 2026. Component: BPDR / IT23141506.

## Outcome

A new context-feature Transformer learned the local correction signal and improved
raw calibration prediction over the matched ExtraTrees baseline. The BPDR-loss
condition reached **5.839 cents MAE**, compared with **10.031** for ExtraTrees and
**26.761** for the historical Transformer. Its raw original-preservation result
still needs improvement. The experiment has not activated a demo checkpoint.

These measurements cover **10 audited interventions out of 16 attempts**, from
**two calibration singers and three scoreable original recordings**. Six failed
construction audits remain unscorable. This previously inspected calibration set
is development evidence. It cannot establish general accuracy on new users.

## Current demo connection

`pp1_demo/workers/bpdr_worker.py` loads the complete exported ExtraTrees notebook
run through the guarded BPDR runtime, using an isolated source snapshot. It does
not load `selected_model/bpdr.pt`. The other team's Transformers in the demo are
harmonic planning/refinement models; changing those would not improve BPDR pitch
repair. The existing ExtraTrees export and selection pointers were hash-checked
before and after the experiment.

## What changed in this experiment

The historical encoder takes 70 acoustic columns plus provisional boundary
metadata, with a 128-wide, four-layer Transformer. The new candidate reuses its
`CorrectionEncoder` implementation with a smaller configuration:

| Setting | New candidate |
|---|---|
| Input | Existing 31 single-recording pitch-context features, plus one boundary column where applicable |
| Encoder | Two layers, width 48, four attention heads, feed-forward width 128 |
| Parameters | 45,761 with boundaries; 45,713 without |
| Context | Every output is the center of an 81-frame window: 1.62 seconds at the 20 ms grid |
| Context features | Include the existing multiscale pitch statistics up to 2.42 seconds; derived from the current recording |
| Output | Signed cents, with a 30-cent numerical training scale and zero-initialized output head |
| Optimizer | AdamW, learning rate 0.0005, weight decay 0.001, gradient clipping 1 |
| Budget | 300 updates per final condition, seed 339, CPU |
| Sampling | Equal singer probability, then equal audited-pair probability; eight pairs per update |
| Supervision | Approximately 50% original, 35% shifted plateau, 15% audited outside in expectation |
| BPDR losses | Normalized squared anchor; add 0.25 paired response and 0.1 boundary consistency after 25% warm-up |

Training sees target labels only in the loss and sample selection. Inference
receives the current recording's feature sequence; it has no clean reference,
injected interval, singer identity or target mask. Padding is explicit and
separate from genuine pauses. The same centered context is used during fitting
and inference.

The 31 context columns are the existing ExtraTrees input features, including its
tuning-relative and multiscale descriptors. This is a declared feature and
architecture redesign. It does not establish that attention alone caused the
improvement, or that each change independently improved the original model.

## Data and checks

- Training: 64 attempted interventions, 43 audited, seven represented singers.
  The original eight-singer training selection remains unchanged; one singer
  supplied no audited interventions.
- Sanity fit: one training pair, 100 updates. Raw error fell from 20.000 to
  0.448 cents; response gain was 0.969. This measures learning on a fitted example.
- Training-pool diagnostic: withhold female4 and male4, fit for 150 fixed updates.
  Twelve of 16 attempts were scoreable. Error fell from 25.143 to 10.754 cents;
  original raw alteration was 5.418 cents. These singers were then returned to
  the full training pool for the final fits.
- Calibration: female5 and male5 remain outside every fit. All three final
  checkpoints were saved before calibration feature loading. No calibration
  checkpoint selection, retuning or final-pilot access occurred.
- Singer and source hashes were checked for train/calibration overlap. Input
  files and existing selected/demo artifacts were checked for changes.
- Checkpoint save/reload was verified for every fit. Four added tests passed,
  covering center alignment, padding, signed outputs, checkpoint loading and
  rejection of invalid feature arrays.

## Same-case raw prediction results

Values are averaged within singer and then equally across singers. The tolerance
column describes scored plateau frames, not recordings or a general singing
accuracy percentage. Original alteration describes raw predictions, not edits
delivered to the user. Repeated original anchors occur in the paired rows.

| Model | Local MAE, cents | Frames within 10 cents | Original raw alteration, cents | Paired response gain |
|---|---:|---:|---:|---:|
| Historical Transformer | 26.761 | 0.35% | 1.265 | -0.040 |
| Matched ExtraTrees | 10.031 | 55.41% | 3.821 | 0.597 |
| New Transformer, augmentation only | 5.644 | 90.21% | 4.174 | 0.882 |
| New Transformer, boundary column excluded | 5.652 | 85.55% | 5.365 | 0.847 |
| New Transformer, BPDR losses | 5.839 | 83.62% | 4.066 | 0.879 |

The predefined BPDR condition has approximately **41.8% lower raw MAE than
ExtraTrees** in this calibration comparison. All new Transformer conditions
improve correction estimates. The augmentation-only condition is slightly better
on correction error; the BPDR condition requests slightly smaller original
changes. This single seed does not establish a BPDR-objective advantage.

The boundary-excluded condition still uses self-attention over a temporal window;
its name follows the historical `frame_only` comparison condition. Its boundary
invariance is guaranteed by excluding that column. For the BPDR condition, mean
prediction sensitivity to the two boundary alternatives was 0.207 cents.

## Decision and next improvements

The supplementary permission replay is complete. With the old comparison's fixed
gain 1 and 3-cent deadband, seven of ten scoreable detuned proposals were withheld,
two were identity proposals, and one nonidentity proposal edited outside the
scored shifted plateau. **None improved the scored shifted region:** permitted
MAE remained 25.833 cents, equal to unchanged audio. All original proposals were
withheld, giving zero permitted original alteration. This is a curve-level
result; no WAV rendering was performed.

Recorded detuned-proposal reasons included correction-limit failures (two cases),
protected-region conflicts (three), and insufficient continuous support (four);
reasons can overlap. Raw prediction accuracy has improved, but the predictions
are not yet yielding useful permitted repairs. Keep the current demo selection
while developing this interface. Do not relax the controller simply to make
these calibration examples pass.

The Transformer is now a credible development candidate. The next priority is
**preserving originals and producing verified audible improvements**. Increasing
model size or simply training longer is not yet justified by this result.

1. **Preservation experiment:** prospectively declare a stronger original/outside
   penalty and broader original-only examples. Compare fixed settings on
   singer-held-out folds within the training pool. Use paired response alongside
   original alteration so a zero-output model cannot win by doing nothing.
   Include correction-limit penalties and continuity/support diagnostics on the
   training side. Check how input-derived transition protection interacts with
   candidate repair spans; any revised boundary policy needs its own transition
   preservation controls.
2. **Broader training and evaluation:** include multiple notes, shorter notes,
   different intervention positions and widths, both correction signs, pauses,
   vibrato and legitimate transitions. Audit augmentation construction and keep
   every failed attempt in the denominator. Keep all variants of a source/singer
   within one split.
3. **Fair architecture comparison:** add a newly fitted MLP with the same features,
   sampling and targets, and repeat fixed neural conditions over several seeds.
   This tests whether attention adds value beyond the engineered context.
4. **End-to-end repair check:** freeze a candidate and operating policy, then
   apply the existing scope, controller and renderer. Re-analyze the delivered
   WAV, count improvements/withholds/regressions, inspect unchanged originals,
   and perform listening review before introducing an optional demo selection.
5. **New evaluation material:** use untouched singers and recordings after model
   and policy choices are locked. Natural intended-note correctness and downstream
   note precision/recall/F1 need suitable independent reference material.

The existing failed research gates remain failed. This experiment measures raw
controlled-correction prediction and does not claim natural musical correctness,
listening approval, or improved harmonization.

## Reproduce and inspect

From the repository root `J26-DS-339`:

```powershell
& .\experiments\bpdr_input\.venv\Scripts\python.exe -B `
  .\experiments\bpdr_input\BPDR_Workspace\tools\train_context_transformer.py

& .\experiments\bpdr_input\.venv\Scripts\python.exe -B `
  .\experiments\bpdr_input\BPDR_Workspace\tools\evaluate_context_permissions.py
```

Every training execution creates a new run. The experiment pointer
`results/transformer_recovery/latest_experiment.json` records the latest completed
experiment; the demo loader does not read that pointer.

This run: `results/transformer_recovery/760a6f1745c94263a5b9a51b7dafffd5/`.

- `declaration.json`: settings, source hashes, input hashes and protected artifacts.
- `data_counts.json`: training attempts and audited counts by singer.
- `sanity.json` and `training_diagnostic.json`: training-side checks.
- `locked_endpoints.json`: fixed endpoint fits and reload checks.
- `evidence.json`: all raw calibration rows, per-singer summaries and exclusions.
- `augmentation_only/checkpoint.pt`, `frame_only/checkpoint.pt`,
  `bpdr/checkpoint.pt`: saved candidates; each is marked ineligible for the demo.
- `permission_audit.json`: supplementary replay of BPDR predictions through the
  existing controller at the old comparison's fixed gain 1 / 3-cent deadband.
  This is a proposal-level audit; it does not render audio.
- `bpdr_permission_predictions.npz`: frozen raw and permitted curves for that audit.

Implementation: `source/bpdr_input/modeling/context_transformer.py`.
Experiment runner: `tools/train_context_transformer.py`.
Tests: `tests/test_context_transformer.py`.
