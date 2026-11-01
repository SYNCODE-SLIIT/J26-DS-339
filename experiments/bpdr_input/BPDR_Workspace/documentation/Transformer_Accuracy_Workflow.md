# Improving and checking Transformer accuracy

Updated 28 October 2026. This workflow is limited to BPDR and uses existing audited
reference-controlled VocalSet pairs. It does not activate a demo model or modify
another member's implementation.

The first complete execution and standalone rendered recheck are documented in
[Transformer improvement results](Transformer_Improvement_2026-10-27.md). All
five objectives fail the training-side all-singer prediction gate; stronger
preservation trades off correction error. The fixed default has no verified
delivered gains on the separate, previously inspected calibration set. A
completed training run is therefore not a model-adoption receipt.

## Current candidate and repeat command

`--latest --render` now follows `results/transformer_usability/latest_candidate.json`
when a completed usability candidate exists, including a failed one. This diagnostic
pointer is separate from `active_model.json`; checking never activates a model.
It currently selects V7 guard and repeats its already-inspected four-singer final:
5/48 attempted improvements, all 12 originals unchanged, failed usability criteria.
A new output folder does not create fresh independent evidence.

The step-context family uses its own saved configuration and policy. To select it
explicitly, run from BPDR_Workspace:

```powershell
& ..\.venv\Scripts\python.exe -X utf8 -B tools\check_held_voice_accuracy.py --run-root results\transformer_usability\2026-10-28_v7_guard --render
```

See the [full outcome and limitations](Transformer_Usability_Result_2026-10-28.md).
The historical five-objective commands below retain their own checkpoint family.
With no completed usability pointer, the legacy latest-experiment behavior remains.

## What accuracy means here

The model predicts a signed correction in **cents**; 100 cents is one semitone.
Use several measurements rather than treating training loss as accuracy:

| Measurement | Desired behavior |
|---|---|
| Raw MAE / RMSE in cents | Small errors against the known compensating correction |
| Within 5 / 10 cents | High proportion of the scored plateau frames within that tolerance |
| Paired response gain | Adding detuning changes the correction in the opposite direction; ideal 1, zero-output model 0 |
| Original mean / p95 request | Small changes requested on unchanged recordings |
| Outside-region request | Small changes requested on known untouched regions |
| Above-30-cent requests | Few over-limit predictions; limits still withhold them rather than clip |
| Boundary sensitivity | Small changes when only metadata changes |
| Permitted error | Accuracy after the existing scope/permission policy, including withheld cases |
| Delivered-WAV before/after error | Actual encoded output improves the known reference-controlled region |

Tolerance percentages describe scored **frames**, averaged by singer; they are
not the percentage of correct recordings or general singing correctness. Unknown
intended notes in arbitrary uploads are not available ground truth.

## Declared improvement experiment

The experiment compares baseline, preservation, continuity, correction-limit and
combined objectives. It keeps the two-layer / width-48 / 81-frame context
Transformer and 300 updates per fit. Preservation doubles original weighting,
adds known-outside weighting and samples additional unchanged-original frames.
Continuity uses target-derivative SmoothL1 at adjacent valid centers; the limit
penalty discourages requests above 30 cents. The existing squared engineering
response terms remain unchanged. This is separate from the proposal-aligned
MAE-response comparison retained in the master plan.

Selection uses four singer-held-out folds **inside the existing training pool**
and seeds 339/340/341: 60 fixed cross-validation fits. All descendants stay with
their singer/source. The three-seed results are averaged within singer, then
across singers; frames/variants/seeds are not additional people. The completely
unscorable training singer remains in attempt/exclusion counts.

First prefer an objective that passes every represented singer's prediction
checks: original mean <=3 cents, improvement over unchanged correction, and
response gain >=0.25. Among passing objectives choose the lowest raw MAE. If
none passes, choose the lowest predeclared balanced development error for
**diagnosis only** and keep the failed gate visible.

Refit the selected objective on all audited training pairs for all three seeds.
Seed **339** is the default checkpoint chosen in advance. Lock all endpoints
before loading calibration features. Report all calibration seeds; never pick
the most favorable calibration seed. Calibration has already been inspected in
earlier project work and remains development evidence, not a fresh final test.

Original/feature/target hashes, source hashes, checkpoints, split lineage and
selection receipts make a saved result auditable. This experiment does not
replace the separate fresh-singer, listening or novice-study obligations.

## Commands

From `BPDR_Workspace`, using the existing component environment:

```powershell
# Train the complete declared comparison and generate reports.
& ..\.venv\Scripts\python.exe -X utf8 -B tools\improve_transformer.py train

# Recheck the latest completed candidate, even when usability failed.
# --render also checks actual encoded WAVs under the existing permission rules.
& ..\.venv\Scripts\python.exe -X utf8 -B tools\check_transformer_accuracy.py --latest --render

# Evaluate a specified improvement checkpoint against other audited pairs.
& ..\.venv\Scripts\python.exe -X utf8 -B tools\check_transformer_accuracy.py `
  --checkpoint "path\to\checkpoint.pt" `
  --pairs-manifest "path\to\pairs.json" --render
```

The checkpoint evaluation command refuses any singer or original source used
in fitting, even if that source appears under a different augmented filename.
It also refuses checkpoints without training-lineage metadata. The existing
historical checkpoints remain readable by their own loaders; the new accuracy
command does not silently pretend their lineage has been verified.

The manifest format is the existing audited pair list: source singer/hash/path,
working A/C WAV paths, saved features, target archive, shift and audit outcome,
or an explicit construction exclusion. References and masks are used for
supervision/scoring only; predictor calls take single-recording context arrays.

## Outputs and interpretation

Every training execution creates a new folder under
`results/transformer_improvement/`. The `latest_experiment.json` pointer is
written **only after the complete matrix, refits and evaluation finish**.
An incomplete run has a declaration/progress record but is not promoted.

- `declaration.json`: prospective arms, folds, seeds, input/source hashes and rules.
- `progress.json`: actual completed cross-validation fit count.
- `cross_validation.json`: all arms, held-out singer results, exclusions and fits.
- `locked_selection.json`: training-side selection, prediction-gate result and fixed seeds.
- `final/seed_*/checkpoint.pt`: reloadable candidate weights with fitted-singer/source lineage.
- `calibration/seed_*/`: separate raw/permission results; seed 339 also evaluates actual WAVs.
- `evidence.json`: completed-run summary and checkpoint identity.
- `index.html`, `REPORT.md`, `cases.csv`: readable results and per-case measurements.
- Each evaluation's `accuracy.json` and `predictions.npz`: detailed metrics, stage traces and reproducible predictions.
- `verification_summary.json`: all-attempt denominators, unique render attempts,
  actual encoded technical passes, rollbacks and delivered improvement counts.

The root report is **cross-validation** evidence; its calibration links are
separate. Singer bootstrap intervals describe a small development cohort;
they are not precision guarantees for the general population. Original
preservation uses the same paired-row weighting as the earlier comparison;
distinct recording counts are also given.

`rendered_waveforms_evaluated=true` means the rendering pipeline was requested.
Read `unique_encoded_verified_wavs` for how many actual encoded files passed,
and read delivered improvement separately. A technical pass can alter an
original or edit outside the injected region, so it is not a repair-benefit pass.
For legacy `--latest`, the manifest is the already inspected calibration;
for the current usability candidate it is the already inspected final manifest;
the standalone report's generic independent-in-fit label does not make it a
fresh final test. A new CLI output folder is a repeat check, not new data.

Three levels remain separate:

1. **Prediction gate**: the learned correction/preservation criteria passed.
2. **Permission and delivered waveform**: the existing controller and actual
   PCM24 output passed; withheld/failed audio remains unchanged in scoring.
3. **Research/user validity**: fresh material, independent listening and intended
   melody evidence where required. The command cannot manufacture these.

If a better raw model remains blocked by long-held scope or span selection, use
the saved trace to continue SVR-01/SVR-03. Do not lower permission thresholds to
improve a displayed accuracy number. The existing demo selection remains intact.
