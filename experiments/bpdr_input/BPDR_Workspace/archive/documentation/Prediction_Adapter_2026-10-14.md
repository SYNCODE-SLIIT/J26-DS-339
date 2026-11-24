# Frame predictions to permission - 14 October 2026

The engineering connection is implemented: frozen checkpoint predictions become declared candidate edit intervals and pass through the existing permission checks. **Runtime preview grants no edits.** No model was fitted, source audio rendered, accepted repaired package produced or research example created.

## Implemented connection

`prediction_adapter.py` validates the source/grid and aligned 70/71 runtime acoustic features, rejecting reference/target fields. Three evaluation forwards use the saved training normalizer: the main acoustic boundary heuristic and versions shifted earlier/later by 40 ms. Only the 71st column changes; audio, timestamps and the first 70 acoustic columns remain fixed. These variants are sensitivity hypotheses, not calibrated natural note boundaries. Dropout is disabled; repeated evaluation is not treated as uncertainty estimation.

A provisional request threshold of **at least 5 cents** creates candidate intervals from the main curve. Gaps and changes of correction direction split intervals. Short sign-split intervals cannot obtain permission by being combined into a longer run. Corrections are not clipped, smoothed or bridged through rests to increase eligibility. The most disagreeing alternate prediction at each frame is retained. Acoustic transition/voicing-gap and crop-edge guards supplement the existing reliable-voicing, magnitude, support and affected-fraction gates.

Missing uncertainty or expressivity protection blocks edits. Low caller-supplied values do not grant calibrated uncertainty: runtime previews always retain the `UNCERTAINTY_UNCALIBRATED` and `PRODUCTION_PERMISSION_POLICY_UNCALIBRATED` blockers. Only explicitly labelled synthetic-fixture mode can expose a permitted **fixture** plan. Both modes have `production_eligible=False` and `accepted_audio=None`. A future calibrated runtime permission provider is not implemented by this stage.

## Actual saved-checkpoint preview

[Recorded preview](../runs/prediction_preview/9ef4b3ca61bb4b4aac2c341770c81e56/evidence.json) freezes:

- The existing `vocadito_37_p00` accepted training crop, S29, 18.79 seconds; no new separation/GAME/crop selection.
- The historical CPU engineering BPDR checkpoint `9cd3c1a81e274e728a08e5e278bc8790`, trained for 30 updates on eighteen variants from one phrase/singer. Its file SHA and run/input-summary lineage are verified.
- Inference architecture/normalizer source revisions match the saved record. The historical training command differs; no current training-fit equivalence is claimed. The legacy tensor checkpoint has no embedded `data_status`; its role is established by the recorded run bound to its verified SHA.
- Fresh frozen acoustic features on the accepted WAV. No human pitch/notes or injected target is supplied to inference. Training used annotation-derived boundaries; transfer to the runtime heuristic-column distribution remains unvalidated.

The three forwards produce **940 frames** each. Evaluation repeats are exactly equal. The main curve's maximum absolute prediction is about **0.1713 cents**, so zero frames reach the provisional 5-cent request threshold: `NO_EDIT_PROPOSED`, zero candidate intervals, no edit permission and no renderer call. Tiny predictions do not establish that the original singing is suitable or that the model can repair detuning. This seen training phrase and smoke checkpoint are wiring evidence, not independent evaluation or research accuracy.

Compressed raw/view predictions, requested curve, boundary scores, reliable/protected masks and timestamps are saved in `predictions_and_masks.npz`. Input/checkpoint/review/code hashes and source snapshots are recorded and rechecked at completion. The source WAV, checkpoints and reviews are untouched.

## Constructed adapter cases and regression

Six declared synthetic cases separately verify the adapter:

| Supplied case | Outcome |
|---|---|
| Bounded varying 10-to-20-cent curve | One permitted synthetic-fixture interval |
| Opposite directions with adequate individual supports | Two separate permitted synthetic-fixture intervals |
| Opposite directions with short individual supports | Withheld despite an adequate combined span |
| Missing uncertainty/expressivity protection | Runtime edit withheld |
| Supplied small but uncalibrated uncertainty | Runtime edit withheld |
| Subthreshold curve | No edit proposed; suitability remains unknown |

These cases supply predictions and uncertainty masks; they do not execute a learned repair or render a variable curve. Additional unit checks cover acoustic-column preservation, target leakage, saved normalization/evaluation mode, worst-view disagreement, protected transitions and invalid/incomplete predictions. All **102 regression checks passed**: [log](../runs/prediction_adapter_regression_2026-10-14.txt).

## Notebooks and next step

Both existing notebooks reflect this work. Notebook 2 shows the recorded checkpoint preview, actual raw/requested curves, conservative blockers and six synthetic cases. Run All reads the recorded results, without refitting the model or repeating inference/extraction. Existing reference-review, intervention and rollback sections remain intact.

Next engineering step: evaluate a duration-preserving **framewise renderer candidate** capable of consuming a changing correction curve, with independent post-render pitch/preservation and rollback checks. Keep it separate from the laboratory constant-shift generator. In parallel, collect the actual independent original-reference decisions and matched real-audio listening observations, address independent voice coverage, and develop/calibrate uncertainty and permission on appropriate separate data. No production repair or research fitting can be credited from this adapter alone.

Reproduce from the component folder; this creates a new read-only preview evidence directory:

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.prediction_preview
```
