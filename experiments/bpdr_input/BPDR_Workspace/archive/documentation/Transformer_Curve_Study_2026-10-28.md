# Frozen Transformer curve and rendering study

**MOD-03 remains active. Curve comparison and development replay complete; usability still fails.**

The [joint-context experiment](Transformer_Joint_Context_2026-10-28.md) improved
raw held-out correction error but lost delivered repairs. Its five development
rollbacks failed only the coarse-tracker per-frame direction check; fine
median movement error stayed below one cent. This study tests whether curve
variation contributes to that mismatch without changing either tracker audit.

## Prospective comparison

The selected joint-augmented weights, singer folds, model inputs and policy are
frozen. Training-side out-of-fold predictions are rendered under three declared
conditions:

| Condition | Correction inside each already permitted support |
|---|---|
| `unchanged` | Existing curve, including its existing five-frame selector filter |
| `median11` | Eleven-frame median filter confined to that support |
| `span_median` | Constant median of the predicted correction in that support |

All three conditions retain exactly the same nonzero frames, direction,
support durations and protected/unreliable exclusions. The transform cannot
increase the maximum requested amplitude. The numerical controller rechecks
the transformed curve. This adds no duration padding, gap bridging, scale
snapping, reference pitch, target interval or learned confidence claim.

Selection uses the training-side actual encoded-WAV result: require original
identity first, then minimize singer-averaged delivered error, with ties
preferring the unchanged curve. Every attempted intervention and construction
failure remains in the report. Original controls include all 50 working
training originals; the five without reliable voiced frames still have an
identity check and are not discarded. Their raw reliable-frame error is
unassessed. No new neural fit is performed.

Only the selected condition may then run on the already inspected IDs5/6
development recordings using fixed final seed339. A new output folder is not
fresh confirmation. The reserved male10/male11 sources remain unopened, and
independent female confirmation is still missing. No condition may activate
the model on the strength of a training or development result alone.

## Implementation and verification

- [Prospective declaration and frozen hashes](../results/transformer_usability/2026-10-28_v9_curve/declaration.json)
- [Support-preserving curve transform](../source/bpdr_input/repair/curve_conditioning.py)
- [Study runner](../tools/run_transformer_curve_study.py)
- [Training-side input curve measurements](../results/transformer_usability/2026-10-28_v9_curve/input_curve_statistics.json)

Four targeted tests pass. They cover sign and amplitude bounds, exact zero
supports across gaps, identity preservation, invalid inputs and an actual
controller replay that retains the exact original permission intervals,
changed-duration fraction and numerical policy. They do not prove audio
quality or repair accuracy.

Run from BPDR_Workspace:

```powershell
& ..\.venv\Scripts\python.exe -X utf8 -B tools\run_transformer_curve_study.py training
& ..\.venv\Scripts\python.exe -X utf8 -B tools\run_transformer_curve_study.py development
```

The first command must finish and save a locked selection before the second
can run. The existing active model/export, checkpoint files, notebook edits,
shared demo and sibling components remain unchanged.

## Completed outcome

The training-side rule selected `median11` by lower singer-averaged delivered error: 24.937 ->20.342 cents, versus20.769 unchanged and20.939 with a constant-span median. Verified training improvements are23/200, versus24/200 unchanged and22/200 constant; all50 originals remain unchanged in every condition. These are repeated conditions on one dataset, not new independent people.

Known development improves from6/48 to8/48 with all12 originals unchanged and delivered error24.846 ->18.048 cents. The female5 error reduction is23.394%, below the unchanged25% requirement; overall usability still fails. No reserved final audio was accessed.

[Measured report](../results/transformer_usability/2026-10-28_v9_curve/REPORT.md). A separate, declared renderer comparison now investigates the same constant predictions under Praat and a phase vocoder, retaining all audits.

## Completed conditional renderer comparison

The [frozen renderer benchmark](../results/transformer_usability/2026-10-28_v10_renderer/REPORT.md) compared Praat and a pinned phase vocoder under the same constant predictions and supports. Phase-vocoder results were worse (16/200 versus22/200 constant-Praat verified improvements). The declared overall rule retained median11/Praat; no phase-vocoder development access, activation or perceptual-quality claim followed. Two additional renderer tests pass, including a real20-cent synthetic-tone movement check and exact preservation outside support. This tone test is technical verification, not real-voice accuracy.

The next model experiment is adjacent-frame temporal supervision with known-valid labels and unchanged-original anchors, compared with the same-budget current model. It is planned, not fitted.
