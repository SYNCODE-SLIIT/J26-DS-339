# Transformer correction-response experiment

**Eight CV fits and both WAV comparisons complete; reweighting rejected. MOD-03 remains incomplete.**

The [matched temporal experiment](Transformer_Temporal_Study_2026-10-29.md)
reduced requests on unchanged originals but failed to improve repair accuracy.
Both arms fell below the weakest singer's response-gain requirement. This
experiment tests the training signal on known detuned frames rather than
repeating the jitter loss or changing an output audit.

## Matched intervention

| Arm | Nonzero-target anchor weight | Unchanged-original / outside weight |
|---|---:|---:|
| `matched_baseline` | 1 | 4 / 2 |
| `positive_weight_2` | 2 | 4 / 2 |

Only the known nonzero-target squared-error term changes. The response,
activity-classification and positive-amplitude auxiliary terms stay fixed.
Both arms use the same two-layer joint-context model, 36 windows/update,
initialization, optimizer, 1,200 updates, seed339, feature-only training
controls and four held-out-singer folds. The prior temporal term is absent
from both arms. All source descendants remain with their singer/source.
Paired sampler hashes must match in every fold.

Three targeted tests pass. They verify that weight1 exactly preserves the
original anchor loss, weight2 leaves zero-target gradients unchanged while
doubling the positive gradient in both signs, and invalid factors are refused.
The development evaluator also refused the actual failed V11 model before
loading refits or accessing evaluation data.

The first two baseline checkpoints reproduce the previous V8 joint-augmented
model parameters exactly. The new comparison therefore avoids the floating
point trajectory change caused by V11's larger neighbor-forward batches.
That observation is a reproduction check, not a new accuracy result.

## Selection and evaluation

Both arms first receive held-out prediction checks and actual out-of-fold
encoded-WAV scoring with the fixed median11/Praat condition. Selection prefers
a prediction pass, then unchanged originals, then lower singer-averaged
delivered error, with ties retaining the baseline. More improved cases alone
cannot override worse singer-averaged error. All 200 attempted interventions,
108 scoreable interventions and 50 original controls remain in the reports.

Only a qualified selection may receive three full refits: seeds339/340/341,
2,400 updates each, default339 fixed before development. The saved checkpoint
hashes, selection, original declaration and policy are bound before the known
development replay. A changed lock is refused. A repeat gets a separate
directory and cannot be called new confirmation evidence.

New CSD and male10/male11 confirmation audio stays unopened until a locked
model/policy passes development usability. No model is activated from a raw
prediction result or a unit-test pass. Original exports, existing notebook
edits, the shared demo and sibling components remain protected.

- [Prospective declaration](../results/transformer_usability/2026-10-29_v12_response/declaration.json)
- [Training implementation](../source/bpdr_input/experiments/learning/response_transformer_training.py)
- [Training command](../tools/train_response_transformer.py)
- [Guarded development/replay command](../tools/evaluate_response_transformer.py)

From BPDR_Workspace:

```powershell
& ..\.venv\Scripts\python.exe -X utf8 -B tools\train_response_transformer.py cv
& ..\.venv\Scripts\python.exe -X utf8 -B tools\train_response_transformer.py delivery
& ..\.venv\Scripts\python.exe -X utf8 -B tools\train_response_transformer.py refit
& ..\.venv\Scripts\python.exe -X utf8 -B tools\evaluate_response_transformer.py
```

Later commands enforce their prerequisites. The current experiment is an
engineering loss comparison; it does not establish the proposal's matched
BPDR-objective advantage, independent listening or natural intended-pitch truth.

## Completed result

Positive weight2 worsens raw MAE11.901 ->12.209 cents and actual-WAV error20.342 ->21.681 cents; verified improvements fall23/200 ->17/200, with50/50 original controls unchanged in both arms. The declared rule retains the already known baseline. Its full refits are not duplicated: the new loss did not win, and the prior baseline already failed known-development usability.

[Measured report](../results/transformer_usability/2026-10-29_v12_response/REPORT.md). A single-channel feature diagnostic on the existing weak-singer holdout points to neighbouring-note context contamination. The next current-region input experiment retains the same evaluation cases and audits; it is not a newly narrowed accuracy claim.
