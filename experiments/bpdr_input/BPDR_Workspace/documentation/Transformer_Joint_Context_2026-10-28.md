# Joint pitch-context Transformer experiment

**Status: comparison and development replay complete; usability failed.** This continues MOD-03.
The prior [four-singer failure](Transformer_Usability_Result_2026-10-28.md) stays
unchanged. The demo remains on its existing ExtraTrees selection.

The step-conditioned model can lose all three inputs when its local shape
fitter abstains. The next model receives five continuous acoustic-context
channels alongside three step measurements. It can therefore observe pitch
shape even when no step cue is returned. Provisional boundary channels are
excluded; invariance is by construction, not evidence for the BPDR loss claim.

## Declared comparison

| Arm | Input and training |
|---|---|
| `joint` | Eight channels, two Transformer layers, width 32, 51 context tokens; existing audited real-WAV training pairs and reliable original anchors |
| `joint_augmented` | Same model and loss; 25% of sampled training pairs drawn from additional feature-only local pulses with shared gradual drift and vibrato |

Both arms use four training-side singer/source holdouts, seed 339 and 1,200
updates per fit. No development recording enters fitting or candidate
selection. Existing real-WAV data retain all 200 attempted interventions,
108 scoreable interventions and all construction exclusions. Positive and
unchanged feature augmentations inherit their original training singer/source;
they never cross a fold and never count as new audio recordings or people.

The augmented original retains the injected gradual drift and vibrato. Its
shifted counterpart adds a separately known local pulse; only that pulse is
the correction target. These are pitch-feature controls, not resynthesized WAVs
or evidence that an intentional musical step should be corrected.

Candidate selection first prefers a prediction-gate pass for every represented
training-side singer, then the existing balanced prediction error. Ties prefer
`joint`. Three refits, seeds 339/340/341 with fixed default 339, are allowed only
after a prediction pass. Each refit has a declared 2,400-update budget. A raw
prediction pass still cannot activate a model: actual-WAV development,
unchanged originals, independent confirmation and the existing numerical
audits remain required.

## Independent confirmation limitation

Six archive entries, vowels a/e/i for previously unused male10/male11, are
reserved before acoustic access. They have not been downloaded or analyzed by
this experiment. The current VocalSet inventory has no female singer left
unseen by the broader project. A new independent female cohort is therefore
still required before a both-voice-category usability confirmation. A test on
the two new male singers alone cannot fill that requirement. Previously
inspected final singers must not be relabelled as fresh data.

## Execution and verification

- [Prospective declaration](../results/transformer_usability/2026-10-28_v8_joint/declaration.json)
- [Training implementation](../source/bpdr_input/experiments/learning/joint_transformer_training.py)
- [Model and input contract](../source/bpdr_input/modeling/joint_transformer.py)
- [Execution command](../tools/train_joint_transformer.py)

The three new tests verify that continuous context survives step abstention,
boundary metadata cannot change inputs, feature pulses retain source lineage
and exact signed targets, and a nonzero checkpoint survives reload/batch
inference. All three passed before fitting. These checks do not measure repair
accuracy. Fit receipts, source/input hashes, executed-source snapshots and
held-out measurements are saved under the declared run directory.

From BPDR_Workspace:

```powershell
& ..\.venv\Scripts\python.exe -X utf8 -B tools\train_joint_transformer.py --phase cv
```

The latest-candidate pointer and active-model selection remain unchanged while
the new candidate fails development usability. No later feature work is started.

## Completed measurements

Eight CV fits and three final refits completed. The declared balanced-error
selection chose `joint_augmented`; seed 339 stayed fixed for development.

| Measurement | Previous step model | Joint | Joint + augmentation |
|---|---:|---:|---:|
| Singer-held-out raw MAE, cents | 13.341 | 11.811 | 11.901 |
| Raw scored frames within 10 cents, singer averaged | 56.64% | 63.63% | 61.32% |
| Raw original request, cents | 0.160 | 0.790 | 0.789 |
| Error after common fixed permission policy, cents | 19.591 | 19.773 | 19.586 |

The selected model lowers raw error by about 10.8% against the historical step
model but makes larger raw requests on originals. Its gain after the same
permission policy is negligible. The historical comparator used 1,000 updates;
the joint arms used 1,200, so this is not an isolated architecture ablation.
The two new arms do share their budget.

On the **already inspected development** set, the selected model delivers
**6/48 attempted improvements (6/24 scoreable)**, versus 9/48 for V7. All 12
original WAV controls remain exactly unchanged. Delivered error is
24.846 -> 19.905 cents; raw MAE is 5.236 cents with 89.02% of scored frames
within 10 cents. Female5 fails coverage/gain; male6 passes; female6/male5 have
no scoreable interventions and stay unassessed. Overall usability fails.
Eleven unique renders yield six verified WAVs and five rollbacks. These are
development results, not new final or general singing accuracy.

Three formerly improved cases are lost to rollback. In two of them, plateau
MAE is slightly better, demonstrating why raw accuracy alone is insufficient.
Saved candidate traces and post-render audits point to curve extent, edge
calibration and delivered tracking as the next diagnostic targets. Their
thresholds remain fixed. The next model study should supervise the temporal
extent of corrections and measure edge/outside behavior alongside plateau
accuracy; that study has not been implemented in this iteration.

The first common-policy replay hit a report-metadata `pairs_manifest` error
after computing predictions. The missing manifest field was added and all
three common-policy replays completed. The pre-fix script and correction
receipt are retained; no weights, model selection, policy or audit threshold
changed to repair the report. Original joint CV reports used default
activation 12 cents; the separate common-policy replay uses V7's fixed 8 cents
for all models. This distinction does not affect raw balanced-error selection.

- [Measured comparison report](../results/transformer_usability/2026-10-28_v8_joint/REPORT.md)
- [Locked training-side selection](../results/transformer_usability/2026-10-28_v8_joint/locked_selection.json)
- [Actual development evidence](../results/transformer_usability/2026-10-28_v8_joint/development/evidence.json)
- [Failure diagnosis](../results/transformer_usability/2026-10-28_v8_joint/development_diagnosis.json)

All five development rollbacks fail **COARSE_DIRECTION** alone. Their fine-tracker median movement errors are below one cent. This does not authorize bypassing the audit: next isolate prediction jitter, support-edge extent and tracker/rendering interaction on training-side controls with the current weights frozen before spending another fitting budget. No listener or musical-quality pass is inferred.

[Final artifact-integrity verification](../results/transformer_usability/2026-10-28_v8_joint/technical_execution/verification.json) checks all 11 checkpoint hashes and singer exclusions, all 528 training-input hashes, preserved protected files and failed development evidence. It verifies integrity, not usability.
