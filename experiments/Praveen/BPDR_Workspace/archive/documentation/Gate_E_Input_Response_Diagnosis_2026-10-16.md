# Gate E input and response diagnosis - 16 October 2026

The failed Jamendo crop **track_0076663 contains the injected pitch-change signal in its stored model inputs**. Its median pitch deltas are -33.333 and +33.333 cents for the -30/+30 injections, with mean absolute errors of 5.869 and 6.181 cents. Only 1.72% and 3.51% of its audited plateau pitch deltas are numerical zero. On 57 plateau frames shared by both variants, the median plus-minus separation is 66.667 cents and no complete feature frames are identical. The crop does not suffer a complete loss of the injected pitch signal. The coarse pitch steps are also visible in the other sources; these are measured estimates, not intended-note labels.

The saved balanced model remains almost unresponsive on that crop: its signed mean C-A response is only +0.00040 cents for the -30 variant and +0.01723 cents for the +30 variant (the latter also has the wrong sign). It should respond by +30/-30 for perfect compensation. The original recovery failures and all eight source/shift outcomes remain unchanged.

Replacing the **explicit pitch column** of C with the original A values changes the balanced model prediction by less than **0.00011 cents** on every tested plateau. Replacing its **64 log-mel columns** changes predictions by about **13.57-15.31 cents** on the six partly repaired variants, but only 0.0072/0.0196 cents on the failed crop. All three saved diagnostic models are probed, not just the balanced arm. Original A and actual C endpoints reproduce saved predictions exactly, and the unchanged-boundary negative control is exact.

These paired-original substitutions and interpolations are **off-distribution diagnostic inputs**. They suggest spectral dependence and a weak response to the tested explicit pitch perturbation, but do not prove why optimization failed, that normalization caused the problem, or that the model ignores pitch in every situation. Different group widths and representations also affect sensitivity. No private detuning target enters the probe features. A paired original is not available during ordinary runtime inference, so these probes are not a proposed repair method.

All **281 regression tests pass**. All **312 sensitivity vectors** reload exactly; eight input-statistic rows, 24 saved residual rows, 48 endpoint vectors and 24 boundary-control identities verify. Parameter tensors remain unchanged; **zero optimizer updates**, no new audio processing, no mask/target/threshold changes, no data rejection and no loss/model adoption occur. The four sources/eight pairs are training diagnostics, not independent accuracy evidence. Research-ready examples remain zero; Gate E validation remains open, D/J4 review deferred and Gate F delivered repair open.

## Input evidence for every selected pair

| Source | Injected cents | Median measured delta | Pitch-delta MAE | Numerical-zero fraction | Normalized pitch RMS | Normalized log-mel RMS per column |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vocadito_37 | -30 | -33.33282 | 7.95970 | 0.08451 | 0.06166 | 0.38247 |
| vocadito_37 | +30 | 33.33368 | 7.43222 | 0.04348 | 0.06574 | 0.36348 |
| track_0076663 | -30 | -33.33331 | 5.86899 | 0.01724 | 0.06475 | 0.29924 |
| track_0076663 | +30 | 33.33331 | 6.18065 | 0.03509 | 0.06347 | 0.28671 |
| track_0347861 | -30 | -33.33325 | 8.74989 | 0.06250 | 0.06936 | 0.38863 |
| track_0347861 | +30 | 33.33313 | 4.81462 | 0.05556 | 0.06539 | 0.35663 |
| track_0368745 | -30 | -33.33362 | 6.41674 | 0.03125 | 0.06253 | 0.54417 |
| track_0368745 | +30 | 33.33266 | 4.73832 | 0.00000 | 0.06523 | 0.54624 |

The plot compares RMS L2 group changes: per-column log-mel RMS multiplied by sqrt(64)=8, versus one pitch column. This describes representation scale; it is not a causal attribution or justification to change stored normalization moments.

## Balanced-model probe detail

| Source | Injected cents | Replace log-mel effect | Replace pitch effect | Actual mean C-A response |
| --- | ---: | ---: | ---: | ---: |
| vocadito_37 | -30 | 14.3645220 | 0.0000185 | 14.3727694 |
| vocadito_37 | +30 | 14.6741295 | 0.0000514 | -14.6603041 |
| track_0076663 | -30 | 0.0072372 | 0.0000969 | 0.0003988 |
| track_0076663 | +30 | 0.0196483 | 0.0001094 | 0.0172321 |
| track_0347861 | -30 | 13.5697803 | 0.0000286 | 13.6239433 |
| track_0347861 | +30 | 15.3126698 | 0.0000605 | -15.3143654 |
| track_0368745 | -30 | 14.3348274 | 0.0000069 | 14.3393431 |
| track_0368745 | +30 | 14.6796284 | 0.0000336 | -14.6746683 |

## Scope and provenance

The measurement protocol was fixed before sensitivity inference, after the prior multi-source outcomes and an initial raw-input inspection. This is explicitly post-hoc diagnosis. It uses the same eight frozen TRAINING pairs, all three final diagnostic model states, nominal-boundary A/C inputs for probes and all saved A/B/C/D residuals. No development or heldout audio is processed. Each group is replaced over the full feature sequence, not only over a private support mask; interpolation uses all features at alpha 0, .25, .5, .75, 1. Private target masks define reporting regions only. All eight groups (64 mel, pitch, confidence, voicing, pitch change, energy, onset, boundary) remain in raw evidence. No waveform is synthesized and intermediate probe points have no physical pitch-correction ground truth.

Directly used input archives, targets, audit receipts, checkpoints, prior-run artifacts, implementation and shared-pipeline files are pinned and reauthenticated. The prior full-corpus verification receipt is retained as historical evidence; this step does not claim a new full-corpus audit or rerun training. Original comparison updates remain 900, earlier diagnostic updates remain 700 and this inference-only step adds zero.

Run: `runs/input_response_diagnosis/950d925de5f643fa936291fc78a1263a`. Protocol: `declaration.json`; raw statistics/sensitivity: `evidence.json`; vectors: `sensitivity_predictions.npz`; replay: `independent_verification.json`; original source snapshot: `input_response_diagnosis_at_declaration.py`. Regression log: `runs/input_response_diagnosis_regression_2026-10-16.txt`.

## Next - Gate E remains open

Gate E: declare a two-arm BPDR-local feature-group weighting ablation, keeping the balanced loss, eight source/shift pairs, initialization, normalization and update budget matched. Compare existing standardized inputs with fixed group-size normalization (64 log-mel columns divided by sqrt(64)=8; scalar groups unchanged). Reuse the verified saved balanced-input control if its full protocol matches, rather than repeating its fit. This is a post-hoc hypothesis from the diagnosis, not a demonstrated fix. Declare before fitting, retain the failed crop and measure all recovery/preservation limits. No common-pipeline change or paired-original/target cue at runtime. Independent human-reference support feasibility remains separate; D/J4 deferred, F open.

The proposed 1/sqrt(group-width) scaling is a design choice derived from feature count, not a fitted gain selected by these outcomes. It changes only a future BPDR model-input adapter after the existing standardizer; no stored dataset features or shared exporter are changed. Its effect must be tested against a matched unchanged-input control before considering a broader comparison. Passing a small training test would still not establish independent recovery accuracy or resolve missing Vocadito paired development.

## Final notebook verification

Both maintained notebooks freshly execute with zero errors, 36 plots and 44 audio controls. Source/executed/HTML/index hashes verify. All 281 regression tests pass. All 312 sensitivity vectors reload exactly, with eight statistic rows, 24 saved residual rows, 48 endpoints and 24 negative controls verified. Parameters and all 28 shared-pipeline files remain unchanged; the new figure was visually inspected. This diagnosis introduces no new fit, data rejection, feature/model adoption or audio processing.

Latest panel: `runs/notebooks/5f95f9f2a06145c9b486fe3c21aeab6d/02_data_review_and_paired_experiments.html`. Proof: `runs/input_response_diagnosis/950d925de5f643fa936291fc78a1263a/final_verification.json`.
