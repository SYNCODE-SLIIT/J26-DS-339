# Gate E multi-source preservation diagnosis - 16 October 2026

The declared multi-source preservation diagnostic is complete: **four existing training sources / eight audited pairs**, one Vocadito recording and three Jamendo crops from three different artist proxies, each with both -30 and +30-cent injections and the existing split-boundary corruption. All source/shift outcomes are retained. The same model initialization, normalizer, full phrase inputs, weighted batches and fixed 100-update budget are used for all three losses. Vocadito and Jamendo receive equal total training and reporting weight. There are no answer-cue inputs in this diagnostic.

The new preservation arm adds **2 times the region-balanced A/B original correction magnitude** to the previous balanced loss. This coefficient is fixed before running; no search or best-checkpoint selection is performed. Target masks, correction values, original WAVs, partitions and shared pipeline remain unchanged. Response losses are active from update 1 with no warmup, so this remains separate from the original 900-update corpus comparison.

The provisional engineering rule requires each pair to have detuned plateau error at most 27 cents (10% better than zero), original plateau alteration at most 1 cent and detuned outside alteration at most 1 cent. These are declared triage limits, not human musical approvals or evidence of original intonation correctness. A strong relative response alone cannot pass this rule.

**Final update 100, equal-dataset macro:**

| Loss | Plateau error (cents) | Original plateau alteration (cents) | Outside alteration (cents) | Pairs passing all limits |
| --- | ---: | ---: | ---: | ---: |
| Existing loss | 29.9987 | 0.0467 | 0.0461 | 0/8 |
| Balanced regions | 17.9073 | 0.3343 | 0.1725 | 5/8 |
| Balanced + original protection | 29.9993 | 0.0017 | 0.0023 | 0/8 |


No loss passes all eight provisional training-pair limits. This bounded test does not demonstrate preservation-safe recovery across the selected sources and directions. Inspect source/direction recovery and feature sensitivity before another fit; neither the region-balanced nor stronger-preservation loss is adopted. One-source learnability does not establish that a shared model can learn these multiple sources and both directions within this budget.

All 274 regression tests pass. Final raw metrics and all 96 GPU prediction vectors are independently replayed/reloaded; exact counts and tolerances are recorded in the verification artifact. The 300 updates here are separate from the earlier 400-update single-source diagnostic and original 900-update comparison. All are preserved. This is one seed and eight repeated training pairs, not new independent observations. Vocadito paired development remains unavailable; research-ready examples remain zero. No loss or model is adopted and no learned audio is rendered. Gate E stays open, D/J4 review deferred, Gate F delivered repair open.

| Dataset | Frozen source | Group identifier |
| --- | --- | --- |
| vocadito | vocadito_37 | vocadito:S29 |
| jamendo | track_0076663 | jamendo:artist_007805 |
| jamendo | track_0347861 | jamendo:artist_340710 |
| jamendo | track_0368745 | jamendo:artist_351337 |


## Dataset and direction detail

### Existing loss

| Dataset or injected shift | Plateau error | Original plateau alteration | Outside alteration |
| --- | ---: | ---: | ---: |
| vocadito | 29.99831 | 0.04699 | 0.04666 |
| jamendo | 29.99899 | 0.04639 | 0.04561 |
| -30.0 | 29.95124 | 0.04668 | 0.04612 |
| 30.0 | 30.04606 | 0.04671 | 0.04614 |

| Source | Injected cents | Plateau error | Original plateau alteration | Outside alteration | All limits pass |
| --- | ---: | ---: | ---: | ---: | --- |
| vocadito_37 | -30 | 29.95079 | 0.04701 | 0.04664 | False |
| vocadito_37 | +30 | 30.04584 | 0.04698 | 0.04667 | False |
| track_0076663 | -30 | 29.95292 | 0.04706 | 0.04593 | False |
| track_0076663 | +30 | 30.04723 | 0.04703 | 0.04595 | False |
| track_0347861 | -30 | 29.95117 | 0.04764 | 0.04510 | False |
| track_0347861 | +30 | 30.04868 | 0.04788 | 0.04510 | False |
| track_0368745 | -30 | 29.95096 | 0.04435 | 0.04577 | False |
| track_0368745 | +30 | 30.04297 | 0.04441 | 0.04578 | False |
### Balanced regions

| Dataset or injected shift | Plateau error | Original plateau alteration | Outside alteration |
| --- | ---: | ---: | ---: |
| vocadito | 15.48499 | 0.22568 | 0.14771 |
| jamendo | 20.32955 | 0.44299 | 0.19730 |
| -30.0 | 17.81780 | 0.34127 | 0.17542 |
| 30.0 | 17.99674 | 0.32741 | 0.16959 |

| Source | Injected cents | Plateau error | Original plateau alteration | Outside alteration | All limits pass |
| --- | ---: | ---: | ---: | ---: | --- |
| vocadito_37 | -30 | 15.40339 | 0.22425 | 0.14905 | True |
| vocadito_37 | +30 | 15.56659 | 0.22711 | 0.14637 | True |
| track_0076663 | -30 | 29.87961 | 0.12000 | 0.16178 | False |
| track_0076663 | +30 | 30.13717 | 0.11987 | 0.16105 | False |
| track_0347861 | -30 | 15.40772 | 1.00219 | 0.16049 | False |
| track_0347861 | +30 | 15.56519 | 0.90919 | 0.16473 | True |
| track_0368745 | -30 | 15.40927 | 0.25268 | 0.28309 | True |
| track_0368745 | +30 | 15.57832 | 0.25404 | 0.25263 | True |
### Balanced + original protection

| Dataset or injected shift | Plateau error | Original plateau alteration | Outside alteration |
| --- | ---: | ---: | ---: |
| vocadito | 29.99932 | 0.00253 | 0.00224 |
| jamendo | 29.99935 | 0.00090 | 0.00235 |
| -30.0 | 30.00093 | 0.00171 | 0.00231 |
| 30.0 | 29.99774 | 0.00171 | 0.00228 |

| Source | Injected cents | Plateau error | Original plateau alteration | Outside alteration | All limits pass |
| --- | ---: | ---: | ---: | ---: | --- |
| vocadito_37 | -30 | 30.00184 | 0.00253 | 0.00226 | False |
| vocadito_37 | +30 | 29.99681 | 0.00252 | 0.00222 | False |
| track_0076663 | -30 | 30.00083 | 0.00074 | 0.00246 | False |
| track_0076663 | +30 | 29.99900 | 0.00075 | 0.00243 | False |
| track_0347861 | -30 | 30.00033 | 0.00096 | 0.00259 | False |
| track_0347861 | +30 | 29.99903 | 0.00097 | 0.00259 | False |
| track_0368745 | -30 | 29.99890 | 0.00098 | 0.00203 | False |
| track_0368745 | +30 | 29.99800 | 0.00097 | 0.00199 | False |

## Protocol and limits

Sources are chosen solely by frozen catalog order: the first Vocadito source and first three distinct Jamendo artists with both audited +/-30 split variants. Each update contains all eight pairs, with weights .25 for each Vocadito pair and 1/12 for each Jamendo pair. The existing 71-input model has 128 hidden units, four layers/four heads, feed-forward width 512 and dropout .1. AdamW uses learning rate .001, weight decay .01, gradient clipping 1 and seed 339. Full phrase context and original training-only normalization are retained. Initial tensors are saved for each arm and compared exactly. The final update is reported; intermediate checkpoints are descriptive metrics only. No development, PP1 evaluation or final-holdout acoustics are processed.

Preservation objective: balanced anchor + boundary invariance + counter-detuning + .1 smoothness + 2 original protection. Original protection is the A/B mean absolute correction, equal-weighted between audited plateau and outside regions, averaged per example. Existing and balanced controls use their earlier exact definitions. Their objective magnitudes are not directly comparable; each has a recorded analytic zero baseline. Only runtime inputs are used; private labels occur solely in the loss and audit/evaluation masks.

Run: `runs/multisource_control/0e07cb1c4e9247d9a8dc9c2f032241a6`. Immutable protocol: `declaration.json`; outcomes: `evidence.json`; replay: `independent_verification.json`; per-pair history/raw predictions and diagnostic-only parameter states remain in each arm folder. Tests: `runs/multisource_control_regression_2026-10-16.txt`. Execution log: `runs/multisource_control_execution_2026-10-16.txt`.

## Next - Gate E remains open

Gate E: diagnose source/direction failures and runtime-input sensitivity on the saved eight pairs without another fit. Quantify whether +/- injections remain distinguishable in the measured pitch/spectral inputs and inspect prediction residuals. Retain failing pairs; do not remove data solely because the model performs poorly. Separately design reference-feasible supports retaining trusted human interiors and explicit audits. No main comparison rerun until a new declaration is justified. D/J4 deferred; F open.

## Final notebook verification

Both maintained notebooks freshly execute with zero errors, 35 plots and 44 audio controls. Source/executed/HTML/index hashes verify. All 274 regression tests pass. All 24 final metric rows and 96 GPU prediction vectors are replayed/reloaded within the declared tolerance. Exactness counts remain in the final verification receipt. All 28 shared-pipeline files remain unchanged; the new figure was visually inspected. No model/loss adoption or learned audio rendering occurs. The learning diagnostic uses only the declared training partition.

Latest panel: `runs/notebooks/9a1dd3d40084439ca835d3fbd4f6fc9c/02_data_review_and_paired_experiments.html`. Proof: `runs/multisource_control/0e07cb1c4e9247d9a8dc9c2f032241a6/final_verification.json`.
