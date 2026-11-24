# Gate C complete fixed-support development audit for Gate E - 16 October 2026

The fixed-support development audit is complete for **all 14 frozen Vocadito development phrases**. The full inventory contains **28 phrase/direction keys: 24 measured shifts (7 pass, 17 fail) and four unattempted keys from two geometry-refused phrases**. Twenty shifts were newly measured; all four pilot attempts, including both failures, were authenticated and reused without rerendering or re-extraction.

The seven passes cover **six phrases, three recordings and three singers**, with two negative and five positive injections. **vocadito_5_p01 is the only phrase passing both -30 and +30 cents**. The other passes are vocadito_2_p00 (-30), vocadito_2_p01 (+30), vocadito_5_p00 (+30), vocadito_19_p00 (+30) and vocadito_19_p01 (+30). These are controlled-shift delivery audits using human-referenced original masks, not learned-repair accuracy or a representative success percentage. Boundary variants and crop repeats must not be counted as independent recordings.

All ten new unchanged-audio checks reproduce every cached acoustic array and original YIN track exactly; together with the two reused controls, twelve source identities are verified. All 24 measured shifts preserve sample count, float-WAV roundtrips and samples outside the fixed support exactly. Outside audit coverage is 100%, with zero median and 90th-percentile pitch drift for all 24. Original masks, guards, minimum counts, confidence threshold, renderer and fine/coarse audit limits remain unchanged.

All seventeen measured failures remain retained and have all-false anchor masks and zero targets. Failure flags overlap: eight attempts fail the minimum plateau-frame count, six fail plateau coverage, seven fail the fine 90th-percentile error limit and six fail the fine median-error limit. Two attempts have no common plateau frames; their fine and coarse pitch metrics are missing and remain explicit. There is also one coarse median-error flag and one fine-direction flag. These flag counts do not sum to the seventeen failed attempts. The original vocadito_9_p00 / vocadito_9_p01 geometry refusals remain unattempted in both directions.

**313 regression tests pass**. Independent verification replays all 24 original audits, separate threshold decisions, private targets and exact outside waveforms, plus a 240-row trusted-plateau frame trace. The four reused pilot records and their producing artifacts remain exact. The 28 pinned common-pipeline files and all directly used original inputs are unchanged; no shared pipeline or reserved PP1/final-holdout acoustics are executed.

This completed run creates thirty diagnostic WAVs (ten identity controls and twenty synthetic injections), not learned repair outputs. It creates **zero paired examples and zero optimizer updates**. The existing prepared pool stays at 936 training variants and 285 Jamendo-only development variants. The seven passing human-reference shift cases await paired-view packaging; 21 boundary variants are a plan, not a completed dataset. The original comparison remains 900 optimizer updates and earlier diagnostics 800. This engineering audit step is complete; Gate C research eligibility is not certified, Gate E validation remains open, D/J4 musical review is owner-deferred, Gate F open and research-ready examples zero.

## Every declared outcome

| Phrase | Injected cents | Origin | Status | Common / original plateau frames | Fine median / p90 error (cents) | Decision |
| --- | ---: | --- | --- | --- | --- | --- |
| vocadito_2_p00 | -30 | verified_pilot_reuse | measured | 10 / 10 | 1.440 / 3.381 | pass |
| vocadito_2_p00 | +30 | verified_pilot_reuse | measured | 10 / 10 | 1.883 / 1202.733 | FINE_P90_ERROR |
| vocadito_2_p01 | -30 | new_completion | measured | 10 / 10 | 5.085 / 6.795 | FINE_MEDIAN_ERROR |
| vocadito_2_p01 | +30 | new_completion | measured | 10 / 10 | 0.855 / 3.642 | pass |
| vocadito_5_p00 | -30 | verified_pilot_reuse | measured | 9 / 10 | 0.541 / 0.966 | PLATEAU_FRAME_COUNT |
| vocadito_5_p00 | +30 | verified_pilot_reuse | measured | 10 / 10 | 0.896 / 2.593 | pass |
| vocadito_5_p01 | -30 | new_completion | measured | 10 / 10 | 0.452 / 2.528 | pass |
| vocadito_5_p01 | +30 | new_completion | measured | 10 / 10 | 1.547 / 3.207 | pass |
| vocadito_7_p00 | -30 | new_completion | measured | 5 / 10 | 1.515 / 4.342 | PLATEAU_FRAME_COUNT, PLATEAU_COVERAGE |
| vocadito_7_p00 | +30 | new_completion | measured | 0 / 10 | missing / missing | PLATEAU_FRAME_COUNT, PLATEAU_COVERAGE, FINE_MISSING, COARSE_MISSING |
| vocadito_9_p00 | -30 | geometry_refusal | not_attempted_geometry_refused | not measured | not measured | GEOMETRY_REFUSED |
| vocadito_9_p00 | +30 | geometry_refusal | not_attempted_geometry_refused | not measured | not measured | GEOMETRY_REFUSED |
| vocadito_9_p01 | -30 | geometry_refusal | not_attempted_geometry_refused | not measured | not measured | GEOMETRY_REFUSED |
| vocadito_9_p01 | +30 | geometry_refusal | not_attempted_geometry_refused | not measured | not measured | GEOMETRY_REFUSED |
| vocadito_16_p00 | -30 | new_completion | measured | 1 / 10 | 4.687 / 4.687 | PLATEAU_FRAME_COUNT, PLATEAU_COVERAGE, COARSE_MEDIAN_ERROR |
| vocadito_16_p00 | +30 | new_completion | measured | 2 / 10 | 0.995 / 1.072 | PLATEAU_FRAME_COUNT, PLATEAU_COVERAGE |
| vocadito_19_p00 | -30 | new_completion | measured | 9 / 10 | 3.843 / 7.909 | PLATEAU_FRAME_COUNT |
| vocadito_19_p00 | +30 | new_completion | measured | 10 / 10 | 1.056 / 6.717 | pass |
| vocadito_19_p01 | -30 | new_completion | measured | 10 / 10 | 4.514 / 15.919 | FINE_P90_ERROR |
| vocadito_19_p01 | +30 | new_completion | measured | 10 / 10 | 2.815 / 7.556 | pass |
| vocadito_21_p00 | -30 | new_completion | measured | 0 / 10 | missing / missing | PLATEAU_FRAME_COUNT, PLATEAU_COVERAGE, FINE_MISSING, COARSE_MISSING |
| vocadito_21_p00 | +30 | new_completion | measured | 4 / 10 | 2.171 / 2.688 | PLATEAU_FRAME_COUNT, PLATEAU_COVERAGE |
| vocadito_21_p01 | -30 | new_completion | measured | 10 / 10 | 6.401 / 8.503 | FINE_MEDIAN_ERROR |
| vocadito_21_p01 | +30 | new_completion | measured | 10 / 10 | 4.469 / 10.582 | FINE_P90_ERROR |
| vocadito_34_p00 | -30 | new_completion | measured | 10 / 10 | 8.865 / 1275.389 | FINE_MEDIAN_ERROR, FINE_P90_ERROR |
| vocadito_34_p00 | +30 | new_completion | measured | 10 / 10 | 7.352 / 1903.379 | FINE_MEDIAN_ERROR, FINE_P90_ERROR |
| vocadito_34_p01 | -30 | new_completion | measured | 10 / 10 | 5.888 / 1202.268 | FINE_MEDIAN_ERROR, FINE_P90_ERROR |
| vocadito_34_p01 | +30 | new_completion | measured | 10 / 10 | 1200.799 / 1204.473 | FINE_MEDIAN_ERROR, FINE_P90_ERROR, FINE_DIRECTION |

## Validation assessment: share with caveats

The declaration predates all twenty new shifts and locks the full 28-key inventory. The four original pilot records are reused, not rerun; the geometry evidence fixes each physical support. Passing subset counts must always be shown beside the 17 failures and four unattempted keys. The training reference-mask inventory is separate and is not expanded by this development-only step.

YIN and PESTO are estimates, not human musical verification. Audit success means this controlled injection meets the retained technical rules at masked positions. It does not validate intended sung notes, perceptual quality, generalization or a repair model. The passing subset is selected by technical audit and has only three recording/singer groups, with direction imbalance (two negative/five positive). Evaluation must retain these limits, use recording/singer-aware summaries and preserve original no-change controls. No conclusion about full-corpus accuracy can be derived from seven successful injections.

All original audit results replay exactly from saved arrays. Independent scalar/boolean statistics use a 0.001-cent numeric comparison tolerance to accommodate float32 versus float64 arithmetic; that verification tolerance does not alter any audit decision threshold. The two empty common-plateau cases remain missing metrics, not zero-error cases. All failed private targets remain zero. Existing masks are not edited to remove difficult frames.

Run: `runs/support_development_audit/527f39f955af4e0b894d3674d9d761ea`. `declaration.json` locks all inputs and twenty new keys; `evidence.json` retains all fourteen sources and twenty-eight keys; `independent_verification.json`, `outcome_table.json` and `plateau_frame_trace_NOT_MODEL_INPUT.json` preserve independent checks and diagnoses. Regression log: `runs/support_development_audit_regression_2026-10-16.txt`. The sole both-direction pass is displayed as a selected demonstration in notebook 2, not a random or representative listening sample.

## Next — Gate E validation preparation

Gate E validation preparation, with Gate C eligibility safeguards retained: declare and package only the seven passing fixed-support shift cases into A/B/C/D development views (split, merge and jitter: 21 planned variants from six phrases / three recordings / three singers). Reuse saved waveforms, acoustics and private masks/targets; retain Gaussian boundary sigma 0.04 seconds and the existing 71-column contract. Keep all 17 measured failures and four unattempted geometry keys in the inventory; failed cases stay outside targets. Keep the existing training-only normalizer, singer partitions and all original no-change development controls. No new rendering, audit retries, support/threshold search or fit. Verify packaging and loader refusals before any separately declared frozen-model evaluation. These would remain exploratory development examples, not research-approved training data. Gate E open, D/J4 review deferred, F open.
