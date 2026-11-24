# Gate E reference-cue ceiling (R0) - results, 17 October 2026

> **Correction, 17 October (later):** this conclusion is withdrawn. The probe tested an absolute tuning-grid cue on pitch that cannot resolve the shifts (PESTO mir-1k_g7 activations are one-hot at 33.3-cent bins). The proposal's signal is the within-note step, which correlates 0.57 with the shift on Jamendo when measured with fine pitch. See the correction entry in `PP1_Phase_Execution_Plan.md`.


**R0 failed its declared gate, so R1 (input adapter and matched comparison) is not unlocked.** Protocol: [declaration](Gate_E_Reference_Cue_Declaration_2026-10-17.md), SHA-256 `3b4396a7…f0fcb`, authenticated by the runner before execution. Evidence: [evidence.json](../runs/reference_cue_ceiling/evidence.json), [mixture_tuning.json](../runs/reference_cue_ceiling/mixture_tuning.json) (hashed before any probe, `bba737ea…bce3`), [log](../runs/reference_cue_ceiling/execution_2026-10-17.log). Code: `src/bpdr_input/reference_cue_ceiling.py`.

Executed as declared: 70 full-track mixture tuning scalars (full track and crop window), YIN fine pitch on 422 existing waveforms, artist-grouped probes for ref0-ref3, 2,000 artist-bootstrap resamples. No neural fit, no new separation/GAME/crops/shifted audio, no development or holdout data.

## Jamendo (gated)

| Reference | Original-plateau scatter SD (cents) | Correction gain [95% CI] | Objective gain [95% CI] | Gain over ref0, lower bound (corr / obj) | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| ref0 own offset, quantized PESTO | 14.8 | 10.4% [5.8, 14.2] | -5.0% [-13.7, 2.5] | - | - |
| ref1 own offset, fine YIN | 11.0 | **18.7%** [11.3, 25.8] | **+3.4%** [-7.3, 12.6] | +4.2 / +3.9 points | no |
| ref2 full-track mixture tuning | 15.5 | 11.1% [4.1, 18.5] | -5.7% [-14.8, 2.6] | -4.8 / -6.4 | no |
| ref3 crop-window mixture tuning | 14.9 | 11.7% [5.4, 17.9] | -3.5% [-13.3, 4.8] | -3.4 / -4.6 | no |

Thresholds: correction gain >= 25%, objective gain >= 10%, bootstrap lower bound of the improvement over ref0 above zero.

## Vocadito (descriptive, not gated)

| Reference | Scatter SD | Correction gain [95% CI] | Objective gain [95% CI] |
| --- | ---: | ---: | ---: |
| ref0 | 21.4 | 13.0% [7.9, 18.2] | -26.8% [-40.9, -10.7] |
| ref1 | 19.1 | -0.1% [-6.4, 4.5] | -38.1% [-53.3, -25.8] |

## What R0 establishes

1. **Pitch quantization is a real, measurable limit.** Fine YIN pitch is the only change that clearly helps: it lowers off-grid scatter from 14.8 to 11.0 cents and roughly doubles the correction gain (10% to 19%). Its improvement over ref0 has a bootstrap interval above zero. It is still below the declared thresholds, and its net objective gain interval includes zero.
2. **The unshifted mixture reference does not help.** Full-track mixture tuning gives higher scatter (15.5 cents) than the singer's own offset (11.0). Mixture tunings vary by 13.7 cents SD across tracks, while the crop window agrees with the full track to a median of 3 cents. In these songs, singers' own intonation is a better reference for their notes than the instruments' tuning.
3. **The remaining limit is natural note-to-note intonation.** About 11 cents of scatter remains on Jamendo even with fine pitch, and 19 on Vocadito. That is comparable to the 10-20 cent shifts. Gains concentrate in +20/+30 cents (24-33%); -10 cents is not recoverable for any reference.
4. **Vocadito cannot be served by this approach.** Probes add 3-4 cents of error on unshifted audio for at most 13% correction gain. The fine-pitch result is no better than zero.

## Consequence (declared R2)

Per the declaration, R0 failure means stop. Blind or reference-assisted detune repair toward the original performance is information-limited on this data. No further fit of this family is justified, including longer budgets, loss reweighting or more variants. The one input change with measured value is fine (non-quantized) pitch. It is worth noting for any future redesign, but it does not reach the gate on its own.

R2 recording is complete: the README, execution plan, evidence tracker, master plan and reviewer-deferral note carry this step, and the evidence index and notebook 2 section 14.39 (both notebooks fresh-executed with zero errors, 51 plots, 53 audio controls; run `runs/notebooks/4fb8b70daaad412e8f19982f0169d83a`) show it. `tests/test_runtime_cue.py` adds 11 regression tests; the full suite passes 399 tests (`runs/runtime_cue_regression_2026-10-17.txt`). Recording script: `runs/record_gate_e_diagnosis_2026-10-17.py`. E/F open, D/J4 deferred, research-ready zero, no model adopted.
