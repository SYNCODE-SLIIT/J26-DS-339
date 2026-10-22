# Estimated-harmonic nonzero correction trial — 15 October 2026

**Current gate: Gate F — usable delivered audio / renderer validation, open.** Musical review (Gate D/J4) remains deferred. These are constructed-source engineering results under supplied correction curves, not natural-vocal repair accuracy, learned correction decisions or production permission.

## Fixed scope and runtime inputs

Completed run: `runs/estimated_harmonic_correction/a2eea865ced64ec19aafe310e9527d61/evidence.json`. Reuse the frozen five [source-estimated harmonic decompositions](Harmonic_Separation_2026-10-15.md). Declare all four original curves per selected profile: **20 requests**, unchanged support 1–2 s. The earlier vibrato-880/glide-220/rested-220 source refusals stay separate; they are not twelve additional measured requests. No new source estimates, waveform coefficient fits, corpus extraction or learner training.

Renderer inputs are source samples, frozen audio-only estimated phase/frequency/support, source-fitted six harmonic coefficients, the preserved residual and supplied frame curve. No known construction F0, phase, center or partial coefficient enters rendering. The original mathematical kernel is reused with **estimated** sample frequencies only. Require strict sign/magnitude-preserving 80-ms ramps, fixed original interior request, zero correction endpoints and integer added-cycle closure using the original numerical equation tolerance. The legacy solver may compute other diagnostics internally; the renderer uses strict keys only and never falls back to relaxed correction.

Shift the estimated harmonic phases and add the exact same float64 residual input, leaving outside samples copied. Actual endpoint samples must match exactly without replacement. Residual input hashes verify unchanged; float32 output quantization can change a residual inferred afterward, so no byte-identity claim is made about that inferred value. Five zero-edit identities separately verify the bypass.

## Eligibility and saved-audio results

**Five crossing-minus30to30 requests render; fifteen requests remain withheld without WAVs.** Nine withheld supports lack an integer in the conditional necessary cycle budget; six contain a necessary integer but the fixed basis has no strict solution. This is a limitation of this model/basis, not a universal impossibility claim. Every request and its numerical plan remain in evidence; no correction bounds or outcomes were tuned.

Each candidate is saved as FLOAT WAV, reloaded exactly and then analyzed by the original frozen PESTO/checkpoint and YIN. Audit version, original supplied target curve, direction rule, confidence/agreement, guards, coverage and error thresholds are unchanged. **All five saved candidates pass the full audit**, fine-error checks and physical count/rate/time/outside/endpoint/finite checks. Five zero identities pass. There is no rested/silence source among the five, so this stage cannot establish silence preservation on that excluded profile.

The matched baseline passes four of these same five cases; constant-880 gains a pass and none is lost. Do not compare this five-case subset directly to the old full 26/32 baseline, count withheld requests as passes, or turn 5/5 into musical accuracy.

| Profile (all crossing curves) | Baseline full pass | Candidate full pass | Fine median / p90 error, cents | Raw transition peak 2048, baseline → candidate | Raw transition peak 1024, baseline → candidate |
| --- | --- | --- | --- | --- | --- |
| constant_110 | True | True | 1.884539 / 3.143957 | 26.893788 → 25.381439 | 29.830698 → 26.035627 |
| constant_220 | True | True | 0.764041 / 1.474570 | 23.444204 → 24.719175 | 27.168143 → 25.834510 |
| constant_440 | True | True | 0.238075 / 0.727518 | 23.188256 → 23.473964 | 26.613752 → 26.038748 |
| constant_880 | False | True | 0.155823 / 0.466826 | 24.117186 → 24.506321 | 26.176739 → 25.910008 |
| vibrato_220 | True | True | 0.467337 / 1.363400 | 24.783477 → 23.773900 | 26.165713 → 25.941542 |

## Transition and amplitude diagnostics

Matched-subset raw transition-change maxima decrease **26.893788→25.381439 cents at 2048** and **29.830698→26.038748 at 1024**. Counts above the descriptive 30-cent marker remain zero for both methods/windows. Individual 2048-window peaks increase for constant-220/440/880; no case increases at 1024. All rows remain visible. These are pitch changes from source, including intended intervention, not target-error or audible-quality scores. No threshold, window or musical interpretation is changed to credit the result.

Amplitude now uses the **saved estimated source center**, not the known construction center, for the fixed fourth-order 0.7–1.3 band. Both baseline/candidate are remeasured with that same center, source-only predicted support and source-relative denominator. Do not treat the result as exactly comparable with earlier known-center or weighted-branch diagnostics. Across ten ramps, counts below ratio 0.5 are baseline/candidate **4/0**, below 0.8 **10/0**, minimum ratios **0.115703501/0.999816681**. These are descriptive signal markers, not auditory acceptance.

## Validation and panel evidence

**170 tests pass** in `runs/estimated_harmonic_regression_2026-10-15.txt`. Eight new tests cover actual source-estimated closed edits and preserved residual/outside/endpoints, exact zero bypass, prior refusal, no relaxed fallback, missing reliable support, changed source/residual, over-limit/outside requests without clipping, and whole-request withholding when any support lacks a strict solution.

The completed run verifies 390 frozen hashes and stores 30 artifacts: twenty numerical plans, five actual WAVs and five audit arrays, plus declaration/code snapshots. No post-outcome rerender or parameter change was made. Both maintained notebooks now retain all twenty statuses, three separate source refusals, both-window individual observations and two labelled source/baseline/candidate audio comparisons. Both maintained notebooks executed from fresh kernels with **zero cell errors**, **21 plots and 40 audio widgets** total: notebook 1 has two plots/two audio; notebook 2 has nineteen plots/38 audio. [Current panel notebook](../runs/notebooks/56831a87e4164502a08db6650605585e/02_data_review_and_paired_experiments.html). Source/executed/export hashes, all 390 frozen hashes and 30 artifacts verify. All twenty numerical plans reproduce and all five saved WAVs reconstruct exactly from frozen inputs; ten fine/coarse metric sets, twenty transition peaks and ten amplitude ramps recompute. Five identities, physical endpoints/outside mappings and residual-input hashes verify; both original review-file hashes are unchanged. [Final verification](../runs/estimated_harmonic_correction/a2eea865ced64ec19aafe310e9527d61/final_verification.json). The new plot was visually checked. Maintained sources retain no stale outputs; earlier executed copies remain historical evidence.

## Next — Gate F (open)

Run the declared source-only transfer screen on ten existing training crops (three Vocadito, seven Jamendo), keeping the estimator/refusal rules fixed and recording every crop before any vocal correction.

The [prospective transfer protocol](../runs/estimated_harmonic_correction/a2eea865ced64ec19aafe310e9527d61/transfer_screen_protocol.json) is **declared, not executed**; its SHA-256 is recorded in final verification before transfer work. It includes all ten already prepared train records, including prior screen exclusions, and preserves their frozen manifest/audio hashes and source grouping. Dataset labels/reference notes are withheld from source estimation/fitting. Use predicted-support coverage and reconstruction/refusal evidence only; Jamendo has no verified human F0 truth here. No corrected vocal WAV is authorized by this screen.

This synthetic candidate is not adopted and no gate automatically closes. Three Vocadito crops and seven Jamendo crops are not independent population/singer evidence. Existing source, license, annotation and review limitations persist. No research-ready example, accepted repaired package, uncertainty calibration, consumer delivery or Windows/macOS integration is credited. Gate D/J4 remains deferred until experienced reviewers are available.
