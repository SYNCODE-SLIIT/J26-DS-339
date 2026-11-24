# Source-estimated harmonic/residual reconstruction — 15 October 2026

**Current gate: Gate F — usable delivered audio / renderer validation, open.** Gate D/J4 experienced musical review remains deferred. This is source decomposition and zero-edit reconstruction evidence, not nonzero pitch correction or a validated natural-vocal renderer.

## Fixed method and eligibility

Completed run: `runs/harmonic_separation/5e708a6b633d49dd9de7a0958783c617/evidence.json`. Reuse the frozen [audio-only estimates](Source_Estimation_2026-10-15.md), without rerunning estimation. Declare the five non-refused profiles before fitting; retain vibrato-880, glide-220 and rested-220 as `WITHHELD_PRIOR_REFUSAL` with no new fit/component arrays. Previous correction decisions are unchanged.

The source-only fitter receives decoded audio, sample rate and the prior estimated phase/frequency/support, without profile, construction F0/phase or true partial coefficients. The fixed basis is sine/cosine at harmonics 1–3 of estimated phase. Fit six coefficients to retained source samples before clip midpoint using `numpy.linalg.lstsq(..., rcond=None)`, no regularization/fallback. Require rank six and retained evaluation samples at/after midpoint. All five fits have rank six, with condition numbers 1.000189–1.001704.

**The full clip was already used for phase/filter estimation.** The second half is excluded only from coefficient fitting; it is not an independent source, forecast, singer split or research validation. Three harmonics match the existing synthetic signal family. The method and results do not establish that order/model is suitable for actual voices.

Set the harmonic model to zero outside unchanged retained support. The residual is float64 source minus that supported model; it is not discarded, denoised, normalized or shifted. Measure raw recombination and its float32 cast separately from the exact zero-edit bypass. Save/hash all five coefficient records and decompositions before loading construction partials into the scorer. No oracle alignment is used in fitting; source-derived sine/cosine coefficients can describe phase offsets from waveform evidence.

## Results

| Profile | Status | Fit / evaluation samples | Total harmonic evaluation relative RMSE | True partial evaluation relative RMSE H1 / H2 / H3 | Float32 recombination differences |
| --- | --- | --- | --- | --- | --- |
| constant_110 | decomposed | 41674 / 41675 | 0.000577455 | 0.000387424 / 0.000876174 / 0.001771793 | 0 |
| constant_220 | decomposed | 41674 / 41675 | 0.000257216 | 0.000199781 / 0.000391780 / 0.000947559 | 0 |
| constant_440 | decomposed | 41895 / 41895 | 0.000139412 | 0.000146813 / 0.000276332 / 0.000702568 | 0 |
| constant_880 | decomposed | 41674 / 41675 | 0.000068835 | 0.000119963 / 0.000246093 / 0.000564997 | 0 |
| vibrato_220 | decomposed | 41674 / 41675 | 0.000260110 | 0.000203085 / 0.000398317 / 0.000963797 | 0 |
| vibrato_880 | withheld: prior refusal | — | — | — | — |
| glide_220 | withheld: prior refusal | — | — | — | — |
| rested_220 | withheld: prior refusal | — | — | — | — |

Total harmonic-model evaluation relative RMSE is **0.000068835–0.000577455**, defined as waveform RMSE divided by source RMS on retained second-half samples. The largest individual-partial evaluation ratio is **0.001771793**, normalized by that true partial's RMS. Each fit/evaluation/full-retained partial score is retained, rather than deriving component quality from the recombination identity. Truth is used only by the scorer after source-only decompositions are sealed. These are descriptive signal ratios, not repair-accuracy percentages or musical acceptance thresholds.

All five raw float64 sums and float32 casts reproduce their input source exactly in this run: **zero differing samples and maximum absolute error zero**. All five no-edit bypasses are separately exact, sample counts unchanged and excluded/outside-support samples preserved. Retained masks/exclusions remain the frozen source-only ones. There is no rested/silence case in the five eligible profiles, so do not claim a new silence-preservation result here.

The residual is defined from the source, so raw recombination is an **accounting check** and can hide harmonic modelling errors by retaining them in residual. It does not prove nonzero correction quality, musical acceptability or a robust component separation for real voices. The separate construction-partial scores supply the limited fidelity evidence on this simple family. Arithmetic differences are measured without forced replacement if they occur; a unit test exercises that case.

## Validation and panel evidence

**162 regression tests pass** in `runs/harmonic_separation_regression_2026-10-15.txt`. Eight new tests cover unlabelled source-derived partial coefficients, withholding prior refusals before fit, unchanged inputs/outside impulses, withholding second-half amplitude changes from coefficient fitting, rank-deficiency refusal, changed grid/nonpositive frequency refusal, honest raw arithmetic differences versus exact bypass, and reference-energy scoring without an epsilon denominator.

The completed run verifies 376 frozen hashes and saves eleven artifacts (five decomposition arrays, one seal, five construction-score arrays), plus declaration/source snapshots. All decompositions remain unchanged after truth scoring. No candidate WAV, pitch shift, new source estimation, learning-model training, dataset pair, review alteration or adoption is performed. Signal coefficient fitting here is explicitly distinct from research learner fitting.

Both maintained notebooks are updated with all eight source statuses, exact reconstruction checks, all fifteen harmonic-fidelity rows, the evaluation limitation and the preserved residual. Both notebooks executed from fresh kernels with **zero cell errors**, **20 plots and 34 audio widgets** total: notebook 1 has two plots/two audio; notebook 2 has eighteen plots/32 audio. [Current panel notebook](../runs/notebooks/6a70d75ded744a3faf0066ba6066b93b/02_data_review_and_paired_experiments.html). Source/executed/export hashes, all 376 frozen hashes and eleven artifacts verify; five sealed decompositions remain unchanged, and all 55 reconstruction/fidelity scores recompute exactly. Frozen support, raw reconstruction, bypass and excluded samples verify directly against the original sources. Both existing review-file hashes remain unchanged. [Final verification](../runs/harmonic_separation/5e708a6b633d49dd9de7a0958783c617/final_verification.json). The new four-panel plot was visually checked. Maintained sources retain no stale outputs; historical copies remain evidence.

## Next — Gate F (open)

Predeclare a nonzero correction trial on all four fixed curves for the same five profiles (20 requests); solve strict phase closure from estimated frequencies, keep the residual unchanged and withhold infeasible cases before unchanged saved-audio audits.

This nonzero trial is planned, not implemented or measured here. It must use estimated phase/frequency and source-fitted harmonics rather than construction truth, retain all outcome rows and the original audit rules/both YIN windows, and keep prediction permission/uncertainty unvalidated. No accepted repair or research-ready anchor is credited. The other research and downstream-consumer gates remain open; Windows/macOS portable integration remains planned.
