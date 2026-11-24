# Audio-only source-estimation feasibility — 15 October 2026

**Current gate: Gate F — usable delivered audio / renderer validation, open.** Experienced musical review (Gate D/J4) remains deferred. This experiment estimates source features only; no corrected audio or ordinary-vocal renderer is produced.

## Declared experiment and separation from truth

Completed run: `runs/source_estimation/8c34deae8796487e9e261cd41bff9171/evidence.json`. The preceding [protocol](Source_Estimation_Protocol_2026-10-15.md) was executed without modifying its machine-readable declaration: SHA-256 `63e2c38648b3d3dc0e217332e6a3d624979d55de5dc1b9e25887c262d3f110de`. All eight original source profiles are included, with no outcome-based removal. These sources are distinct from the eight earlier correction cases and 24 withheld requests, whose decisions remain unchanged.

The two-argument estimator receives decoded float32 samples and sample rate only. No source path/profile, known center, construction F0/phase or prior correction enters the estimator. It uses the existing frozen PESTO/checkpoint and YIN-2048 reliability, estimated median YIN filter center, fixed bandpass/Hilbert method and unwrapped-phase derivative. No global phase alignment is fitted to truth. Source-only masks use nearest frames with later ties, 100-ms guards at clip ends/predicted transitions and the declared amplitude floor.

All eight predictions, status/reasons and prediction arrays were saved and hashed in `predictions_sealed.json` **before** the separate scorer loaded construction metadata. The seal and every prediction hash remain unchanged through scoring. Frozen analyzer provenance matches the original fixtures. The run verifies 356 input hashes and saves eight prediction arrays, one prediction seal and eight scoring arrays; no WAV.

Implementation notes were fixed before measuring: the stated nonpositive-frequency refusal is applied to the **entire** derived frequency array, including excluded edges/silence. Supports and score coverage are reported separately. Descriptive scores may be retained for a refused source wherever jointly supported values are defined, but its status remains refused. This does not introduce a post-outcome support-only relaxation.

## Results: five estimated, three refused

| Source | Status | Estimated center, Hz | Joint scored / true voiced samples | Frequency absolute error median / p90 / max, cents | Circular phase error p90, radians | Nonpositive samples total / on retained support |
| --- | --- | --- | --- | --- | --- | --- |
| constant_110 | estimated | 110.151696 | 83349/88200 | 0.302764 / 1.290647 / 5.394208 | 0.000746165 | 0 / 0 |
| constant_220 | estimated | 220.123873 | 83349/88200 | 0.229938 / 0.671768 / 2.670428 | 0.000388564 | 0 / 0 |
| constant_440 | estimated | 440.223298 | 83790/88200 | 0.219046 / 0.467466 / 1.653205 | 0.000270583 | 0 / 0 |
| constant_880 | estimated | 880.853311 | 83349/88200 | 0.208048 / 0.349270 / 0.834539 | 0.000203816 | 0 / 0 |
| vibrato_220 | estimated | 220.167073 | 83349/88200 | 0.232874 / 0.677991 / 2.745659 | 0.000391514 | 0 / 0 |
| vibrato_880 | refused | 881.135152 | 83569/88200 | 0.201811 / 0.414981 / 1.206325 | 0.000241308 | 3 / 0 |
| glide_220 | refused | 220.203123 | 83349/88200 | 0.244942 / 0.898506 / 3.209815 | 0.000519418 | 25 / 0 |
| rested_220 | refused | 220.123873 | 123480/132300 | 0.213096 / 0.576572 / 2.674890 | 0.000333175 | 21129 / 0 |

Five profiles have status `estimated`, not approved or repaired. **Vibrato-880, glide-220 and rested-220 remain refused**, all for `NONPOSITIVE_ESTIMATED_FREQUENCY`. Vibrato-880 has three such samples in the last 0.14 ms; glide-220 has 25 in the final approximately 1.14 ms; rested-220 has 21,129 between 2.575692 and 4.414422 s, inside constructed silence. None is on retained support. Those observations diagnose where the fixed rule stops; they do not authorize accepting the refused profiles or changing the rule silently.

On joint support across all eight rows, including descriptive refused-row scores, per-profile median frequency errors range 0.201811–0.302764 cents, p90 0.349270–1.290647, and maximum 0.834539–5.394208. Phase p90 values range 0.000204–0.000746 radians with no oracle offset fit. Amplitude-ratio medians range 0.999996–0.999999; each profile's p10/p90 is retained in evidence/notebook. These are measurements on simple known harmonic sources, not independent singer evidence, intended-note correctness or repair accuracy.

## Coverage, exclusion and silence

The scorer uses **707,584 of 749,700 true-voiced samples** across all eight sources; 42,116 true-voiced samples are excluded. These are correlated samples, so the ratio is coverage only, not accuracy or population suitability. All support masks and excluded counts remain visible.

For rested-220, source-only tracker reliability has **441 samples of false support in silence** and excludes 441 true-voiced samples before guards. Guarding removes false support; retained false support is zero, with **123,480 / 132,300** true-voiced samples retained and 8,820 excluded. This distinguishes the initial reliability error from the guarded result. All 21,129 nonpositive phase-derived frequencies occur outside retained support, and the source remains refused under the fixed global rule.

## Validation and panel evidence

**154 regression tests pass** in `runs/source_estimation_regression_2026-10-15.txt`. Seven new tests check an unlabelled 197-Hz signal with an arbitrary phase, refusal without reliable trackers, retaining global refusal for bad edge frequency, source-only transition guards and tie behavior, counting false silence support, measuring a phase offset rather than fitting it away, and refusing to silently drop invalid joint frequency values from a score. No implementation fix or parameter tuning followed measured outcomes.

Both maintained notebooks are updated with the sealed input boundary, all eight statuses/metrics, three refusals, silence/exclusion counts and a clear distinction between estimation and repair. Both notebooks executed from fresh kernels with **zero cell errors**, **19 plots and 34 audio widgets** total: notebook 1 has two plots/two audio; notebook 2 has seventeen plots/32 audio. [Current panel notebook](../runs/notebooks/ff62b11851f64836af4911018edf8cfa/02_data_review_and_paired_experiments.html). Source/executed/export hashes, all 356 frozen hashes and 17 artifacts verify. All eight sealed predictions remain unchanged after scoring, and all 24 metric distributions recompute exactly. Both existing review-file hashes are unchanged. [Final verification](../runs/source_estimation/8c34deae8796487e9e261cd41bff9171/final_verification.json). The new four-panel plot was visually checked. Maintained sources retain no stale outputs; historical executed copies remain evidence.

## Next — Gate F (open)

Predeclare and test source-estimated harmonic/residual separation on the five non-refused profiles, withholding the three refused profiles and preserving their results. Verify reconstruction and support before any pitch correction.

This next stage is planned, not implemented or measured here. It must declare its method and source-only coefficients/phase use before measuring, preserve the three current refusals and verify modelled harmonic fidelity plus residual preservation. Exact zero-edit recombination alone cannot establish nonzero correction quality. Continue to keep uncertainty, strict support/phase feasibility, natural-vocal transfer and audible quality unvalidated. No Gate D/J4 review, research-ready anchor, model fit, accepted repair or dual-OS integration is credited.
