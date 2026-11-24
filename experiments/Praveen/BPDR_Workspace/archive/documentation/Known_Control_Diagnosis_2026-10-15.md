# Frozen known-control diagnosis — 15 October 2026

**Current gate: Gate F — usable delivered audio / renderer validation, open.** Experienced musical review (Gate D/J4) remains deferred by the owner. This stage reanalyzes saved constructed audio only; it does not render or adopt a method.

## Fixed scope and replay

Completed run: `runs/known_control_diagnostics/bf53253259254eef89f9bb3fe09e24bd/evidence.json`, version `known-control-diagnostics-0.1`. All eight [previous known-source controls](Known_Source_Control_2026-10-15.md), their matched baseline and both YIN windows were declared before measurement. Twenty-four unsupported requests remain withheld. All **16 complete guarded audit dictionaries**, **32 original transition peaks** and saved per-frame delta arrays reproduce exactly. Baseline and control each retain seven full passes; gained/lost cases do not change. The run verifies 344 frozen hashes and adds eight diagnostic NPZ files, no candidate WAV.

## Glide rejection: exact values, unchanged decision

The one opposing control frame is 62 at **1.24 s**. Requested and known-control sample-law correction are -15.6 cents. PESTO changes **215.80462646484375 → 215.8046875 Hz**, a difference of 0.00006103515625 Hz. The existing float32-ratio calculation gives **+0.000412758323 cents**; dividing the same frequency values in float64 gives **+0.000489637908 cents**. YIN measures **-17.420746995266 cents**. Source/output PESTO confidence is 1.0.

Both aligned frames come from native frame 62 at 1.24 s with the same respective frequency values. This frame does not expose a nearest-frame timestamp mapping bug. The positive sign survives float64 division, so float32 ratio/log rounding alone does not explain it. The observation is consistent with an almost unchanged coarse tracker estimate despite the known correction, rather than evidence of a substantial opposite shift. It does not establish that the tracker/audit is wrong or that the audio is musically acceptable. **The original `COARSE_DIRECTION` rejection remains.** No epsilon, float64 substitution or altered acceptance policy is adopted.

The baseline's old constant-880 rejection is also retained: frame 61 at 1.22 s, requested -16.8, coarse +0.000619137 cents (float64 +0.000600376), fine -16.224466 cents. Its passing control does not justify forgiving the glide rejection.

## Transition changes are not target errors

The old statistic is maximum absolute YIN change from the source inside fixed transition windows. It includes intended correction. Every old value and per-case increase is preserved, but an increase in this statistic alone cannot show greater target error or audible damage. In particular, raw changes around 25 cents are not automatically 25-cent errors.

For the **known-source control only**, evaluate a separate residual: YIN output/source change minus the exact frozen strict sample-law correction at the nominal frame center. This uses privileged construction truth for diagnosis. It does not change the original audit, whose guarded plateau excludes transitions. Finite-window YIN estimates need not equal instantaneous construction frequency; the residual is not a calibrated musical threshold. The baseline has no verified actual output frequency law, so it is not scored against the control's law. No claim of better baseline-relative target accuracy follows from this new diagnostic.

| Profile | Window | Old raw-change peak, cents | Signed residual at that raw peak, cents | Maximum absolute center-law residual, cents | Residual peak time, s |
| --- | --- | --- | --- | --- | --- |
| constant_110 | 2048 | 25.383600 | -1.383600 | 4.248881 | 1.00 |
| constant_110 | 1024 | 26.036718 | -3.236718 | 7.289706 | 1.86 |
| constant_220 | 2048 | 24.721260 | 0.478740 | 5.422835 | 1.06 |
| constant_220 | 1024 | 25.836475 | -0.636475 | 3.859037 | 1.06 |
| constant_440 | 2048 | 23.472211 | -0.527789 | 4.385127 | 1.94 |
| constant_440 | 1024 | 26.037882 | 0.837882 | 1.889273 | 1.94 |
| constant_880 | 2048 | 24.506185 | -0.506185 | 4.773268 | 1.94 |
| constant_880 | 1024 | 25.908727 | -0.708727 | 1.994007 | 1.94 |
| vibrato_220 | 2048 | 23.773280 | -0.226720 | 4.429931 | 1.02 |
| vibrato_220 | 1024 | 25.938132 | 3.138132 | 3.651922 | 1.08 |
| vibrato_880 | 2048 | 24.287598 | -1.487598 | 4.640774 | 1.06 |
| vibrato_880 | 1024 | 24.728590 | -0.728590 | 2.560385 | 1.06 |
| glide_220 | 2048 | 24.323153 | 0.876847 | 5.193136 | 1.06 |
| glide_220 | 1024 | 27.088008 | -1.888008 | 2.865961 | 1.86 |
| rested_220 | 2048 | 24.721260 | 0.478740 | 5.422835 | 5.06 |
| rested_220 | 1024 | 25.836475 | -0.636475 | 3.859037 | 5.06 |

Across controls, maximum absolute residual is **5.422835 cents at 2048** (rested-220, 5.06 s) and **7.289706 at 1024** (constant-110, 1.86 s). This retains sensitivity to window and source profile. Do not substitute the shorter window to claim a pass.

Glide's raw peaks **24.323153/27.088008 cents** occur at 1.08 s, with center law **-25.2 cents** and signed residual **+0.876847/-1.888008 cents**. Its maximum absolute residuals, **5.193136/2.865961 cents**, occur at 1.06/1.86 s respectively. Raw-peak and residual-peak locations differ; neither summary may silently replace the other.

## Validation and reproducibility

**147 regression tests pass** in `runs/known_control_diagnostics_regression_2026-10-15.txt`. Six new tests cover intended change versus error, distinct peak locations and invalid-frame masking, empty/mismatched grids, exact strict sample-law placement, refusal of overlapping/relaxed-only supports, and actual native-frame tie selection.

Initial diagnostic attempt `9f6b3cd2273549bd9c1c3ba7cacc240f` is retained with its failure/log/source snapshot. It stopped at the first baseline audit comparison because live policy tuples were compared directly to saved JSON lists. The new diagnostic comparison was normalized through JSON, as in the existing original replay. No audio, saved audit values, rule or threshold was changed. The corrected run replays all audit values exactly. No diagnostic outcome was used to tune or regenerate audio.

Both maintained notebooks are updated with the exact rejecting frame, all sixteen case/window rows, distinct change/residual plots, preserved historical outcomes and the declared-next-stage status. Both notebooks executed from fresh kernels with **zero cell errors**, **18 plots and 34 audio widgets** total (notebook 1: two plots/two audio; notebook 2: sixteen plots/32 audio). [Updated panel notebook](../runs/notebooks/221df64a53b34f22bb9b3d75130b0d1c/02_data_review_and_paired_experiments.html). Source/executed/export hashes, all 344 frozen hashes, eight diagnostic artifacts, all sixteen residual summaries and unchanged hashes of both existing review files verify in [final verification](../runs/known_control_diagnostics/bf53253259254eef89f9bb3fe09e24bd/final_verification.json). The new plot was visually checked. Maintained notebook sources retain no stale outputs; historical executed copies remain evidence. The future source-estimation declaration hash is also recorded there, while its status remains declared, not executed.

## Next — Gate F (open)

Run the declared audio-only source-estimation feasibility on the eight existing constructed sources; withhold construction F0/phase from the estimator and use it only in scoring. No new rendering or vocal trial yet.

The [source-estimation protocol](Source_Estimation_Protocol_2026-10-15.md) is **declared, not executed**. It precedes a practical renderer comparison because the successful oracle's construction F0/phase cannot be assumed available for ordinary recordings. Harmonic/residual waveform reconstruction and strict support feasibility also remain unvalidated. Gate D/J4 stays deferred. There are no new approvals, training-ready examples, accepted repaired packages, model fits, new corpus extraction, corpus expansion or portable-pipeline implementation.
