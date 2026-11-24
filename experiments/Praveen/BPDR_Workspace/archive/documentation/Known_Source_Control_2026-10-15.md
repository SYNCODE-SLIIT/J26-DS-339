# Known-source analytical audio controls — 15 October 2026

Current gate: **Gate F — usable delivered audio / renderer validation, open**. Gate D/J4 experienced musical review remains deferred by the owner. Engineering continues; research-ready examples and accepted repaired packages remain zero.

## Question and fixed selection

Can the preceding conditional mathematical solutions be written as actual audio without losing the fixed audio checks? Reuse the frozen [source-anchored feasibility](Anchor_Feasibility_2026-10-15.md), original constructed waveforms, supplied curves and unchanged audits. Before measurement, select every case whose every support has the strict fixed-basis solution: eight crossing-minus30to30 cases across the eight existing profiles (nine supports, because the rested profile has two). Explicitly withhold the other 24 cases. Do not synthesize the relaxed solutions, widen correction bounds, tune after outcomes, or treat unresolved cases as passed.

This is a **known-source oracle control**, version `known-source-anchor-control-0.1`. Synthesis requires the exact construction F0, phase and fixed harmonic amplitudes. It verifies the supplied source matches that construction sample for sample, then adds the saved strict phase curve only inside the permitted supports. There is no ordinary vocal waveform estimator or runtime renderer here. Auditors receive the saved audio, not construction F0/phase.

## Execution and preservation

Completed run: `runs/known_source_control/1d7a2f6206d94eb5993cfe2cbd0b56f2/evidence.json`. Declaration, selected/withheld lists, source-code snapshots and input hashes are saved before measuring. Eight FLOAT WAVs are saved and reloaded exactly; eight NPZ files retain pitch-transition and amplitude diagnostics. The saved audio is analyzed with the original PESTO checkpoint and independent YIN. The original framewise supplied curve, 80-ms ramps, 100-ms guard, fine/coarse thresholds, coverage and strict per-frame direction rule are unchanged.

All eight controls preserve sample count, rate/time mapping, samples outside intervention supports, rested silence, exact first/last support samples and finite unclipped output. Eight empty-correction identity checks reproduce their sources exactly. No endpoint replacement is forced. All eight fine-error checks pass. Full guarded audits pass seven cases. These are dependent constructed conditions, not independent songs/singers or a musical repair accuracy percentage.

The matched baseline also passes seven of these same eight cases; comparing this subset to the earlier 26/32 baseline would be misleading. **Constant-880 gains a full pass, while glide-220 loses a baseline pass with one opposing coarse-direction frame (`COARSE_DIRECTION`).** Keep the rejection; no epsilon, changed direction rule or relaxed acceptance is adopted.

| Profile (all crossing curves) | Baseline full pass | Control full pass | Transition peak 2048, baseline → control (cents) | Transition peak 1024, baseline → control (cents) |
| --- | --- | --- | --- | --- |
| constant_110 | True | True | 26.8938 → 25.3836 | 29.8307 → 26.0367 |
| constant_220 | True | True | 23.4442 → 24.7213 | 27.1681 → 25.8365 |
| constant_440 | True | True | 23.1883 → 23.4722 | 26.6138 → 26.0379 |
| constant_880 | False | True | 24.1172 → 24.5062 | 26.1767 → 25.9087 |
| vibrato_220 | True | True | 24.7835 → 23.7733 | 26.1657 → 25.9381 |
| vibrato_880 | True | True | 28.5991 → 24.2876 | 30.1962 → 24.7286 |
| glide_220 | True | False | 22.5696 → 24.3232 | 24.7424 → 27.0880 |
| rested_220 | True | True | 26.1011 → 24.7213 | 27.8588 → 25.8365 |

## Transition and amplitude diagnostics

The two fixed YIN-window maxima improve on this matched eight-case subset: 28.5991 → 25.3836 cents (2048) and 30.1962 → 27.0880 (1024). Counts exceeding the descriptive 30-cent marker are 0 → 0 and 1 → 0. Individual regressions remain: constant-220, constant-440, constant-880 and glide-220 worsen at 2048; glide-220 also worsens at 1024. Lower aggregate maxima do not erase those regressions or demonstrate better singing.

Amplitude uses a **new source-relative denominator**, output narrowband fundamental amplitude divided by the original source fundamental amplitude, rather than the older weighted-branch reference. Both methods use the same source-supported samples: known-center filter, original 80-ms ramps, weights strictly between 0.1 and 0.9, source amplitude above 10% of its median voiced amplitude. The source voicing mask excludes the constructed silence. Low output amplitude never removes a sample from evaluation. Do not compare these counts directly with prior differently defined ratios.

Across 18 ramps, baseline/control counts below 0.5 are 9/0, below 0.8 are 17/0, and minimum ratios are 0.115704/0.999621. These signal diagnostics do not establish audibility, timbre preservation or expert approval. Known harmonic amplitudes are part of this oracle's construction.

## Validation and panel evidence

Final regression: **141 tests passed**, recorded in `runs/known_source_control_regression_2026-10-15.txt`. Seven new tests cover whole-case eligibility, exact physical preservation and identity, refusal of different source waveforms, no relaxed fallback, phase/direction rejection, and retaining zero-amplitude output samples in the diagnostic mask.

The first attempt, `runs/known_source_control/0a28148a807a4ba4b844df6c614f5e1f`, failed before producing any control because of a missing `independent_pitch` import. It is retained. The import and explicit source-voicing amplitude floor were corrected before measured outcomes, and the unchanged selection/audits were rerun once. No failed or regressed measured case was removed or tuned.

The same two maintained panel notebooks are updated: notebook 1 states the current result and limits; notebook 2 verifies frozen hashes and retains all 32 status rows, both-window per-case peaks, the glide rejection, one low-pitch and one rejected audio comparison. Both executed from fresh kernels with **zero cell errors**, **17 plots and 34 audio widgets** total: notebook 1 has two plots/two audio widgets; notebook 2 has fifteen plots/32 audio widgets. [Current panel notebook](../runs/notebooks/d4de0f3e23b5464a82cf5b6d0a968d23/02_data_review_and_paired_experiments.html). Source/executed/HTML hashes, 326 tracked hashes, 16 artifacts, eight exact WAV reconstructions, identity checks and unchanged review-file hashes verify in [final verification](../runs/known_source_control/1d7a2f6206d94eb5993cfe2cbd0b56f2/final_verification.json). The new six-panel plot was visually checked for readable labels and complete output. Maintained sources retain no stale outputs. Historical executed notebooks remain evidence.

## Next — Gate F (open)

Diagnose the remaining glide coarse-direction rejection and per-case transition regressions on the frozen WAVs; use that evidence to predeclare a practical source-estimated renderer comparison without oracle F0/phase.

Gate F is not closed by an oracle that has privileged source information. No practical renderer is adopted and no output is delivered as accepted repair. Continue to preserve the original audit, failures and 24 withheld rows. Gate D/J4 experienced review remains deferred until reviewers are available. Gate B voice support, Gate E matched research comparisons and downstream consumer/integration gates remain open; the Windows/macOS portable pipeline remains planned. No new extraction, pair generation, training or review alteration was performed.
