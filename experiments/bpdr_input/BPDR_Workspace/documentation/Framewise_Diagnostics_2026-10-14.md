# Gate F: framewise direction and transition diagnosis — 14 October 2026

**Current gate: F — usable delivered audio, renderer validation; open.** The diagnostic step is complete; the renderer is unadopted. Gate D/J4 independent original-reference approvals remain pending, with zero research-ready examples. No real vocal was repaired, research pair generated, model fitted or acceptance rule changed.

## Frozen replay

The [new diagnostic record](../runs/framewise_diagnostics/4ece4923dc68495ba7efd2580733d89e/evidence.json) uses all 32 cases from the existing constructed experiment. It verifies original code, backend, analyzer/checkpoint and all 42 original waveform artifact hashes. All original audit values and decisions replay exactly after JSON representation normalization: **26 pass, six reject**. Reconstructing the private full PSOLA output and applying the original envelope reproduces all 32 saved local candidates sample for sample. Original evidence and source WAVs remain unchanged.

Before measurement, the diagnostic declares all-case selection, descriptive magnitude bins (0.001/1/10 cents), four contrasts and two YIN windows. Per-frame arrays and exact opposing-frequency values are saved. The first diagnostic attempt stopped on a tuple-versus-JSON-list comparison of the same policy; its failed record is retained and is not credited. The completed record uses JSON-normalized exact comparison, without numerical tolerance for audit replay.

## Coarse-direction rejects

All **17 opposing frames** across the six rejects have PESTO changes smaller than **0.001 cent**. The maximum is **0.00082552 cent**, versus requested magnitudes of at least 10 cents. Recomputing frequency ratios in float64 retains the tiny opposing signs (maximum 0.00073446 cent); this is not solely rounding in the float32 ratio calculation. The stored PESTO frequencies themselves differ by only about 0.000046–0.000366 Hz. YIN changes have the requested direction at these frames.

| Profile / curve | Opposing frames | Largest opposing magnitude, cents |
|---|---:|---:|
| Constant 220 / -20 | 3 | 0.000413 |
| Constant 880 / -20 | 1 | 0.000619 |
| Constant 880 / crossing | 1 | 0.000619 |
| Vibrato 880 / -20 | 2 | 0.000826 |
| Vibrato 880 / rising | 1 | 0.000722 |
| Rested 220 / -20 | 9 | 0.000413 |

These observations support a numerical-scale coarse-output discrepancy at the failure trigger, rather than a measured substantial opposing shift. They do **not** establish accurate coarse tracking, prove intended pitch or approve the audio. The existing rule rejects any opposing coarse sign; therefore all six rejects remain rejects. The 0.001-cent bin is descriptive, not a proposed epsilon or amended audit threshold. Any future consistency-rule revision needs a separately declared validation across independent vocals, no-change and known opposing-shift controls, with its own version.

## Transition contrasts

For the same original ramp/guard windows, compare the saved local candidate, private whole-output shifted PSOLA and analytic shifted-contour oracle. Also measure private unchanged-tier resynthesis once for each of the eight profiles. Whole-output and oracle waveforms are diagnostic contrasts, not permitted local deliveries: outside samples/phase/rests are not guaranteed preserved. Actual zero corrections still bypass the backend exactly.

| YIN window | Local candidates with peak >30 cents /32 | Full shifted PSOLA /32 | Analytic oracle /32 |
|---|---:|---:|---:|
| 2048 samples, existing auditor | 5 | 4 | 1 |
| 1024 samples, sensitivity diagnostic | 11 | 13 | 8 |

The 1024-sample result does not replace the existing auditor. The shorter window covers fewer pitch cycles, especially for 110 Hz. Comparing these windows demonstrates estimator sensitivity; the counts are not musical pass rates. A crossfade has no known linear instantaneous-pitch target, and oracle YIN peaks can also exceed 30 cents despite a bounded analytic contour.

For the five original local peaks above 30 cents:

| Profile / rising or +20 curve | Local peak | Full peak | Oracle peak |
|---|---:|---:|---:|
| Constant 110 / +20 | 34.43 | 22.39 | 22.82 |
| Constant 110 / rising | 42.42 | 30.36 | 30.75 |
| Vibrato 220 / rising | 36.87 | 29.08 | 29.42 |
| Glide 220 / rising | 31.05 | 28.71 | 28.90 |
| Rested 220 / rising | 32.54 | 30.10 | 29.37 |

Values are maximum absolute **relative YIN changes**, in cents, within each case's transition windows; maxima can occur at different timestamps. All five local peaks exceed the corresponding full-output peaks. The exactly reproduced local/full contrast supports a blending contribution to these measured peaks, but does not isolate a perceptual cause or prove waveform instantaneous-pitch overshoot. Whole-output PSOLA itself has peaks and the oracle shows estimator bias. Private unchanged-tier PSOLA yields up to 3.46 cents at the existing window (up to 6.24 at the shorter window) and changes waveform samples. Keep exact zero-edit bypass. Human listening, transition/timbre/expressivity preservation and actual vocal suitability remain unmeasured.

## Verification and next gate action

The diagnostic helper adds four regression checks, including a tiny opposing change that **must remain rejected**, a substantial opposing control, transition-window boundaries and invalid-frequency exclusion. The full suite passes **112 checks**: [saved log](../runs/framewise_diagnostics_regression_2026-10-14.txt). Both maintained panel notebooks now expose the current gate, the six retained failures, numerical-scale direction measurements and window-dependent transition contrasts. Fresh execution at `runs/notebooks/7f657b7697674a1797014ba818f9be81` has zero cell errors, twelve plots and sixteen audio controls. Source, executed-copy, HTML and original-input hashes verify; the new diagnostic plot was visually checked. See [final verification](../runs/framewise_diagnostics/4ece4923dc68495ba7efd2580733d89e/final_verification.json) and the current evidence tracker.

**Next — Gate F, controlled transition handling:** predeclare one boundary/phase handling candidate and compare it against the unchanged current renderer on the same eight profiles and four curves. Keep the existing full audit, exact zero identity, sample-count/time-origin/outside/rest checks and both fixed transition diagnostics. Report every regression; do not choose a favorable window or erase the six rejects. Passing guarded interiors cannot close transition preservation. After a candidate has justified controlled evidence, run a bounded supplied-curve comparison on existing training-only Vocadito/Jamendo crops and collect matched musical listening observations. Keep validation/evaluation sources untouched and preserve original references.

**Parallel — Gate D/J4:** obtain the two actual independent original-reference reviews using the existing packet. These decisions and verified supports are required before research pairs/fitting. Gate B independently identified voice coverage, Gate E matched model research evidence and Gate G/H1 accepted consumer validation also remain open. The future Windows/macOS exporter remains planned.

Reproduce only when a new diagnostic run is needed, from this component folder:

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.framewise_diagnostics
```

This writes a new run and never rewrites the original measurements or adopts a candidate.
