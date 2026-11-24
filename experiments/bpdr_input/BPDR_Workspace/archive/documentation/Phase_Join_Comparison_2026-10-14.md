# Gate F: phase-aware polar joining comparison — 14 October 2026

**Current gate: F — usable delivered audio, renderer validation; open.** One phase-aware candidate was declared and tested on all fixed controls. It removes the measured fundamental-amplitude dips but worsens transition pitch estimates, so the existing advancement filter **refuses the candidate**. It remains unadopted. Musical review in Gate D/J4 stays deferred; engineering continues under the owner's direction.

## Fixed candidate and scope

[Evidence](../runs/phase_comparison/6bdf0a7f2d3741e6a1a7c1c48dfa39de/evidence.json) and [declaration saved before measurement](../runs/phase_comparison/6bdf0a7f2d3741e6a1a7c1c48dfa39de/declaration.json). All eight original constructed profiles and four curves use the same source WAVs/supports and pinned raw-curve PSOLA branch. No natural vocal or intended-note reference is used. There is one candidate, no parameter search, no excluded case and no retuning after results.

`praat-polar-join-candidate-0.1` changes only the existing 80-ms entry/exit ramps. Compute full-band analytic source/resynthesis signals using [SciPy Hilbert](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html). Within each ramp, make the relative phase continuous with [NumPy unwrap](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html), then subtract an integer cycle so the sample nearest half blend lies in [-pi,pi). Linearly interpolate analytic amplitude and that relative phase using the existing blend weight; take the real part. This is an experimental joining method, not a validated harmonic or natural-vocal phase model.

The original request is validated through the unchanged renderer. Zero edits bypass native/phase processing exactly. Outside samples are copied exactly; the fully resynthesized interior is copied exactly. Near-zero analytic amplitude (<=1e-8) refuses joining rather than selecting a fallback; this is a numerical floor, not calibrated suitability. Nonfinite/clipped results are refused without normalization. Original duration/sample-time geometry remains unchanged. The private native branch reproduces every saved baseline candidate exactly before the new join is applied.

Analytic phase interpolation can itself alter phase rate and apparent pitch; cancellation removal was therefore a hypothesis to test, not a promised quality improvement. Installed versions are NumPy 2.5.3, SciPy 1.18.1, Parselmouth 0.4.7 and Praat 6.1.38. No dependencies, active processing path, renderer policy or audit tolerance changed.

## Results: successful preservation, unsuccessful transitions

| Check | Baseline | Polar join |
|---|---:|---:|
| Full existing guarded audit | 26/32 | 26/32 |
| Fine median/p90 limits | 32/32 | 32/32 |
| Physical/outside/rest/interior preservation | 32/32 | 32/32 |
| Exact zero identities | 8/8 | 8/8 |

All 32 complete audit dictionaries match their baseline values exactly, including the six retained coarse-direction rejects. There are no lost or gained guarded passes. This is expected for a method that copies the resynthesized interior; it does not demonstrate valid boundary delivery. Audit targets remain the original supplied curves, with no coarse epsilon or relaxed threshold.

| Fixed YIN window | Baseline peaks >30 cents /32 | Polar join /32 | Baseline largest peak | Polar largest peak |
|---|---:|---:|---:|---:|
| 2048 samples | 5 | **12** | 42.42 | **80.81** |
| 1024 samples | 11 | **18** | 58.24 | **127.94** |

Both candidate maxima occur in constant-110/crossing -30-to-30. Per-case peaks increase in 18 cases at the primary window and 23 at the shorter window; the full increased/decreased lists are retained. These are relative YIN observations in fixed transition windows, not verified instantaneous waveform pitch or musical ratings. Neither a favorable window nor a guarded pass can authorize delivery.

Using the same known-center narrowband diagnostic and all **72 ramps**:

| Amplitude diagnostic | Baseline | Polar join |
|---|---:|---:|
| Ramps below 0.5 phase-aligned amplitude reference | 25 | **0** |
| Ramps below 0.8 reference | 52 | **0** |
| Minimum measured ratio | 0.02660 | 0.99793 |

All ramps have diagnostic amplitude support. Ratios describe the filtered fundamental relative to weighted branch amplitudes; they are not full-waveform loudness, musical preservation percentages or calibrated auditory thresholds. The candidate improves this diagnostic while failing the transition checks. Thus eliminating amplitude cancellation alone does not justify a repair method. This constructed experiment supports examining phase/pitch constraints within resynthesis; it does not prove a perceptual cause or natural-vocal suitability.

The unchanged advancement filter passes physical preservation, fine checks and baseline-pass retention, but fails transition-summary nonregression and improvement. `eligible_for_bounded_vocal_comparison=False`, `adopted=False`; no research pair or accepted repaired package exists. The candidate is retained for evidence and is not routed to downstream melody processing.

## Verification and notebooks

Six meaningful checks cover opposed-phasor amplitude behavior, continuity across wrapped phase, original request refusal, actual ramp/outside/interior preservation, exact zero processing bypass and near-zero/invalid-input refusal. The full suite passes **129 checks**: [saved log](../runs/phase_comparison_regression_2026-10-14.txt).

Thirty-two candidate FLOAT WAVs round-trip exactly. Their hashes, 32 diagnostic-array hashes, source/code/backend snapshots and frozen declaration are recorded. Both maintained notebooks show the improvement/regression together, with two transparent diagnostic examples: the case with the largest measured transition peak (constant-110/crossing), and the previously diagnosed deep baseline dip (constant-220/-20). Showing these examples does not change the full-case selection. Fresh execution at `runs/notebooks/0b1a7d7ed7bf4c43816e0ce6b5dbe4a5` has zero cell errors, fifteen plots and twenty-eight audio controls. Source/executed/HTML, declaration, prior-input/review/backend, code-snapshot and all array/WAV hashes verify. The new panel plot was visually checked. See [final verification](../runs/phase_comparison/6bdf0a7f2d3741e6a1a7c1c48dfa39de/final_verification.json) and the current evidence tracker; earlier copies remain historical.

## Next action by gate

**Next — Gate F (open): source-anchored resynthesis feasibility.** Examine a local resynthesis approach that constrains boundary phase against the original samples, and declare its phase/pitch feasibility requirements before another waveform trial. Retain the unchanged original-request audit, exact identity/time/outside/rest checks, both YIN windows and amplitude diagnostics. Do not retune the refused polar join after these outcomes. A later bounded training-only vocal comparison requires justified controlled evidence; current supplied fixtures do not authorize accepted delivery.

**Future — Gate D/J4 (deferred):** experienced musical review when reviewers are available. Automatic engineering and the [revised no-reviewer workflow](Reviewer_Deferral_2026-10-14.md) continue now. Research-ready examples remain zero; model/permission calibration, musical preservation, independent effect and consumer validation remain open. The future Windows/macOS pipeline remains planned.

From this component folder, reproduction writes a new fixed evidence directory:

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.phase_comparison
```
