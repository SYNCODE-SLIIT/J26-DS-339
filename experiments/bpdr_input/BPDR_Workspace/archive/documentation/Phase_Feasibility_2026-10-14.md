# Gate F: phase and amplitude feasibility diagnosis — 14 October 2026

**Current gate: F — usable delivered audio, renderer validation; open.** The phase/amplitude diagnostic is complete. Musical review is now [deferred by the owner](Reviewer_Deferral_2026-10-14.md); engineering continues without fabricated approvals. No renderer is adopted, new candidate WAV issued, research pair generated or model fitted.

## Declared scope and numerical method

[Evidence](../runs/phase_feasibility/d215daf94aa749ce8798f72ab13976d9/evidence.json), [declaration saved before measuring](../runs/phase_feasibility/d215daf94aa749ce8798f72ab13976d9/declaration.json). Use all 32 previous cases for each existing method: baseline local blending and the refused boundary-taper candidate. The rested profile has two supports, so there are **72 ramps per method, 144 total**, across eight constructed sources. These are dependent variants, not independent singers or musical trials.

The private full PSOLA branch is reconstructed for each method. Applying its original waveform blend reproduces all **64 saved local candidates exactly**. Original code, policies, sources, prior results, arrays and backend hashes are checked and preserved. No whole-resynthesized audio is exposed as a deliverable.

On these known harmonic signals only, isolate the fundamental using a Butterworth bandpass with design parameter N=4 and cutoffs 0.7/1.3 times the declared source-center Hz. A bandpass design has twice the prototype order; forward/backward application filters twice. Use default odd padding and SOS filtering, then a Hilbert analytic signal. SciPy 1.18.1 is the installed runtime. The APIs are documented in [SciPy Butterworth design](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html), [forward/backward SOS filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html) and [Hilbert analytic signal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html). No dependencies or active renderer settings were changed.

Let source/full analytic branches be z0/z1 and the existing blend weight be w. Relative phase is the angle of z1 times the complex conjugate of z0. The algebraic blend-amplitude ratio is:

`abs((1-w)*z0+w*z1) / ((1-w)*abs(z0)+w*abs(z1))`.

The denominator is a phase-aligned amplitude reference, not an accepted waveform. Separately measure the filtered saved local waveform's amplitude over that same denominator. Filtering and time-varying mixing do not commute, so the algebraic and measured ratios need not be identical. Ratios can exceed one; they are not a gain-normalization command.

Measure only ramp samples with w strictly between 0.1 and 0.9, and both branch amplitudes above 10% of the source's median voiced fundamental amplitude. These masks and the 0.5/0.8 ratio markers are fixed descriptive settings, not calibrated auditory or acceptance thresholds. Known band centers are construction metadata, never inferred intended notes. Filtering, finite windows and narrowband phase are diagnostic limitations; this method is not validated on natural vocals or multivoice audio.

## Results

| Diagnostic observation | Baseline | Boundary taper |
|---|---:|---:|
| Cases | 32 | 32 |
| Supported ramps / tested ramps | 72/72 | 72/72 |
| Ramps with some phase opposition, cos(gap)<0 | 56 | 60 |
| Ramps whose measured ratio falls below 0.5 | 25 | 45 |
| Ramps whose measured ratio falls below 0.8 | 52 | 59 |
| Smallest measured ratio | 0.02660 | 0.05877 |
| Smallest algebraic ratio | 0.02605 | 0.05913 |

The baseline's deepest measured dip occurs at constant-220/-20 entry, about 1.0403 seconds; its branch phase gap is about -3.09 radians, near opposite phase. The taper's deepest dip occurs at constant-110/-20 exit, about 1.9599 seconds, with gap about 3.02 radians. These are **filtered fundamental amplitudes relative to the phase-aligned branch reference**, not full-waveform silence or a percentage of preserved musical quality.

The fixed constant-110/rising illustration, already used in prior reports, shows an entry minimum ratio of about 0.506 baseline and 0.425 taper. Its exit ratios are about 0.874 and 0.799. Thus tapering cents did not generally prevent destructive fundamental mixing. The numerical branch/mixture measurements support examining phase-aware joining. They do not prove that cancellation alone causes the prior YIN peaks: those peaks are window dependent, harmonics remain outside this band, and phase/rate changes also matter. No statistical causal claim or human auditory verdict is made.

Eleven controls pass: equal-phase, opposite-phase and quarter-cycle complex-phasor controls, plus one source/source diagnostic for each of eight profiles. The phasors yield ratio 1, 0 and sqrt(0.5), respectively. Source/source diagnostics retain ratio 1 and zero phase gap on supported samples. Tests also check known harmonic phase recovery, silence refusal and invalid inputs. Saved plot arrays are sampled every eight audio samples; summaries use full sample resolution.

## Verification and next action

The full suite passes **123 checks**: [saved log](../runs/phase_feasibility_regression_2026-10-14.txt). The new step adds numerical diagnostics and 64 array artifacts only. Existing 26/32 baseline and 27/32 taper audits, the taper's refused advancement and all prior candidate WAVs remain unchanged. Both maintained notebooks include the results and the owner-directed reviewer deferral. Fresh execution at `runs/notebooks/0ffb99cc1da14d1ea353e0349419f4ca` has zero cell errors, fourteen plots and twenty-two audio controls. Source/executed/HTML, original-input/backend, code-snapshot, declaration and array hashes verify; no new candidate WAV exists in the diagnostic run. The phase panel plot was visually checked. See [final verification](../runs/phase_feasibility/d215daf94aa749ce8798f72ab13976d9/final_verification.json) and the current evidence tracker.

**Next — Gate F (open):** predeclare and test one phase-aware joining candidate on the same eight profiles/four curves. Require original-request interior targets, exact zero/time/outside/rest preservation and the unchanged full audit. Report all lost/gained cases and both fixed YIN windows; include these phase/amplitude observations as diagnostics, not a new acceptance shortcut. Fundamental alignment alone is insufficient to promise harmonic/timbre or natural-vocal preservation. Keep the candidate separate from active processing and stop at controlled evidence before deciding on a bounded training-vocal comparison.

**Future — Gate D/J4 (deferred):** add actual experienced musical review when reviewers are available. Meanwhile automatic screening, known-signal checks, existing annotation-based numerical checks and exploratory engineering work proceed under the [revised no-reviewer sequence](Reviewer_Deferral_2026-10-14.md). Zero research-ready examples and no accepted repaired package are credited. The future dual-OS pipeline remains planned.

Reproduce only when a new fixed diagnostic run is needed, from this component folder:

```powershell
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.phase_feasibility
```
