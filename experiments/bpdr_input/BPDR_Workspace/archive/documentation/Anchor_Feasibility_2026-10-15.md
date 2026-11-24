# Gate F: source-anchored phase/pitch feasibility — 15 October 2026

**Current gate: F — renderer validation; open.** The conditional feasibility step is complete. The fixed sign/magnitude-preserving bridge has mathematical solutions for **8/32 constructed cases**, covering **9/36 supports**. This is not an audio pass count: no new renderer, candidate WAV, model fit, research pair or accepted repair is issued. Gate D/J4 remains deferred; engineering continues without experienced reviewers.

## Declared model and requirements

[Evidence](../runs/anchor_feasibility/fa470618ad02487d90cb4436c2864e47/evidence.json), [declaration saved before measuring](../runs/anchor_feasibility/fa470618ad02487d90cb4436c2864e47/declaration.json). All eight previous known harmonic profiles and four curves are retained. The rested profile has two edit supports, giving **32 cases and 36 supports**. These are dependent constructed variants, not independent singers. Saved original WAV samples match the original generator exactly; prior experiment artifacts/source hashes are preserved.

The question is conditional: **can a direct harmonic resynthesis model with unchanged amplitude, fixed interior correction, two 80-ms ramps and matched fundamental phase at both endpoints satisfy the proposed bounds?** The known constructed F0 is used, not an estimator or intended-note guess. This is not a universal restriction on every audio editing method, nor a validated phase requirement for natural vocals.

Phase rate determines frequency; see the analytic-signal example in [SciPy's Hilbert documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html). For this model, the accumulated added phase in cycles on the actual sample grid is:

`added_cycles = sum((f_new - f_source) / sample_rate)`.

To recover source fundamental phase modulo 2*pi at the far boundary, added cycles must be an integer. Harmonics that share this fundamental phase then close modulo their respective integer cycles as well. This is a mathematical model condition, stronger than merely matching one endpoint sample value. It assumes well-defined phase/amplitude on these supports, not a zero-amplitude phase reset.

Use the original 20-ms supplied curve interpolated to source samples. Hold its correction fixed outside the two actual 80-ms ramps, including the guarded audit interior. Set correction to zero at first and last support samples. Inspect two separately declared ramp hypotheses:

- **Sign/magnitude preserving:** at each ramp sample, correction lies between min(0,requested) and max(0,requested). This disallows reversal or exceeding the local supplied amount.
- **Relaxed diagnostic:** ramp correction may be anywhere in [-30,+30] cents. This can reverse direction or exceed a smaller local request; it is not a permitted repair policy.

These are new explicit feasibility hypotheses. They do not change the existing renderer/permission/audit rules. No threshold is relaxed to make an existing failure pass.

First integrate the pointwise minimum/maximum possible ramp frequency changes with the fixed interior. An integer inside this interval is a **necessary condition**, not proof of a smooth feasible bridge. Next use one fixed sin-squared frequency-offset basis per ramp. Compute each coefficient's interval from all pointwise bounds. The sum of their cycle contributions gives a narrower interval. If that interval contains an integer, select the integer nearest nominal tapered cycles (lower on ties) and use one deterministic interpolation fraction through both coefficient intervals. No search, clipping, altered support or outcome-dependent basis tuning is used.

Validate phase residual <=1e-10 cycles, positive frequency, pointwise bounds, unchanged interior targets and zero endpoint correction. The 1e-8-cent roundoff check is a numerical equation check, not a new audible or pitch-audit tolerance. Saved frequency/correction/phase arrays are diagnostic targets only.

## Results

| Hypothesis | Supports meeting necessary integer condition /36 | Supports solved by fixed basis /36 | Cases with every support solved /32 |
|---|---:|---:|---:|
| Sign/magnitude preserving | **18** | **9** | **8** |
| Relaxed +/-30 diagnostic | 36 | 36 | 32 |

For the stricter hypothesis, **18 supports cannot satisfy this conditional phase budget**, nine more meet the necessary budget but are not solved by this particular basis, and nine have validated mathematical solutions. Failure of the fixed basis alone is not evidence that every other smooth bridge is impossible.

All eight stricter solved cases are the **crossing -30-to-30 curve**, one per profile; the rested case contributes two solved supports. The other 24 cases are not justified for the next sign-preserving analytic demonstration. These eight cases are not eight successful repairs, and none is research eligible. Maximum phase-equation residual among strict solutions is about 6.60e-16 cycles; the relaxed diagnostic's maximum is about 1.78e-15.

**Concrete incompatible example:** constant-220/+20, one-second support. The sign-preserving necessary interval is about **[2.14727, 2.53061] added cycles**, containing no integer. No arbitrary choice of ramp frequencies within those sign/magnitude bounds can satisfy both phase anchors in this model. The fixed-basis interval is narrower still, about [2.32508,2.40289].

The relaxed solver reaches integer 2, but its curve dips to about **-25.42 cents** during a requested upward correction. Numerical phase closure alone would conceal an opposing boundary intervention. The relaxed solution is therefore diagnostic, not approved for the next strict demonstration. Do not expand bounds, reverse direction or alter the original request to report success.

**Concrete supported example:** glide-220/crossing closes at integer 0, with effective correction about -25.54 to +25.30 cents and unchanged original interior targets. This remains a known-source analytical solution; it has not been rendered or independently remeasured as audio.

## Verification and next action

Five new checks cover inclusive integer intervals/ties, an analytically incompatible constant example, a bounded closed solution with preserved endpoints/interior, exact zero-request closure and refusal of invalid inputs. The full suite passes **134 checks**: [saved log](../runs/anchor_feasibility_regression_2026-10-15.txt). There are 36 array artifacts, source snapshots and a frozen declaration. No native resynthesis or new audio candidate was run. Previous 26/32 baseline and refused 27/32 taper/26/32 polar results remain historical and unchanged; feasibility counts do not replace those audits.

Both maintained notebooks show all cases, the strict/relaxed distinction, an incompatible example and a closed mathematical example. They retain original review/refusal gates and the owner deferral. Fresh execution at `runs/notebooks/56999c1ddb464dfab626942734813cf0` has zero cell errors, sixteen plots and twenty-eight audio controls. Source/executed/HTML, declaration, prior-input/review, code-snapshot and array hashes verify; every saved mathematical solution is independently rechecked for closure, bounds, endpoints and interior target. The new panel plot was visually checked. See [final verification](../runs/anchor_feasibility/fa470618ad02487d90cb4436c2864e47/final_verification.json) and the evidence tracker; prior executed notebooks remain historical.

**Next — Gate F (open):** predeclare a bounded known-source analytical audio control for the eight strict fixed-basis cases, explicitly withholding the other 24. Reconstruct from known harmonic-source phase, then test actual saved audio with unchanged audit, both fixed YIN windows, phase/amplitude and exact identity/time/outside/rest checks. Keep this separate from a practical raw-vocal renderer: known construction F0/phase is not available as ground truth for ordinary input. The control can diagnose a later source-anchored method; it cannot itself validate natural-vocal repair or accepted delivery.

**Future — Gate D/J4 (deferred):** add actual experienced musical review when reviewers become available. Follow the [no-reviewer engineering sequence](Reviewer_Deferral_2026-10-14.md) now. Research-ready examples remain zero; production permission/uncertainty, musical quality and independent consumer validation remain open. Future dual-OS integration remains planned.

From this component folder, reproduction creates a new declared diagnostic run:

```powershell
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.anchor_feasibility
```
