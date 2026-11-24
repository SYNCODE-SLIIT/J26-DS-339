# Gate F: controlled boundary-taper comparison — 14 October 2026

**Current gate: F — usable delivered audio, renderer validation; open.** The declared boundary-taper experiment is complete. It does **not** meet the requirements for moving to the planned bounded vocal comparison. Preserve it as unadopted evidence. Gate D/J4 actual independent reference approvals remain pending; research-ready examples remain zero.

## Fixed experiment

[Evidence](../runs/boundary_comparison/e69c9a73c3f54f7c843701f5dadbf522/evidence.json) and [declaration saved before measuring](../runs/boundary_comparison/e69c9a73c3f54f7c843701f5dadbf522/declaration.json). All eight prior constructed profiles and four supplied curves are tested on the same source WAVs, supports, analyzer/checkpoint, pinned native backend and policies. There are 32 variants of eight constructed signals, not 32 independent vocal performances.

One candidate, `praat-boundary-taper-candidate-0.1`, multiplies the original 20-ms cents curve by an 80-ms raised-cosine envelope before native pitch-tier interpolation. The waveform blend and local supports remain those of the prior renderer. This hypothesizes that smoothing the tier correction at boundaries might reduce transition estimates. It does not implement phase alignment. There is no parameter search, additional candidate or outcome-dependent exclusion.

Original requests are validated **before** tapering, so multiplication by zero cannot hide an over-limit or outside-support request. Guarded interior requests remain exactly the original values; audits use those original requested targets, not the attenuated boundary curve. No scale/MIDI target snapping, duration change, gain normalization or accepted output is introduced. The old framewise renderer, framewise audit and active v0.3 laboratory renderer/policy are unchanged.

## Results and advancement decision

| Check | Baseline | Boundary taper |
|---|---:|---:|
| Full unchanged candidate audit | 26/32 | 27/32 |
| Fine median/p90 error limits | 32/32 | 32/32 |
| Exact physical/outside/rest/interior-target preservation | 32/32 | 32/32 |
| Exact zero-edit identities | 8/8 | 8/8 |

The candidate loses **vibrato 220 / crossing -30-to-30**, an existing baseline pass. It gains **constant 880 / -20** and **vibrato 880 / rising**. The five candidate rejects all report `COARSE_DIRECTION`, with no other failed audit check. Those decisions remain failures; the extra aggregate pass is not evidence of improved musical accuracy. The maximum candidate fine p90 error is about 3.17 cents. Do not interpret it as a population accuracy estimate or a validated natural-vocal result.

| Fixed YIN window | Baseline peaks >30 cents /32 | Candidate /32 | Baseline largest peak | Candidate largest peak |
|---|---:|---:|---:|---:|
| 2048 samples, original auditor | 5 | 4 | 42.42 | 40.73 |
| 1024 samples, sensitivity diagnostic | 11 | 9 | 58.24 | **59.40** |

These are absolute **relative YIN transition changes** in the existing ramp/guard windows. They are not an instantaneous waveform-pitch target, audible artifact judgement or musical pass rate. Both worst candidate peaks occur in constant-110/rising. Per-case peaks increase in 12 cases at the original window and 14 at the shorter window; the complete lists and arrays are saved. Summary improvements do not imply every case improved.

The predeclared engineering advancement filter requires all preservation/identity and fine checks, no loss of any baseline full-audit pass, no worsening of either peak count or maximum at either window, and some transition-summary improvement. It is a filter for considering the next bounded training-vocal/listening experiment, **not an audio acceptance rule**. The result fails two conditions: a lost baseline pass and the larger 1024-window maximum. Thus `eligible_for_bounded_vocal_comparison=False`, `adopted=False`, with no accepted prepared package or research pair. No audit epsilon or threshold is changed to remove these failures.

## Verification and panel explanation

The new renderer checks original-request refusal, exact zero backend bypass, actual local-render geometry/outside preservation and unchanged interior targets. Advancement checks prevent aggregate improvement hiding a lost baseline pass or choosing only a favorable window; meeting the filter still cannot adopt/accept audio. The full suite passes **119 checks**: [regression log](../runs/boundary_comparison_regression_2026-10-14.txt).

All 32 candidate FLOAT WAVs round-trip exactly; their hashes and 32 diagnostic-array hashes are recorded, with code snapshots and immutable declaration hash. Both maintained notebooks show the changed decisions, both windows and two fixed playback/plot illustrations: constant-110/rising and vibrato-220/crossing. These examples expose the secondary-window peak and lost audit pass, rather than selecting only successes. Historical notebook copies remain evidence. Fresh execution at `runs/notebooks/452f4752612c40e8943904392385563b` has zero cell errors, thirteen plots and twenty-two audio controls. Source/executed/HTML, declaration, original-input/backend and all candidate-array/WAV hashes verify; the new panel plot was visually checked. See [final verification](../runs/boundary_comparison/e69c9a73c3f54f7c843701f5dadbf522/final_verification.json) and the current evidence tracker.

## Next action by gate

**Next — Gate F (open): phase-aware transition feasibility.** Predeclare a different boundary-joining strategy and fixed phase/amplitude diagnostics on the same controls before another renderer comparison. The taper-only strategy did not resolve the issue. Retain the original-request audit, all baseline decisions, identity/time/outside/rest checks and both fixed YIN windows. Do not tune this candidate after these outcomes or expand to real vocals as though it passed. Any later bounded training-only Vocadito/Jamendo comparison requires justified controlled evidence and matched musical listening; actual delivery additionally requires validated model permission, uncertainty and musical preservation.

**Parallel — Gate D/J4 (pending):** complete the two independent original-reference reviews using the existing packet. Gate B independent voice/support coverage, Gate E matched model research evidence and Gate G/H1 accepted consumer validation remain open. The future Windows/macOS pipeline remains planned; this experiment does not implement it.

To reproduce a new fixed run, from this component folder:

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.boundary_comparison
```

This preserves original measurements and writes a new declared evidence directory. It does not train, extract more songs, alter reviews or adopt a renderer.
