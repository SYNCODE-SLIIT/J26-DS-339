# Duration-preserving framewise candidate - 14 October 2026

An experimental renderer now consumes changing relative correction curves. The controlled run passes **26/32 full candidate audits** and all 32 fine pitch-error checks, with eight exact identities and a verified supplied-fixture transaction. **The candidate is not adopted.** No real vocal was repaired, accepted repaired package issued or research-ready example created. Runtime uncertainty/permission and real musical review remain unvalidated.

## Method and dependency

The proposal names Praat/Parselmouth. This implementation uses its measured PitchTier and overlap-add resynthesis: each native pitch point is scaled by `2**(interpolated_cents/1200)`, with no intended MIDI/scale snapping. The original DurationTier remains unchanged. This follows the documented [Parselmouth manipulation interface](https://parselmouth.readthedocs.io/en/stable/examples/pitch_manipulation.html) and [Praat Manipulation object](https://praat.org/manual/Manipulation.html).

Full resynthesis stays private. A raised-cosine envelope exposes only declared edit supports and copies original samples exactly elsewhere. Zero correction bypasses the backend. Sample count, rate, time origin and first-sample position must match exactly; the implementation refuses mismatches rather than trimming/padding. Nonfinite/clipped results are refused without automatic gain normalization. Invalid/over-limit curves, overlapping/short supports or corrections outside support stop before rendering. Missing measured pitch points also stop.

`praat-parselmouth==0.4.7` is an optional `rendering` dependency in the component's project/lock files. Only that package was added; existing locked package versions are unchanged. The installed candidate backend reports Parselmouth 0.4.7 / Praat 6.1.38, tested on Windows 11 with Python 3.12.14. Package release information is recorded by [PyPI](https://pypi.org/project/praat-parselmouth/); macOS execution and later dual-OS pipeline integration remain unverified. No shared/team dependency file was changed.

## Frozen constructed experiment

[Evidence](../runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json). Eight profiles and four curves were declared before measuring outcomes:

- Four-second harmonic constants at 110/220/440/880 Hz; 35-cent, 5-Hz vibrato at 220/880 Hz; a rising glide at 220 Hz.
- An eight-second harmonic source at 220 Hz with an exact silent rest at 2.5-4.5 seconds.
- Supplied -20-cent and +20-cent curves, a rising 10-to-30-cent curve, and a -30-to-30-cent crossing. Tonal supports are 1.0-2.0 seconds; the rested case additionally uses 5.0-6.0 seconds. Only 25% of samples are in edit supports. Waveform ramps and analysis guards are excluded from pitch targets.

There are **32 nonzero renders** and **eight exact identity cases**, not 32 independent vocal performances. All nonzero candidates preserve sample count/time origin and exact outside-support samples; the silent rest remains bit-identical. Original/candidate float-WAVs were reloaded and matched generated samples exactly. Source snapshots, native backend hash, policies, per-case results and waveform artifacts are preserved.

## New per-frame candidate audit

`framewise_audit.py` compares realized YIN/PESTO changes with the requested curve **at each audited frame**, rather than replacing the changing curve with one constant target. It retains the v0.3 fine median/p90 limits (5/10 cents), coarse limits (20/40 cents), minimum frame counts, 75% common coverage and outside-drift limits. Additionally, every common frame with requested magnitude >=10 cents must have the correct fine direction and nonopposing coarse direction. This is stricter than the existing constant-shift median-direction check.

The candidate audit is separately versioned `framewise-render-audit-0.1`. The active v0.3 laboratory policy and generator remain unchanged. These full-pass counts cannot be compared directly with earlier median-direction or fine-only benchmarks.

| Supplied curve | Full candidate audit passes /8 | Analytic fine oracle passes /8 |
|---|---:|---:|
| -20 cents | 4 | 8 |
| +20 cents | 8 | 8 |
| Rising 10-to-30 cents | 7 | 8 |
| Crossing -30-to-30 cents | 7 | 8 |

All 32 rendered cases meet the fine median/p90 limits: maximum median error about **1.30 cents**, maximum p90 about **3.04 cents**. All 32 analytic oracle cases meet the fine limits, with maximum p90 about 2.98 cents. Oracle waveforms diagnose the estimator on a known contour; their changed accumulated outside phase is not an admissible local renderer and is not preservation evidence.

Six cases remain rejected solely by the stricter per-frame **coarse-direction** gate:

| Profile / curve | Opposing coarse frames |
|---|---:|
| Constant 220 / -20 | 3 |
| Constant 880 / -20 | 1 |
| Constant 880 / crossing | 1 |
| Vibrato 880 / -20 | 2 |
| Vibrato 880 / rising | 1 |
| Rested 220 / -20 | 9 |

Do not erase these failures or lower tolerances to increase counts. Fine agreement and a coarse-direction failure must be diagnosed separately, including tracker precision/activation variability, before a different consistency rule could be justified. These known-signal results do not establish real separated-vocal suitability, audible artifacts, expressive preservation or population accuracy.

**Transition limitation:** the [separate read-only diagnostic](../runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/transition_diagnostic.json) measures YIN relative-shift peaks in excluded ramp/guard windows. Five of 32 cases exceed 30 cents in that estimator, with a maximum about 42.42 cents. The passing vibrato-220/rising illustration has a transition peak about 36.87 cents. These are estimator observations, not a proven instantaneous pitch or human musical judgement; a waveform crossfade has no known linear pitch target. They demonstrate why the guarded interior audit is insufficient for accepted delivery. Transition preservation remains open, with no new threshold or policy relaxation.

## Supplied-fixture integration

One eight-second constant-220-Hz fixture supplies a varying 10-to-20-cent proposal over 1.0-1.9 seconds, plus explicit alternate/uncertainty/expressivity test inputs. The chain executes **adapter -> permission planner -> candidate renderer -> independent framewise audit -> transaction**. It returns `VERIFIED_FIXTURE_CANDIDATE`, with fine median/p90 errors about 0.88/1.87 cents and an affected sample fraction of 11.25%.

This is a constructed control-flow test, not a learned repair. Its transaction has `accepted_audio=None`, `production_eligible=False`; no `ACCEPT_PREPARED` package exists. The existing smoke-model runtime preview still requests no edits and still lacks calibrated uncertainty/permission. The artificial constant-shift renderer is not replaced.

## Validation, notebooks and remaining gates

All **108 regression checks passed**: [log](../runs/framewise_regression_verified_2026-10-14.txt). New checks exercise the actual optional backend's count/time/outside preservation, exact zero identity, invalid curves/supports, per-frame signed targets and rejection of an inappropriate constant shift. Both maintained notebooks include this stage, with a passing changing/vibrato illustration and a rejected case, source/candidate playback and failure reasons. No extra maintained notebook is added.

Next: diagnose the six coarse-direction rejects without weakening the active policy, then predeclare a bounded comparison on existing training-only vocal crops using supplied test curves and unchanged source/provenance gates. Assess real-audio transition/timbre/expressivity and listening observations before adoption. Keep actual original-reference reviews, independently identified voice coverage, calibrated uncertainty/permission, latency and accepted delivery open. Research fitting remains gated. Future macOS/Windows pipeline work remains deferred.

From this component folder, install the optional dependency using the existing component-local environment:

```powershell
uv sync --locked --group notebooks --extra rendering
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& '.\.venv\Scripts\python.exe' -B -m bpdr_input.framewise_fixtures
```

The replay creates a new evidence directory. Source data, reviews and prior runs must remain unchanged. See [final verification](../runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/final_verification.json).
