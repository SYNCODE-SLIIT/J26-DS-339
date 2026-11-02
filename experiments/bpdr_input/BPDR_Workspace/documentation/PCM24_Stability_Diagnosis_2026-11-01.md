# Development PCM24 stability diagnosis

Declared 1 November 2026. Reconstruct all11 permitted development candidates,
including successful controls, with unchanged curves/supports/renderer/auditor.
Measure source, in-memory candidate and PCM24 round-trip using the locked
precision auditor. Bind to existing encoded files when present. Encoding a
candidate that failed pre-render audit is a private hypothetical probe, not a
delivered repair or changed acceptance state. Save measurements and opposing
frame details, sample quantization error and exact outside-support identities.

Independent replay reconstructs all11 candidates/PCM24 probes and re-infers all33
source/float/encoded measurements using separate Torch float64 decoding. Record
strict-sign backend disagreement, not just aggregate errors. No thresholds,
frames, labels, model, supports, dataset denominator or selection change.

Known development remains48 attempted/24 scoreable/24 excluded,12 originals,
two scoreable groups and two unassessed groups. The frozen V34 qualification
still fails at male6 error reduction24.744%<25%; fresh audio stays unopened.

Run `diagnose_pcm24_stability.py run`, then `verify`. Evidence:
`results/transformer_usability/2026-11-01_pcm24_stability`.


## Completed diagnosis

All11 candidate waveforms and33 source/float/PCM24 measurements independently
replay, including the strict opposing masks, with no backend disagreements.
Case26 has three opposing coarse shifts of about+1.5376e-12c despite requested
negative shifts near-19 to-23c and matching fine direction. They appear before
and after encoding. Case45 has no in-memory veto but one encoded-only coarse
shift-1.3454e-12c at11.50s against a+13.3767c request; fine measures+13.2287c.
Maximum sample quantization error for that case is1.1895e-7. Its exact detailed
reason is now reconstructed as `COARSE_DIRECTION`; no failed file is accepted
by this diagnostic. Source/delivered outside-support samples remain exact.

These are numerical/measurement-stability observations, not proof of intended
pitch or listening quality. The subsequent [centered arithmetic study](Centered_Auditor_Study_2026-11-01.md)
keeps the same weighted estimator but computes it around the existing peak to
avoid large absolute-pitch multiply/divide/subtract cancellation. No epsilon,
rounding-to-zero rule or relaxed gate is introduced.
