# Phase 1 verification — 27 September 2026

**Implemented:** unchanged musical-audio path, controlled stops, frozen acoustic analysis and inspectable outputs. All project artifacts remain inside `experiments/bpdr_input`. Raw vocadito is read-only and external.

## Observed results

| Check | Actual result | Interpretation |
|---|---|---|
| Automated suite | 17 tests pass: 5 foundation, 11 Phase 1 integrity/failure checks, 1 real frozen-PESTO smoke check. | Tested execution and contract behavior; not general musical accuracy. |
| Full-recording replay | 40 recordings: 21 `ACCEPT_UNCHANGED`, 19 `RERECORD_REQUIRED`, 0 technical failures. | All 19 stops are the documented >20-second scope limit, not evidence of poor singing. |
| Original preservation | 40/40 preserved-copy SHA-256 values match source files. | No original recording was altered. |
| Accepted audio verification | 21/21 audio checksum/metadata/value checks pass. | Consumer receives the checked exported working copy. |
| Timeline | 21/21 accepted durations differ from source by at most one 22,050 Hz sample. | Full time origin retained; no phrase truncation. |
| Acoustic shapes | 21/21 have aligned finite 70-/71-feature matrices. | NaN unvoiced F0 stays outside the model matrix; masks remain explicit. |
| Rejected exports | 19/19 have no accepted WAV or accepted path. | Stops propagate rather than passing an unchecked original downstream. |

Full replay evidence: `runs/phase1_dataset/summary.json`. Individual rows point to the corresponding diagnostic, original, accepted output, feature arrays and plot/report. Automated test evidence: `runs/phase1_tests.txt`.

Example accepted report: `runs/phase1_dataset/fbedcabc9c5d46899eacfe86d29e74d9/report.html` (`vocadito_10.wav`, full 9.098-second recording). Example duration stop: `runs/phase1_dataset/93675e3bae0842288219d133e2993834/diagnostics.json` (`vocadito_1.wav`). These local outputs are ignored by Git; reproduce them with the README commands rather than committing recordings.

## Meaningful checks exercised

Corrupt WAV, zero-frame WAV, unsupported extension, non-finite float samples, silence, excessive clipping, >20-second input, antiphase stereo, internal rests in FLAC, odd resampling lengths, 48 kHz resampling overshoot with logged gain, identical output across repeated processing, missing analyzer, invalid analyzer timestamps and tampered accepted export. Non-accepted results block the downstream consumer.

Unit pitch fixtures deliberately create an unvoiced region so missing pitch cannot become a zero-Hz note. Their diagnostics identify injected pitch separately. The real-model test checks a 220 Hz constructed tone, seconds/Hz units, disabled gradients, evaluation mode and reuse of the same model. Its broad frequency bound is a smoke test, not a cents-accuracy benchmark.

## Analyzer and reproducibility

Runtime tested: Windows, Python 3.12.14, CPU. Resolved package versions and sources are locked in component `uv.lock`. Recorded runtime versions: PESTO 2.0.1, librosa 1.0.0, SoundFile 0.14.0. PESTO's packaged `mir-1k_g7` checkpoint is hashed in every diagnostic; no BPDR checkpoint is present. PESTO is frozen and loaded once per process.

Working waveform: mono 22,050 Hz PCM24. Features: centered 1024-sample STFT windows, 441-sample hop, 64 log-mel bands, cents relative to 440 Hz, confidence, voiced mask, valid voiced pitch changes, log-energy dB, onset and provisional boundary score. Native pitch timestamps remain saved. Nearest-frame mapping never interpolates across unvoiced gaps.

Technical thresholds are provisional, centrally recorded in diagnostics. No dataset reference annotation, intended pitch, key or genre is supplied to the analyzer. No reference annotation validation or pitch-accuracy evaluation has been performed here.

## Proposal and PP1 evidence

| Requirement / assessment area | Evidence available | Remaining work |
|---|---|---|
| FR1 short input | WAV/FLAC command entry point, supported-limit checks. | Recording capture or final upload screen. |
| FR2 original/failures | Preserved bytes and controlled technical/re-record outcomes. | Final user wording and wider recordings. |
| FR3 assessment/preparation | Provisional technical checks and measured diagnostics. | Musical usability calibration, confidence/controller policy and human review. |
| FR4 timing/rests | Unchanged path and resampling integrity tests. | Local musical repair and renderer checks. |
| FR5 playback/feedback | Per-run original/output audio controls, reasons and report. | Prepared-audio comparison and user feedback. |
| FR6 accepted export | File-level validator and rejection propagation tests. | Live teammate consumer and render rollback. |
| FR7 analysis/logs | Aligned measured F0 plot, feature arrays, hashes, versions, operations and empty edit intervals. | Prediction/correction overlays after model implementation. |
| LO2 technology; LO3 design/practices | Actual feature pipeline, deterministic audio processing, tests and saved provenance. | Research losses, trained conditions and integration. |
| LO1 novelty proof of concept | Foundation only. | Four-view interventions, paired-response training and comparison in P2–P5. |

**Phase 1 engineering checks are verified.** A human musically experienced reviewer has not signed off the example as a usable performance; retain that gate item for review. Playback controls and report assets exist, but listening/browser rehearsal on the presentation machine still needs a human check. This file does not certify general musical usability, research novelty, repair accuracy, live team integration or a project completion percentage.

## Next phase

Phase 2 joins and validates annotations, partitions by singer before augmentation, records reviewed references and mapped <=20-second crops, and builds audited original/detuned boundary views. The 19 longer recordings remain intact and become eligible only through explicit crop provenance. No pitch repair should be demonstrated as BPDR until its model and rendering gates pass.
