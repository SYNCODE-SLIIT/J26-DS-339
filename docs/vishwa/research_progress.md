# Perera E. V. — research progress and decision log

Project J26-DS-339 · Harmony-Oriented Melody Understanding · branch `feature/vishwa-pp1-melody-understanding`

## Logging rules

Use actual session dates (Asia/Colombo). Preserve failed experiments and superseded
choices. Never backdate entries or describe planned work as executed. Each experiment
records its question, hypothesis, changes, configuration, commands, results, issues,
interpretation, decision, next experiment, and related implementation commit.
Small aggregate reports/configurations belong in Git; raw audio, per-file manifests,
prediction caches and checkpoints stay under ignored `datasets/humtrans/derived/`.

## Status (2026-10-21)

| Stage | Status | Evidence / remaining work |
|---|---|---|
| A: exploration | Completed initial audit | Executed notebook 01; independent MIDI audit completed; original timing finding corrected |
| B: preprocessing | Partial | Resampling, optional trim with offsets, Mel, pYIN; CFP pending |
| C: baseline | Pilot completed | 20 TRAIN / 10 VALID recordings; low F1, no full benchmark |
| D: CNN–Transformer | Pending | No training or performance claim |
| E: harmony representation | Partial | Predicted duration-weighted pitch-class histogram only |
| F: controlled comparisons | Pilot completed; diagnosis in progress | Smoothing did not improve validation; preserve default baseline |

## Decision register

| ID | Date recorded | Decision | Evidence / limitation |
|---|---|---|---|
| D001 | 2026-10-21 | Python modules hold reusable methods; notebooks explain experiments | Same methods must be used in experiments and integration |
| D002 | 2026-10-21 | Preserve original timestamps and raw audio; no trim by default | Note timing must remain comparable to original MIDI |
| D003 | 2026-10-21 | Keep official splits and reserve TEST for final evaluation | Filename composition IDs are disjoint; all ten singers cross splits |
| D004 | 2026-10-21 | Independently verify annotations before scoring predictions | Notebook flagged 834 files with MIDI tails >50 ms beyond audio |
| D005 | 2026-10-21 | Null unsupported outputs, document partial objectives | HumTrans supervision for beat/key/phrase tasks remains unestablished |
| D006 | 2026-10-21 | Allow local focused commits under new user authorization | No merging, destructive Git operations, or changes to other components |

## R000 — baseline inherited at session start (retrospective)

- **Recorded:** 2026-10-21. This entry summarizes earlier verified work; it does not invent a new execution date.
- **Objective:** establish dataset integrity and a runnable vocal-to-note baseline.
- **Changes:** notebook 01, audio/Mel/pYIN/note modules, JSON CLI, three contract tests.
- **Configuration:** HumTrans 14,614 pairs; 22,050 Hz analysis, 4096-frame / 256-hop pYIN; 60 ms minimum note duration.
- **Commands previously executed:** `.venv/bin/python -m unittest discover -s tests -v`; `.venv/bin/python scripts/analyze_vocal.py datasets/humtrans/downloads/wav_data_sync_with_midi/F01_0001_0001_1_D.wav --output datasets/humtrans/derived/baseline/section01_example.json`.
- **Actual prior evidence:** all pairs covered exactly once by official splits; 56.218 hours; notebook reported 391,595 notes. Three tests passed. One 8-second clip produced 690 frames and 19 predictions in approximately 2.61 seconds; notebook reference count was 16.
- **Problems:** no transcription accuracy measured; notebook custom MIDI parser needed independent confirmation; 834 files had note tails beyond audio; clipped-peak proxy is not proof of clipping.
- **Interpretation:** baseline execution succeeds, but note counts alone do not establish accuracy.
- **Decision:** independently audit annotation timing, then build quantitative pilot evaluation.
- **Next:** R001.
- **Related commit:** `d027a04` (already present when this session began).

## R001 — independent annotation audit (completed)

- **Actual date:** 2026-10-21.
- **Question:** are the note counts and tail overruns reproducible with maintained MIDI readers?
- **Hypothesis:** independent tempo-aware readers confirm the custom parser's totals; disagreements must be investigated before label creation.
- **Changes/configuration:** planned reusable Mido reader, PrettyMIDI cross-check, full-corpus metadata manifest; 50 ms tail tolerance.
- **Dependencies installed:** mido 1.3.3, pretty-midi 0.2.11.post0, mir-eval 0.8.2 using `uv pip install --python .venv/bin/python mido pretty-midi mir-eval`.
- **Results:** pending execution; no audit conclusion yet.
- **Next:** record actual audit outputs and freeze a deterministic training pilot cohort.

## Entry template

Copy for each meaningful experiment: actual date; question; hypothesis; changes;
dataset/split and configuration; exact commands; actual results; problems;
interpretation; decision; next experiment; related commit and report path.

### R001 outcome and correction — 2026-10-21

- **Command:** `.venv/bin/python scripts/audit_humtrans.py --manifest datasets/humtrans/derived/research/manifest.jsonl --report experiments/vishwa/results/alignment_audit.json`.
- **Actual results:** both readers agree within 1e-7 seconds on all 14,614 pairs and 391,595 notes. Zero read errors, unmatched events, overlapping notes, missing splits, or tails >50 ms. Maximum tail overrun 0.000022676 s (one 44.1 kHz sample).
- **Root cause of earlier failed finding:** custom notebook `tempos.sort()` sorted equal-tick events by tempo value. Default `(0, 500000)` could override a faster explicit tempo at tick zero. Example F01_0019_0001_1: old note end 11.948958 s, corrected 11.029797 s, actual tempo 461538 μs/beat.
- **Correction:** replaced notebook custom parser with PrettyMIDI, executed all cells, added duration/pitch histograms, added tempo regression tests. The old 834-file defect claim is retracted. Corrected maximum note onset is 28.125 s; minimum note duration 0.101 s.
- **Reproducibility fix:** sort file paths before seeded sampling. The newly fixed sample has 21/100 near-full-scale peaks (formerly 14/100 with filesystem-order-dependent sampling). Neither count proves clipping or estimates the entire corpus without uncertainty analysis.
- **Interpretation:** structural annotation timing is consistent; agreement is not proof of acoustic/perceptual alignment. PrettyMIDI uses Mido for binary parsing, so the independent check concerns note interpretation and tempo conversion rather than wholly independent file decoding.
- **Decision D007:** use maintained MIDI readers; preserve this failed finding and its correction. Do not exclude 834 valid recordings based on the obsolete parser result.
- **Next experiment:** R002 deterministic TRAIN pilot for note segmentation, then frozen VALID comparison; TEST remains reserved.
- **Related report:** `experiments/vishwa/results/alignment_audit.json`; implementation commit recorded in the next entry.

## R002 — controlled segmentation pilot (completed; original preregistration follows)

- **Date:** 2026-10-21.
- **Question/hypothesis:** does short median smoothing within voiced runs reduce vibrato-induced note fragmentation without bridging rests?
- **Configuration frozen before execution:** seed 339; two TRAIN examples per singer (20 total), one VALID example per singer (10 total), widths 1/3/5 frames, default 22,050 Hz/256 hop. Select by TRAIN macro onset+pitch note F1; tie favors smaller width. Compare baseline and selected width on VALID; no TEST inference.
- **Metrics:** Mir_eval one-to-one note matching at 50 ms onset and 50 cents pitch; report with and without offsets. Frame agreement uses MIDI-held notes as a proxy, not continuous ground-truth vocal F0.
- **Results:** pending. This small pilot cannot establish full-dataset or unseen-singer performance.

### R002 execution and decision — 2026-10-21

- **Command:** `.venv/bin/python scripts/run_baseline_pilot.py --output experiments/vishwa/results/pilot_001`.
- **Related annotation commit:** `e399480`; initial log commit `e7674fc`.
- **Actual TRAIN macro onset/pitch F1:** width 1 = 0.047592, width 3 = 0.047979, width 5 = 0.048927 (20 recordings, 586 reference notes).
- **Frozen VALID comparison:** width 1 F1 = 0.028077, width 5 F1 = 0.027592 (10 recordings, 300 reference notes). Offset F1 also fell: 0.022020 → 0.021604.
- **Frame agreement proxy:** 0.301738 TRAIN / 0.309404 VALID; MIDI voicing recall 0.926505 / 0.929407. These are MIDI-note-reference proxies, not continuous annotated F0 accuracy.
- **Verification:** eight tests passed; notebook 02 executed and recomputed one score from cached features. Plots inspected: weaker case shows onset/pitch drift; several clips show octave discrepancies.
- **Problem:** the training improvement was tiny and did not generalize in the pilot. Low F1 requires diagnosis before adding architecture complexity.
- **Decision D008:** retain unsmoothed baseline as default; keep median smoothing as an experimental variant. Do not claim improved accuracy.
- **Decision D009:** investigate octave and onset sensitivity using TRAIN only. Do not transpose labels from reference-driven guesses or singer identity.
- **Next:** R003 diagnostic sensitivity analysis; no TEST predictions.

## R003 — octave and onset sensitivity (planned before execution)

- **Date:** 2026-10-21.
- **Question:** how much does strict note scoring change when octave differences or onset tolerance are relaxed?
- **Configuration:** reuse the same 20 TRAIN predictions, width 1; compare exact pitch vs pitch class and onset tolerances 50/100/200 ms. No retraining or label changes.
- **Hypothesis:** octave and onset differences explain part of the low agreement; relaxed scoring is diagnostic and must not be presented as a model improvement.
- **External evidence inspected:** [HumTrans official repository](https://github.com/shansongliu/HumTrans) describes use of official splits and reference MIDI evaluation. [Dynamic HumTrans](https://arxiv.org/html/2410.05455v1), sections 1.2 and 1.3, discusses onset/offset annotation concerns and separate octave-aware/invariant evaluation. These motivate investigation but do not prove the cause of our specific errors.
- **Results:** pending.

### R002 reproducibility and commit notes

- **Execute report notebook:** use the NotebookClient example in `docs/vishwa/README.md` to ensure the project kernel and working directory.
- **Artifacts:** `experiments/vishwa/configs/baseline_pilot.json`, `experiments/vishwa/results/pilot_001/`, notebook 02. Detailed cached frame arrays are ignored under `datasets/humtrans/derived/research/pilot_features/`.
- **Rerun:** use a new output directory, e.g. `--output experiments/vishwa/results/pilot_001_rerun`; the script refuses to overwrite a previous experiment directory.
- **Scope:** no model or label correction accepted from this pilot. The exact eight tests and both executed notebooks are recorded in this milestone's commit.
