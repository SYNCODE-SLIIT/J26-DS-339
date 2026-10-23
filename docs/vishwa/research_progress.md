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
| F: controlled comparisons | Completed initial experiments | Smoothing and energy gates rejected; timing/octave cause remains open |

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

## R003 — octave and onset sensitivity (completed; original preregistration follows)

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

### R003 actual outcome — 2026-10-21

- **Command:** `.venv/bin/python scripts/diagnose_pitch_timing.py`.
- **Actual results (20 TRAIN clips):** macro frame agreement exact 0.301738 versus octave-invariant/chroma 0.482577. Exact note F1 at 50/100/200 ms: 0.047592 / 0.116345 / 0.266216. Chroma note F1 at the same tolerances: 0.060211 / 0.175032 / 0.401247.
- **Interpretation:** octave and onset differences materially affect scoring. These higher relaxed scores do not demonstrate a better model, and do not isolate label error from estimator error.
- **Decision D010:** retain strict baseline metrics and raw annotations; report relaxed criteria separately. Do not select per-recording transpositions or time shifts from reference labels.
- **Source clarification:** [HumTrans original paper, section 2.2.2](https://arxiv.org/html/2309.09623v1) describes `D` as a recording lowered by one octave, and `DD` by two. This does not establish whether the downloaded per-recording MIDI already incorporates the shift, so no automatic suffix-based correction is applied.
- **Evidence:** `experiments/vishwa/results/pitch_timing_diagnostics.json`; baseline evaluation commit `39acfed`.
- **Next:** R004 check low-frequency pYIN behavior on known synthetic pitches independently of corpus labels.

## R004 — synthetic pitch sanity (completed; original preregistration follows)

- **Date:** 2026-10-21.
- **Question/hypothesis:** does the configured baseline systematically estimate low pitches an octave too low even when the true signal is known?
- **Configuration:** 22,050 Hz; 3.6-second signal with 0.8-second tones at MIDI 45/57/69 (110/220/440 Hz), silent gaps, 10 ms edge ramps; clean and full-waveform 10 dB white-noise conditions; seed 339. Reuse unchanged production baseline.
- **Decision:** this only diagnoses implementation/parameter behavior on simple signals; it cannot validate real humming labels or prove robustness to vocal noise.
- **Results:** pending execution.

### R004 actual outcome — 2026-10-21

- **Command:** `.venv/bin/python scripts/synthetic_pitch_check.py`.
- **Results:** three notes extracted in both clean/noisy cases. Voiced-frame pitch agreement 1.0000 clean / 0.99517 noisy; interior pitch errors 0 cents clean and at most 10 cents noisy. Strict onset note F1 was **0 in both conditions**.
- **Interpretation:** no systematic octave failure on these known tones, but timing is inadequate even on simple signals. Long centered pYIN windows (4096/22050 ≈186 ms) can mark frames voiced before actual tone onset and after offset.
- **Decision D011:** investigate short-window energy masking as an onset/offset refinement; do not replace the pitch estimates or relax the metric to claim improvement.
- **Limitation:** synthetic success/failure does not prove the cause of real vocal octave discrepancies.

## R005 — short-window energy-gating experiment (completed; original preregistration follows)

- **Date:** 2026-10-21.
- **Hypothesis:** 512-sample energy windows can remove long-window voicing spill around rests and improve strict onset matching.
- **Configuration:** same cached TRAIN 20 / VALID 10 F0 tracks, width-1 segmentation, 512 RMS / 256 hop; compare no gate, -20 dB and -10 dB peak-relative RMS gates, and max(5% peak, 2×20th-percentile RMS). Choose by TRAIN macro note F1; baseline wins ties. Freeze before VALID scoring, then check the two known-tone conditions.
- **Risk:** quiet vocal frames may be rejected; report frame/voicing metrics alongside note F1. VALID has already been inspected during R002, so it remains development evidence, not a final independent evaluation.
- **Results:** pending execution.

### R005 initial result and diagnostic extension

All three energy gates reduced TRAIN note F1; the unchanged baseline won selection.
VALID therefore rechecks only baseline. Extend the independent known-tone check to
all four gating modes (instead of only selected baseline) to distinguish synthetic
boundary benefits from real-corpus failure. This does not change model selection or
use synthetic labels to tune the real-data validation result.

### R005 outcome and decision — 2026-10-21

- **Command:** `.venv/bin/python scripts/run_energy_pilot.py`.
- **TRAIN macro note F1:** unchanged 0.047592; -20 dB gate 0.028002; -10 dB gate 0.012622; adaptive noise gate 0.028213. All gates also reduce MIDI voicing recall (baseline 0.926505 versus 0.740347 / 0.464356 / 0.705818).
- **VALID:** baseline wins TRAIN selection; baseline score remains 0.028077. No rejected gate was selected using VALID.
- **Synthetic check:** every gate raises clean-tone note F1 from 0 to 1.0. On 10 dB noise, -20 dB remains at 0, while -10 dB/adaptive reach 1.0. This confirms a synthetic boundary benefit but does not transfer to real-data scoring.
- **Interpretation:** real-data boundary/label behavior differs from simple tones. Energy gating sacrifices voiced coverage; favorable synthetic results are insufficient for adoption.
- **Decision D012:** retain baseline defaults. Preserve all failed alternatives and their code as experimental only. Prioritize acoustic/annotation review and explicit target definitions over training a model on unverified frame labels.
- **Regression checks:** nine tests passed, including non-mutation and silence masking of the energy gate. Both notebooks executed; notebook 02 includes diagnostics and an octave-discrepancy example.
- **Source/results:** `scripts/diagnose_pitch_timing.py`, `scripts/synthetic_pitch_check.py`, `scripts/run_energy_pilot.py`, `src/melody_understanding/energy.py`; associated JSON under `experiments/vishwa/results/`.
- **Dependencies:** same versioned project/research environment and baseline feature configuration recorded in `pilot_001/summary.json` and `requirements-vishwa-research.txt`.
- **Commit association:** this entry and R003–R005 evidence are included together in the diagnostic milestone; locate it with `git log --oneline -- scripts/run_energy_pilot.py`.

## Current decision register additions

| ID | Date | Decision | Evidence |
|---|---|---|---|
| D007 | 2026-10-21 | Replace notebook MIDI timing parser; retract 834 flags | Full-corpus reader agreement; explicit tick-zero tempo regression |
| D008 | 2026-10-21 | Do not adopt median smoothing | Tiny TRAIN gain, lower frozen VALID note F1 |
| D009 | 2026-10-21 | Diagnose octave/timing errors on TRAIN | Large discrepancies observed; avoid reference-driven correction |
| D010 | 2026-10-21 | Keep strict metrics and relaxed diagnostics distinct | Relaxed onset/chroma scoring raises numbers without changing predictions |
| D011 | 2026-10-21 | Test short-window energy gating | Known pitches correct, synthetic note onsets fail strict scoring |
| D012 | 2026-10-21 | Reject energy gate as default | Better synthetic boundaries but lower real TRAIN F1/voicing recall |

## Verified milestone commits

- `d027a04`: inherited baseline and initial exploration notebook.
- `e7674fc`: structured log and research dependency pins.
- `e399480`: independent annotation audit and corrected notebook tempo interpretation.
- `39acfed`: fixed-cohort baseline evaluation and smoothing comparison.
- Diagnostic milestone: commit containing this R003–R005 outcome; hash shown in session handoff.

## Highest-priority continuation

1. Establish a small, explicitly reviewed TRAIN reference subset with waveform/spectrogram/pitch evidence and annotation provenance; determine whether octave discrepancies are estimator, label, or recording-convention effects. Do not infer corrections from filename suffixes alone.
2. Compare an independently implemented F0 estimator on that subset. Define tolerances and whether the task predicts performed pitch or intended score pitch before model training.
3. Build supervised feature/label tensors with recorded timing policy; then prototype the proposed CNN–Transformer and a matched simpler baseline. CFP and additional task labels still need implementation/verification.
4. Expand validation only under a documented protocol; reserve TEST. Do not extrapolate the current 20/10 pilot to full-dataset accuracy.

No raw recordings, frame caches or checkpoints were committed. No CNN–Transformer,
CFP, beat/key/phrase model or learned harmony embedding has been trained or validated.

## R006 — independent pitch and annotation review (preregistered 2026-10-21)

- **Question:** do octave disagreements persist across independent F0 implementations?
- **Hypothesis:** some differences persist between audio estimators and intended-score MIDI; this must not be mistaken for proof of corrected labels.
- **Configuration:** the same 20 TRAIN recordings in pilot_001; no VALID/TEST inference. Praat autocorrelation via praat-parselmouth 0.4.7, C1–C7 range, 256/22050 s step, silence threshold .03, voicing threshold .45. Map nearest native Praat frames within half a hop; never interpolate through unvoiced gaps.
- **Review rule frozen before execution:** shared-octave flag requires >=80% agreement between estimators within 50 cents on common voiced MIDI-active frames, with both median deviations within .75 semitone of the same nonzero octave. Flag interpretation remains provisional.
- **Planned evidence:** individual and aggregate comparisons, source audio/MIDI hashes, waveform/spectrogram/pitch plots, review status explicitly automated and pending human verification.
- **Target decision to follow evidence:** distinguish acoustic performed pitch from intended-score MIDI. Preserve annotations; no per-file transposition or timing corrections based on predictions.
- **Dependencies:** installed `praat-parselmouth==0.4.7` via uv into existing .venv; added reproducibility pin.
- **Results:** pending execution.

### R006 outcome — 2026-10-21

- Executed `scripts/review_annotations.py` and notebook 03 on the 20 TRAIN examples.
- Macro agreement between estimators on common voiced reference-active frames: 0.927279. Common-grid MIDI pitch agreement: pYIN 0.302030; Praat 0.276300.
- Eight examples met the preregistered shared-octave flag; twelve require timing/pitch inspection. These flags do not establish annotation errors or justify automatic correction.
- Decision D013: preserve performed F0 and original score MIDI as separate target meanings. Human listening/reference verification remains pending.
- Eleven regression checks passed, including nearest-frame mapping and a known-tone independent-estimator check.

## R007 — weak score feature preparation (preregistered 2026-10-21)

- Prepare the fixed 20 TRAIN / 10 VALID pilot only; no TEST processing.
- Target: rest=0 and original MIDI pitch modulo 12 plus 1. This is weak score supervision, not verified performed pitch.
- Features: normalized untrimmed audio, 22050 Hz; 80 Mel bands, 4096 FFT, 256 hop; relative log power. Keep original timestamps.
- Exclude supervision within 100 ms of note and recording boundaries. This fixed heuristic is not a proven timing correction and can remove short notes.
- Fit per-band mean/std only on TRAIN frames. Record source/tensor hashes; keep tensors ignored by Git.
- Hypothesis: this yields a reproducible feasibility dataset; no model quality improvement is claimed by preparation alone.
- Results: pending execution.

### R007 outcome — 2026-10-21

Executed `scripts/build_score_dataset.py`. TRAIN: 26,157 frames, 15,120 supervised; VALID: 12,817 frames, 7,382 supervised. Source and derived hashes recorded in `weak_score_dataset.json`. Original files unchanged. Decision D014: retain this as a weak-score feasibility dataset; the boundary mask removes about 42% of frames and does not validate remaining labels.

## R008 — small architecture feasibility pilot (preregistered 2026-10-21)

- Question: can the proposed acoustic encoder learn the weak score-chroma target on the fixed pilot?
- Compare majority TRAIN class, two-layer temporal CNN (80→32→32; kernel 5), and the same CNN plus one Transformer encoder (32 dimensions, four heads, feedforward 128, dropout .1, sinusoidal positions). Both use a 13-class head.
- Fixed seed 339, CPU/two threads, five epochs, Adam .001, batch eight, 256-frame nonoverlapping padded windows. Mask padding and uncertain labels from loss. TRAIN normalization is reused; no VALID fitting.
- Evaluate last epoch only: pooled and recording-macro accuracy, macro F1 over reference-present classes, confusion matrices. VALID is reused development evidence. No TEST access.
- Hypothesis: neural models may exceed majority class; failure is retained. Different parameter counts, single seed, short budget and uncertain score supervision prohibit architecture superiority claims.
- Results: pending execution.

### R008 diagnostic extension

Initial five-epoch recording-macro VALID accuracy: CNN .425281, hybrid .394514, majority .121011. Before drawing a model conclusion, add the existing cached pYIN pitch-class prediction under exactly the same weak-label boundary mask. This addition is post-result diagnostic, not preregistered model selection. Rerun the unchanged deterministic configurations and report all baselines. No parameters are tuned in response to VALID.

### R008 outcome and decision — 2026-10-21

Executed `scripts/train_score_pilot.py`, then repeated unchanged training while adding the matched pYIN diagnostic. Neural metrics reproduced exactly. Both explanatory notebooks 03 and 04 executed successfully; embedded plots were visually inspected and compacted below 1 MB per notebook.

| Method | TRAIN recording-macro accuracy | VALID recording-macro accuracy |
|---|---:|---:|
| Majority TRAIN class | 0.160161 | 0.121011 |
| Existing pYIN, pitch class | 0.585316 | 0.485526 |
| CNN, 18,413 parameters | 0.584813 | 0.425281 |
| CNN–Transformer, 31,117 parameters | 0.601113 | 0.394514 |

- CNN TRAIN cross entropy fell from 2.409170 to 1.432791; hybrid from 2.398661 to 1.553604. Both learn this small training target but neither exceeds pYIN on development VALID.
- Decision D015: preserve pYIN as the working baseline. Keep learned models as experimental weak-score prototypes, not a validated improvement or completed multi-task architecture.
- All 13 regression checks passed. The independent pitch check, target semantics and masking checks are included. A local Jupyter TCP transport warning was emitted; notebook execution completed without cell errors.
- Generated tensors/checkpoints remain ignored; original recordings/annotations were read only. No TEST feature generation or evaluation.
- Single seed, short training, reused VALID, unequal parameter counts and unverified targets limit inference. No claim of architecture superiority or full-corpus accuracy.
- Source/results: `review.py`, `targets.py`; scripts `review_annotations.py`, `build_score_dataset.py`, `train_score_pilot.py`; JSON manifests and results; executed notebooks 03/04; `target_policy.md`.
- Commit association: this milestone is committed with message `feat: add weak-score research pilot`; use `git log --oneline -- scripts/train_score_pilot.py` to resolve its hash.

## Current status after R008

Completed: independent TRAIN pitch comparison; automatic annotation flags; explicit weak-score target policy; reproducible pilot features; executed CNN/hybrid feasibility comparison and matched pYIN/majority baselines.

Pending: human listening/reference verification; defensible performed-pitch labels; expanded repeated-seed experiments; CFP features; beat/key/phrase multi-task supervision; learned harmony representations. The earlier statement that no CNN–Transformer had been trained describes the pre-R008 state and is superseded by this limited pilot.

Immediate next step: review the eight flagged TRAIN recordings with human listening, preserving original annotations and documenting reviewer evidence. A larger training run should follow a justified target decision, not merely seek a higher score on the reused validation pilot.

## R009 — listening review packet (preregistered 2026-10-21)

- Objective: make all eight shared-octave TRAIN cases reviewable by a human, with source audio, original MIDI tones, pitch plots and explicit decision provenance.
- Configuration: preserve original audio/MIDI; synthesize sine reference tones at original MIDI and fixed ±12 semitone hypotheses. These are comparison aids, never corrected annotations. Verify original source hashes against R006.
- Diagnostic: compare both cached estimators to all three fixed hypotheses on common voiced frames more than 100 ms inside a MIDI note. Report all candidates; do not select corrections automatically.
- Human review: listen to original first, then references; record reviewer, date, confidence and timestamped evidence. Uncertain cases remain unresolved. No model expansion before reviewed target decisions.
- Hypothesis: the packet makes discrepancy review reproducible; automatic pitch similarity is insufficient to establish performed pitch.
- Results: pending execution. No claim that the assistant has listened to audio.

### R009 diagnostic extension and implementation correction

The initial fixed ±12 comparison had zero agreement for two cases. R006 median errors for M02_0302_0002_1 and M05_0263_0001_2 are −23.7 and −24.0 semitones. Add a −24-semitone hypothesis for all eight cases, retaining the original three comparisons. This is a post-result extension, not an annotation correction. Initial validator execution exposed a missing closing bracket; corrected before validation or commit.

### R009 outcome and decision — 2026-10-21

- Executed `scripts/build_listening_review.py`, `scripts/check_listening_review.py` and notebook `05_listening_review.ipynb`.
- Generated eight local playback sections, eight waveform/pitch plots and 32 synthesized comparison references. Original audio/MIDI hashes match R006; derived media remain ignored by Git.
- Six cases have their highest candidate agreement at −12 semitones, two at −24. pYIN candidate agreement ranges from 0.109375 to 0.978862 on common voiced interior frames. Original-score agreement is zero on those selected frames for both estimators. This conditional diagnostic is not corpus accuracy or verified ground truth.
- A representative two-octave plot was visually inspected. Human listening was not performed by the assistant. Status: 0 reviewed, 8 pending; no annotation changes authorized.
- All 17 tests pass, including duplicate membership, pending status, missing reviewer provenance and unresolved-decision safeguards. Notebook executed without cell errors (same local Jupyter transport warning as earlier).
- Decision D016: retain original labels and the current baseline; do not apply blanket octave corrections or expand training from these diagnostics. Obtain timestamped human listening evidence using `docs/vishwa/listening_review.md` and `listening_review_decisions.json`.
- No new dependencies, TEST evaluation or teammate changes. Review-form completeness does not certify musical correctness; a separate target-policy decision is required even after all forms are complete.
- Commit association: `feat: add TRAIN listening review packet`; resolve with `git log --oneline -- scripts/build_listening_review.py`.

## R010 — simultaneous listening playback — 2026-10-22

- User requested one-click original/MIDI-0 playback to compare timing.
- Added a separate comparison player and play/stop buttons to all eight cases. Render one stereo WAV per case: original left, synthesized unchanged MIDI right, sharing source sample rate, start and sample count. Preserve relative levels with shared attenuation only if needed. The common file avoids drift between independently started players.
- Headphones permit separate listening to each side; seeking the comparison player moves both together. Starting any player pauses the others. Original individual players remain available.
- Regenerated the ignored HTML/media through `scripts/build_listening_review.py`. Verified eight comparison files have two channels and exactly match source sample rate/frame count; verified eight paired-play controls and clean diff formatting. Browser interaction and subjective listening have not been verified by the assistant.
- Existing decision records are preserved. No pitch/annotation correction or training expansion. Automatic evidence values are unchanged; generator provenance hash updated.
- Local milestone: `feat: add synchronized listening comparison`.

## R011 — adjustable MIDI listening delay — 2026-10-22

- User reported MIDI 0 sounding closest but quieter, with a possible overall lead of half a second or less on M01_0311_0002_1. These are subjective observations; exact offset and pitch remain unverified and no completed review is inferred.
- Added per-case 0–1000 ms MIDI-delay sliders in 10 ms steps and reset buttons. Positive values move the right-channel reference later while left-channel source timing remains fixed. Changing delay stops playback; use Play both to restart or seek in the rebuilt comparison.
- Render shifted PCM16 stereo in browser memory from embedded comparison data, permitting direct local HTML use without a web server. Extend the comparison tail instead of truncating delayed notes. No source, label or review-file modifications.
- Executed generator and verified all eight sliders. Executed the extracted JavaScript renderer with an actual comparison WAV at zero/500 ms: original samples unchanged, reference shifted by the exact sample count, leading silence and full ending preserved. Browser interaction is not yet independently verified.
- Decision: slider values are temporary listening hypotheses, not measured corrections. Compare several note transitions before documenting an offset estimate.
- Local milestone: `feat: add MIDI delay listening slider`.

## R012 — preliminary user listening observations — 2026-10-22

- User reported best positive MIDI delays for all eight flagged TRAIN cases: M01_0311_0002_1 220 ms; M01_0048_0001_2 240 ms; M02_0312_0001_2_D 740 ms; M02_0302_0002_1 800 ms; M04_0070_0001_1_D 170 ms; M04_0370_0002_2_D 160 ms; M05_0263_0001_2 220 ms; M05_0016_0002_1_D 130 ms.
- Preserved in `listening_review_observations.json`, separate from the full review decision form. These are user-reported alignment impressions, not algorithmically measured offsets or verified label corrections. No confidence, octave judgment or complete-reference-listening confirmation is inferred for the remaining cases.
- First case: user favored unchanged MIDI pitch despite automatic octave flag; 220 ms improved beginning/checkpoints/end after comparing 200 and 250 ms. That disagreement remains unresolved.
- Decision: no blanket delay or octave correction. Next compare unshifted versus user-delayed TRAIN annotations with targeted acoustic evidence, including whether note boundaries improve throughout and whether octave discrepancy persists. Keep original labels and TEST untouched.
- Checked observation membership against the eight evidence cases and delay range against player controls. No model training performed.

## R013 — merged project integration review — 2026-10-22

Inspected merge 67c84e0 against 37f5d54, component READMEs, core input/output code, saved Pasan notebook results and root installed-package metadata. Recorded findings in `merged_project_review.md`. Vishwa paths unchanged by the merge; root dependency requirements changed. Several teammate packages and expected generated/raw-data paths are absent locally. No sync, downloads, model inference or teammate tests executed. Historical metrics are attributed to saved outputs, not fresh runs.

Decision: do not assume an integrated application from merged files. Resolve architecture discrepancy (proposed final validator versus implemented BPDR input component), select a downstream predictor/checkpoint, and map note fields while establishing required key and complete-bar beat metadata. Continue HumTrans timing/label work without automatic transposition. User's existing environment preserved. Local milestone: `docs: map merged components and integration gaps`.

## 2026-10-23 — R014: portable checkpoints and live PP1 integration

**Objective:** restore Vishwa's skipped stage in the merged team demonstration and
make the existing learned prototypes available on teammates' machines.

**Repository:** fetched and fast-forwarded the clean feature branch from `80fb89b`
to `4f60115`. Inspected the team's standard-library HTTP/HTML app, live BPDR export,
sixteen-bar Pasan input contract and Rahul's fixed-major mode. Kept their model
source and selected checkpoints unchanged.

**Checkpoint export:** copied the existing CNN (18,413 parameters) and hybrid
(31,117 parameters), plus TRAIN normalization, to
`experiments/vishwa/checkpoints/weak_score_pilot/`. Both checkpoint digests match
the original `score_model_pilot.json` report. The bundle is approximately 211 KB.
No new training or accuracy evaluation occurred. Historical VALID recording-macro
score-class accuracies remain 42.53% / 39.45% on the fixed 20 TRAIN / 10 VALID pilot.
Added a standalone model definition, hash manifest and reproducible setup.

**Decision:** run pYIN on BPDR's accepted waveform to supply octave-aware note
intervals; run both trained score-class models on the same live Mel features for
visible experimental predictions. Do not fabricate octave or note boundaries
from pitch-class outputs. Replace every saved note inside the audible crop,
including rests; preserve clipped saved notes only outside it to satisfy the
existing sixteen-bar context. Saved beats/key and fixed-major mode remain labelled.

**Execution:** `uv run --no-sync python -B -m unittest discover -s pp1_demo/tests -v`
passed 12 tests. `uv run --no-sync python -B pp1_demo/tools/rehearse.py --track
track_0000382` completed all four components on a real 19.790-second clip: 57 live
notes, both learned checkpoints executed, 42 excerpt-aligned basic and refined
chords, and generated melody/chord/audio artifacts. Cold BPDR initialization made
that first run 107.574 seconds; this is not representative warm latency.

**Browser/API verification:** the browser run completed in 16.745 seconds including
presentation holds; Vishwa took 1.042 seconds. HTTP integration check completed in
16.024 seconds, confirmed accepted-audio hash equality, live notes in the downstream
context, both classifiers, all four stages, artifact downloads and path boundaries.
Verified the visible melody piano roll, F0, Mel spectrogram, checkpoint probability
maps and research disclosures. These are functional checks, not musical accuracy.

**Portability:** uv setup and standalone audio-check commands added; an app-local
BPDR runtime was needed because the legacy interpreter was absent on this Mac.
No raw recordings, dataset shards, generated media or environments are committed.
The existing library still requires local Jamendo extraction assets. macOS was
executed; Windows setup paths were implemented but not tested on Windows hardware.

**Next research work:** finish performed-pitch/timing annotation review, then improve
and evaluate the learned model. This integration does not close those research gaps.

Second real-recording check: `track_0000383` completed in 11.986 s without display holds, with 41 live notes and 41 excerpt chord predictions. Both learned checkpoints ran; the minor source key remained saved metadata and Rahul mode remained the explicit major constant.

Checkpoint export commit: `56d2eb5` (`feat: export portable Vishwa melody checkpoints`).
