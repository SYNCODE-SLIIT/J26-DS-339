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
| A: exploration | Completed initial audit | Executed notebook 01; independent alignment verification in progress |
| B: preprocessing | Partial | Resampling, optional trim with offsets, Mel, pYIN; CFP pending |
| C: baseline | Working smoke test | Quantitative evaluation pending |
| D: CNN–Transformer | Pending | No training or performance claim |
| E: harmony representation | Partial | Predicted duration-weighted pitch-class histogram only |
| F: controlled comparisons | In progress | Audit → training pilot → frozen validation pilot |

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

## R001 — independent annotation audit (in progress)

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
