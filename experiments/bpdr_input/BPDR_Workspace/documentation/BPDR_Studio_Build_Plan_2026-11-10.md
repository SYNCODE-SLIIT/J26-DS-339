# BPDR Vocal Rescue Studio: build plan for the next phase (written 10 Nov 2026)

**Purpose of this file.** A later session (human or Claude) should be able to read only this file and build the next features correctly. It records what exists, what was measured, what was decided and why, the exact specs of each feature, how to test it, and what must not be claimed. **Nothing in sections 5-8 is built yet.** Status labels: **[V]** verified by running or reading on 10 Nov 2026; **[R]** taken from project records, not re-derived; **[P]** proposed, untested; **[W]** web source, not independently verified.

Related files (read order if needed): `documentation/BPDR_Redesign_Plan_2026-11-10.md` (strategy, impact table, section 10 = what was built), `documentation/Continuation_Method_and_January_Plan_2026-11-07.md` and `Next_Experiment_Protocol_2026-11-07.md` (frozen hygiene rules and gates, still in force), the proposal `documentation/IT23141506_J26_DS_339.pdf`, `documentation/CONTINUATION_STATE.md` (older, partly stale).

---

## 1. The component and its job

Owner: Praveen Liyanage (IT23141506), project J26-DS-339, SLIIT final-year research. Component = **vocal-input preparation** for a genre-aware vocal-melody harmonisation system: take a short (<=20 s, one voice) recording, decide whether it is usable, optionally prepare it, and hand audio plus evidence to the next component. Proposal title: *Vocal Input Validation and Minimal Musical Repair using Boundary-Pitch Disentangled Learning (BPDR)*.

Research hypothesis (proposal sec 1.3, 3): a compact Transformer trained with boundary-invariance and local counter-detuning losses plus supervised anchors repairs small pitch errors more stably under false-split/merge boundary errors than (a) an augmentation-only Transformer and (b) a capacity-matched frame-only Transformer, without more harm at matched coverage. A negative result is explicitly acceptable. Combining libraries/rules/a Transformer is not the claimed novelty.

Responsibility boundary (proposal Table 4 and `pp1_demo`): BPDR owns recording validity, noise/level preparation, acoustic evidence, bounded local relative repair, protection/rollback/review, accepted-WAV provenance. **Other owners:** Perera E V = final melody extraction (notes, durations, tempo, key); Perera P D = genre-conditioned harmonic planner (Pasan checkpoint); Jayakody J A S R = chord realisation/refinement (Rahul checkpoint). Vishwa's code in this repo (`src/melody_understanding`) is the shared extractor. BPDR must **not** output notes, key, beats, tempo or chords, and must not edit teammates' code.

Public hand-off statuses (fixed by proposal): `ACCEPT_UNCHANGED`, `ACCEPT_PREPARED`, `RERECORD_REQUIRED`, `TECHNICAL_FAILURE`. Rejected statuses carry no accepted-audio path.

Proposal scope limits that matter for features: <=20 s, one voice, no melody invented from ordinary speech, no clipping recovery, no multi-voice separation, speech-like delivery alone must not reject a recording, noise reduction only with a reliable voice-free noise region, original always available, every edit logged.

Deadline: evaluated demo ready by **31 Dec 2026**, presentation January 2027 (exact date unknown). Mandatory decision point **11 Dec**: keep a passing method or report the failed hypothesis.

## 2. What exists now (all **[V]**, 10 Nov 2026; nothing committed to git)

Branch `bpdr-experiment`. Untracked new files (the whole Studio):

| Path (under `experiments/bpdr_input/BPDR_Workspace/`) | Role |
|---|---|
| `source/bpdr_input/studio/evidence.py` | decode (mono without phase cancellation, clipping measured on decoded samples before resampling), `pyin_track`, `pesto_track` (model `mir-1k_g7`, threshold 0.5), `notes_from_track` (mirrors the shared extractor's nearest-semitone rule), `quality_report`, `normalize`/`normalization_gain` |
| `.../studio/clean.py` | `spectral_gate` (noise profile = quietest 15% of frames; a local policy, NOT the proposal's voice-free-region rule) and `choose` (do-no-harm selector: accept cleaned only if PESTO confidence rises >=0.01, median pitch shift <=25 c, voicing loss <=5%) |
| `.../studio/pipeline.py` | `prepare(path, outdir, intent, with_pyin)`: validate -> PESTO at normalised level -> cleaned candidate -> choose -> statuses, located guidance, `prepared.wav`, `sidecar.json`, synthesized "heard" WAVs, listening-only cleaned preview |
| `.../studio/consumer.py` | calls the shared extractor `analyze_vocal(path, trim=False)` exactly as `pp1_demo/adapters/vishwa.py` does |
| `.../studio/speech.py` | `plateau_fraction` speech detector (threshold 0.22, preliminary), `speech_to_melody` (major/pentatonic only) |
| `.../studio/chords.py` | read-only call into `pp1_demo.adapters.pasan` and `rahul`; tiles the phrase to Pasan's fixed 16-bar window |
| `.../studio/synth.py`, `plots.py`, `examples.py`, `scoring.py` | additive synthesis; piano-roll/chord figures; 8 disclosed demo inputs; note-F1 via the workspace's isolated mir_eval 0.8.2 |
| `source/bpdr_input/application/studio_app.py` | Streamlit UI (tabs: Prepare a recording; Research evidence) |
| `Run-Studio.ps1` | launcher, port 8504 (a copy was left running during the session) |
| `tests/test_studio.py` | 11 unittest tests, pass (`PYTHONPATH=source ../.venv/Scripts/python.exe -m unittest tests.test_studio`) |
| `tools/studio_evidence.py` | reproduces the PESTO-vs-pYIN table, singer-level bootstrap -> `results/studio/evidence/pesto_vs_pyin.json` (~8 min) |
| `tools/studio_validate.py` | false-rejection count + F1 on annotated examples -> `results/studio/evidence/validation.json` (~10 min) |
| `tools/probe_tuning_offset.py` | the tuning-offset probe (needs a cache build step; low value, kept for the record) |

Environment facts: Windows 11; use `experiments/bpdr_input/.venv/Scripts/python.exe` (torch 2.14 CPU, librosa 1.0, pesto 2.0.1, streamlit 1.66, matplotlib, parselmouth 0.4.7). No pytest, no plotly, no pretty_midi. `mir_eval` is NOT in the venv: use the workspace copy at `results/transformer_usability/evaluation_dependencies/mir_eval_0_8_2` (see `scoring.py`). `uv pip install` hung once; `python -m pip install --no-deps --target <dir> <pkg>` worked. `/results/*` is git-ignored in this workspace. `analyze_vocal` returns a `(result, features)` tuple. PESTO's timestamps are milliseconds. Pasan needs exactly 16 bars of 4/4; the chord stage is fixed to major mode; Rahul's genre vocabulary has 62 genres (pop, jazz, soul, blues, rock, folk are in it; **gospel and lofi are not**).

## 3. What was measured and what it means (**[V]** unless marked)

Setting for all tables: 21 training-role Vocadito sources (IDs 1,3,6,8,10,13,14,18,20,25-29,31,32,35-39; 14 singers), first 8 s, mean F1 over annotators A1/A2 (onset 50 ms, pitch 50 c, mir_eval), one digital noise clip (ESC-50 vacuum cleaner; labels of the four workspace clips verified against `datasets/environmental_noise/esc50.csv`: rain, breathing, vacuum_cleaner, clock_tick) at 10/0 dB, bootstrap over singers, development evidence only.

**3.1 PESTO evidence + level normalisation scores far above the shared pYIN; cleanup adds nothing. (Cause not yet isolated; see the step-0 control below.)**

| Condition | pYIN (today's shared extractor) | PESTO at recorded level | PESTO + peak normalisation | + always-on cleanup | Studio choice |
|---|---|---|---|---|---|
| clean | 0.288 | 0.435 | 0.492 (+0.204, CI 0.116-0.247) | 0.490 | 0.492 |
| 10 dB | 0.273 | 0.501 | 0.501 (+0.229, 0.134-0.291) | 0.501 | 0.501 |
| 0 dB | 0.276 | 0.483 | 0.481 (+0.206, 0.118-0.248) | 0.488 | 0.481 |

- Level normalisation alone lifted PESTO clean F1 0.435 to 0.492 (source peaks 0.07-1.01). An earlier table that credited "cleaning" for the clean gain was a normalisation confound; fixed.
- The cleaned candidate was chosen in 0 of 21 files per condition. Prepared audio therefore still sounds noisy.
- **Untested confound (step 0, run before any hand-off request):** the shared extractor's pYIN uses a 4096-sample frame (about 186 ms at 22,050 Hz) while F1 tolerates 50 ms onsets. Part or most of the gap may come from frame/hop settings or voicing rules, not from PESTO itself. Control: rerun the same 21 files with pYIN at a shorter frame (e.g. 1024-2048) and report onset-only F1 as well as onset+pitch F1 for both trackers. If a shorter frame closes most of the gap, the ask to Vishwa is a parameter change, not a sidecar, and the claim wording must change. Do not call the tracker "the biggest lever" until this is done.
- **Critical caveat:** the shared extractor runs pYIN on the WAV it is handed. The prepared WAV is the original at a new level. So the +0.20 exists only if the extractor reads BPDR's pitch-evidence sidecar (**proposed, needs Vishwa**). The "shared extractor on prepared WAV" F1 equals "today" in every example.

**3.2 Earlier project evidence (read 10 Nov, **[R]** for the study, **[V]** for the numbers I recomputed):** neural noise removal (DeepFilterNet3) on the shared extractor, singer-macro note-F1 change: clean -0.010, 10 dB -0.009, 0 dB +0.003 (95% CI -0.017..+0.024); per-case >0.02 changes 0 dB 32 better/35 worse of 84, 10 dB 26/40 of 84, clean 2/4 of 21. A perfect original-or-cleaned chooser would add only +0.033 / +0.026 / +0.003. Baseline extractor F1 on clean audio is only ~0.29, so the extractor, not the audio, limits the current pipeline.

**3.3 Global tuning offset is not a lever [V, probe]:** shifting f0 by 0/25/50 c (truth shifted with it): raw F1 0.303-0.306 flat; blind re-centring ~0.313 (within noise). Do not build tuning normalisation as a headline; at most report the tuning centre as metadata.

**3.4 False rejections [V]:** 96 sung/hummed cases (21 Vocadito x clean/rain/vacuum/ticks = 84, plus 12 HumTrans training hums from `datasets/humtrans/subset/train/*/original.wav`): 95 ACCEPT_UNCHANGED, 1 ACCEPT_PREPARED, **0 rejected**. Speech detector: 3 speech clips plateau 0.04-0.16, 21 sung files 0.29-0.89 (calibrated before normalisation was added; recheck, see the speech-detector item in section 5).

**3.5 Annotated demo examples [V]** (note F1: today / shared extractor on prepared WAV / proposed PESTO evidence): rain 0 dB 0.261/0.261/0.358; vacuum 0 dB 0.136/0.136/0.333; clock ticks 10 dB 0.252/0.252/0.306; **clean expressive hum (vocadito_13) 0.302/0.302/0.201, PESTO is worse here; keep it in the demo and say so.**

**3.6 Breakdown by language and pitch range (descriptive, tiny groups, not a finding) [V]:** PESTO > pYIN in every language group (English 8 files 0.49 vs 0.31; French 7: 0.47 vs 0.29; Tagalog 3: 0.65 vs 0.32; Mandarin 2: 0.41 vs 0.25; Hawaiian+English 1: 0.41 vs 0.10, clean). High-pitch tercile is lowest for both trackers (PESTO 0.43 vs 0.55 mid). Weakest PESTO files: 13, 39 (clean), 25. PESTO voiced ratio was only 0.03-0.19 on files 1, 3, 8 (unexplained; investigate in F1a).

**3.7 BPDR research status [R, receipts read V; arm table V]:** the corrected balanced-anchor three-arm run (12 checkpoints, 4,482 arrays) has an independent replay receipt; `advance_to_delivery = false`. First-pass singer-macro correction error BPDR 21.39 / augmentation-only 23.25 / frame-only 24.91 cents against ~24.8 for no correction; BPDR fails the raw gate in 11 of 17 scoreable groups; BPDR jitter boundary sensitivity is worse (0.258 vs 0.109). Fixed-batch memorisation with the balanced anchor works (positive error 22.7 -> 3.06 c, signed response 0.84), so the earlier collapse was a loss-balance problem (positive targets carried 1/6 of anchor weight; ~170:1 gradient imbalance). **Read-only check of that run's `results.json` [V] (17 scoreable groups; the split/merge/jitter aggregates are identical to many decimals, which also deserves a look): signed response median / max: augmentation-only 0.0000 / 0.663; frame-only 0.0000 / 0.000 (exactly zero); BPDR 0.114 / 0.685. Positive error means 23.25 / 24.91 / 21.39 c.** So the two controls are still largely collapsed to "do nothing" in this run, and "BPDR is lowest" may reflect control collapse, not a loss-function benefit. **This is a gate: before the three-seed learning curve, verify that the controls fit the same fixed training batch as BPDR (positive signed response near 1 and preserved zero targets) under the shared balanced anchor; if they do not, repair the control fit (a declared, matched change) before interpreting any arm ordering.** Architecture is not the bottleneck: Transformer 23, GRU 18, TCN 19, ExtraTrees 20 verified repairs of 200. Short-note repair (V71): 22 helped, 89 harmed of 445. Exact-reference repair works (24.9 -> 8.6 c) but needs a reference; realistic nearest-semitone reference falsely flags ~33% of natural spans.

**3.8 Novelty check (10 Nov, **[W]**):** BERT-APC ([arXiv 2511.20006](https://arxiv.org/abs/2511.20006)) tests shifted boundaries only; the 2026 reference-free preprint ([arXiv 2610.11524](https://arxiv.org/html/2610.11524v1)) uses SOME-extracted notes, assumes the note sequence is correct and fixes onsets/durations. Neither tests false splits/merges. This is a gap statement from two papers, not a priority claim.

## 4. Decisions already made (do not re-litigate without new evidence)

1. The Transformer is kept as the **research arm** (three matched arms, BPDR loss = novelty). Stop architecture, loss-weight and activity-gate sweeps. The balanced anchor stays as the one declared correction.
2. The Studio's measured value = (candidate) better pitch evidence + level normalisation + restraint (do no harm) + honest feedback; the cause of the F1 gap is to be isolated by the step-0 control. Heavy audio editing is not the lead.
3. Cleanup is a supporting, optional candidate (listening-only preview shown), never the headline.
4. Speech-to-melody is built as an **optional creative mode**, labelled "not repair, outside the research claim". The owner said no supervisor approval is needed for it; still keep the label.
5. Notes/key/tempo/chords stay with their owners. The demo's note lists use the shared extractor's nearest-semitone rule only to show effects; chords come from Pasan then Rahul via read-only adapter calls; key and tempo are user-chosen.
6. New work goes in the new `studio` package; the frozen review demo (`review_demo_app.py`, `repair/`) is untouched.
7. Reserved-data rules from the 7 Nov protocol stay in force. **Disclosure to keep in the evidence log:** during the first tuning probe (10 Nov) audio and A1 annotations of Vocadito IDs 2,4,5,7,9,11,12 (not among the 21 training-role sources) were read; nothing trained on them, results discarded, reported probe uses training-role IDs only. The owner should decide whether those files stay "unopened".

## 5. Feature specs (build in this order; each section is self-contained)

General rules for every feature: put code in `source/bpdr_input/studio/`, UI in `studio_app.py` (or a new tab), tests in `tests/test_studio_<feature>.py` (unittest, no pytest), evidence tools in `tools/`, outputs in `results/studio/`. Use only the 21 training-role Vocadito sources and the 12 HumTrans training hums (verified in `datasets/humtrans/intake_declaration.json` `person_roles`: F01-F05 and M01 = train, M02/M03 = development, M04/M05 = reserved_unseen; the 12 hums come from the six train singers) for development evidence; no reserved sources. Every number shown in the UI must be read from a results file or computed live and labelled. Keep the four statuses.

### F1. Voice profile and fairness report  (priority 1)

Problem and evidence: per-singer error varies widely (3-5% vs 11-23%, [Viitaniemi 2003](https://www.ee.columbia.edu/~dpwe/papers/ViitKE03-melodies.pdf) **[W]**); per-user training gave a 20-30% error cut ([Microsoft Research](https://www.microsoft.com/en-us/research/?p=326648) **[W]**); trackers are domain-specific with no combined benchmark across voice types ([2025 pitch-contour paper](https://arxiv.org/html/2503.19161v1), [breathiness study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8659512/) **[W]**).

Parts:
- **F1a diagnostic first (no UI).** Why PESTO voiced ratio is 0.03-0.19 on Vocadito 1, 3, 8. Check amount of silence/instrument in those files, f0 range vs PESTO model range, confidence histogram. Output `results/studio/evidence/pesto_low_voicing.json` + a short note. Decide whether a lower per-voice threshold or a different checkpoint is justified.
- **F1b fairness report tool** `tools/studio_fairness.py`: from `pesto_vs_pyin.json` per-file scores and `datasets/vocadito/vocadito_metadata.csv` (columns `track_id,singer_id,average_pitch,language`): F1 by language, pitch tercile, singer; counts; worst-k files. Descriptive only; print n per group; no significance claims. Add a "Who does it work for?" panel to the evidence tab.
- **F1c voice profile.** `studio/profile.py`: from a 5-8 s calibration hum compute register (5/50/95th percentile f0 in MIDI), confidence distribution, noise floor, tuning-centre (metadata only). UI: optional "Calibrate my voice" step; the profile is stored only in session state (privacy). **Hypothesis to test, not assume:** a profile-chosen PESTO voicing threshold beats fixed 0.5. Protocol: for each of the 21 files, calibrate on the first 5 s only, test on the remaining seconds; choose threshold from a fixed rule declared before looking at F1 (e.g. confidence percentile); compare F1 to fixed 0.5 with singer-level paired bootstrap. If null, ship the profile as feedback only ("your register is X; this recording sits at the edge of the tracker's range"). A null result is acceptable and should be reported.
Acceptance: F1a note written; fairness tool reproducible from the JSON (unit test with a tiny fixture); profile unit tests (register computation on synthetic tones); threshold experiment run once, result recorded either way.
Risk: group sizes 1-8; never describe as fairness proof.

### F2. Recording coach and room check  (priority 2)

Problem: bad capture conditions degrade later conversion (vendor/practitioner guidance only, [ScoreCloud](https://scorecloud.com/learn/capture-musical-ideas-fast/) **[W]**; no study of idea-loss rates exists; do not claim one). Also the proposal's conservative noise reduction **requires a reliable voice-free noise region**, which a room check supplies, which makes it a research-aligned feature, not a toy.

Parts:
- **F2a Room check (MVP, no new dependency).** Step 0 in the UI: "Stay silent for 2 s". `studio/room.py`: from a 2 s clip estimate noise floor (dB), spectral flatness, tonal hum (50/60 Hz and harmonics via spectrum peaks), intermittent events (frame-energy variance). Return level, verdict (`good`/`noisy`/`too loud`) and a plain-language tip. Pass the noise-only clip into `clean.spectral_gate` as the profile when present (declared policy: voice-free reference), and record which policy was used in the sidecar. Compare the room-check profile against the quiet-frames profile on the 21-file noisy set (does it change F1 or the choose decision?). Report the result either way.
- **F2b Live level meter (optional).** Streamlit's `st.audio_input` is record-then-submit, so live feedback needs a small custom component using the browser Web Audio API (AnalyserNode) showing RMS, peak (clip warning at >= 0.98), and a rolling noise-floor estimate. Plain HTML/JS component folder, no build step. If bidirectional messaging is hard, show feedback only (no data returned).
- **F2c Recording tips from the report:** map each measured quantity to one action (move back, lower gain, find a quieter room, sing louder), with time ranges; reuse the located guidance in `pipeline.prepare`.
Acceptance: unit tests with synthetic hum, fan-noise, 50 Hz hum, clipping; the room check labels them correctly; no new dependency required for F2a. A pilot with real users is needed to claim the coach improves acceptance (see section 9).

### F3. Optional tap-the-rhythm input  (priority 3)

Problem: segmentation, not pitch tracking, dominates sung-melody transcription errors ([Viitaniemi 2003](https://www.ee.columbia.edu/~dpwe/papers/ViitKE03-melodies.pdf), [ISMIR 2014](https://archives.ismir.net/ismir2014/paper/000147.pdf) **[W]**, old); this attacks the same false split/merge problem the proposal targets and also helps users who cannot hold steady pitches (accessibility, see F6). It is an **optional extension with its own input contract**: boundary hints from the user are not "reference-free", so never feed them into the three-arm research comparison.

Parts:
- **F3a offline test first (no UI): simulated taps.** Generate taps from one annotator's note onsets with jitter sigma 40 ms, 10% missed, 5% spurious, and score against the OTHER annotator (taps from A2 scored against A1, and the reverse). Taps from A1 scored against A1 can only be reported as an upper bound, never as the go/no-go test, because the taps already contain the answer. Build `studio/tapseg.py`: given PESTO track + tap times, force boundaries at taps, suppress boundaries not within +-150 ms of any tap inside tapped regions. Compare onset-F1 and onset+pitch F1 with vs without taps under the same segmenter, on the 21 files, clean/0 dB. Declare the go/no-go rule before running (e.g. singer-level 95% lower bound of the cross-annotator onset-F1 gain above 0). **Label as simulated taps.** Report per-file and singer-level bootstrap.
- **F3b UI.** Record the hum first; then play it back while the user taps (button or space bar); tap times are read from the playing `<audio>` element's `currentTime` to avoid clock drift (a live tap during recording has no reliable sync to `st.audio_input`). Needs a bidirectional Streamlit component (`declare_component` with a plain HTML frontend). Show taps over the piano roll; let the user clear and retry.
Acceptance: F3a result recorded before any UI work; if simulated taps do not help, say so and stop F3b. Component contract tests with fixture tap arrays.
Risk: tapping adds user burden; user taps are noisy; never describe the result as better "intent recovery".

### F4. Voice-memo triage  (priority 4)

Problem: old phone voice memos pile up unused (practitioner claim only **[W]**). Demo value: drop a folder, get a table.

Parts:
- **Classes:** `sung_or_hummed`, `speech`, `instrument_or_music`, `noise_or_silence`, `mixed/unclear`. Start rule-based from existing features (plateau fraction, voiced ratio, spectral flatness, harmonic ratio, onset regularity); consider a pretrained audio tagger only after checking licence, size and offline availability **[P, unverified]**.
- **Labelled data needed:** sung = Vocadito training-role + HumTrans training hums; speech = LibriSpeech clips (librosa's `libri1-3` were used; get more, e.g. a small LibriSpeech dev subset from OpenSLR, CC BY 4.0 **[P: verify licence/size before downloading; ask the owner before any download]**); noise = ESC-50 (workspace has 4 clips; full set is CC BY-NC **[P: verify]**); instruments = needs a source (verify licence). Split by source/speaker before any threshold is chosen; report a confusion matrix with counts.
- **UI:** multi-file uploader -> table (name, class, confidence, status, reason) -> select rows -> run `prepare` -> download a zip of accepted WAVs + sidecars. Never alter the originals.
Acceptance: confusion matrix on a held-out set; zero accepted-audio paths for non-sung classes unless the user overrides; tests with fixtures. Honest caveat: the triage classifier is engineering support, not a research claim.

### F5. Two-take consensus  (priority 5)

Idea: the user records the phrase twice; the Studio aligns the two PESTO contours with DTW (subtract each take's median so key/transposition does not matter), marks where takes agree (within tolerance) and where they disagree, lets the user choose a take. It does **not** generate a "true" melody. Evidence plan **(unverified: no suitable data on disk yet)**: the 12 local HumTrans training hums are 12 different melody numbers, so they contain no repeated melodies to align, and HumTrans guide MIDI is not performed-pitch truth. Real repeated takes must be recorded (pilot participants, ethics approval) or a repeated-melody subset found and its licence checked. Then test whether cross-take agreement predicts extractor correctness (calibration curve: agreement bins vs note-F1). If agreement does not predict error, drop the feature. Fits the proposal's "singer's own judgement" validation and the 7 Nov plan's later "multiple takes" item.

### F6. Accessibility mode  (priority 6, mostly framing)

Problem: people with speech or motor impairments want voice-based music creation but voice alone is not reliable for everyone: [Boris, PoliMi](https://re.public.polimi.it/bitstream/11311/1197654/1/3464385.3464713%20%283%29.pdf), [single-switch composing](https://equalentry.com/composing-music-with-a-single-switch/), [Babbled Notes](https://brookehoward2008-droid.github.io/babbled-notes/docs/) **[W]**, small prototypes and studies. Build only a thin layer: one "How do you want to give me the melody?" menu (hum, sing, read aloud -> speech-to-melody, tap rhythm (F3)), keyboard-operable controls and text labels (proposal NFR), plain-language messages. Do not claim clinical or accessibility outcomes; present as a use case. A real accessibility claim needs users and ethics approval.

### Cross-cutting engineering items (do these alongside F1-F3)

- **Extractor hand-off contract with Vishwa:** versioned sidecar schema (`bpdr_studio_handoff/0.1` today), contract test that the user-selected prepared WAV's SHA-256 equals the file the extractor actually reads, and a decision on whether the extractor will read the sidecar (pitch track + confidence) or keep its own pYIN. Ask which extractor is primary (pYIN baseline vs the GAME adapter in `experiments/pasan/melody_extraction` and the demo's BPDR review workflow). Do not change `src/melody_understanding`.
- **Speech detector re-validation:** the 0.22 threshold used 3 speech clips and pre-normalisation tracks. Re-run on a larger speech set, with the normalised tracks the pipeline now uses, plus noisy speech, plus chanting/rap-like and very monotone humming (which can look speech-like). Keep the proposal rule: speech-like delivery alone must not reject; when unsure, ask the user instead of rejecting.
- **Evidence refresh:** re-run `tools/studio_evidence.py` and `tools/studio_validate.py` after any pipeline change; keep JSONs under `results/studio/evidence/`.
- **Add more noise types and levels to the evidence study** (rain, clock tick, breathing were only in demo examples; babble/street/wind if licensed data is available). Do not call results general until fresh singers/recordings are used.
- **Normalisation gain** is now logged in result/sidecar (the proposal requires gain logging). Keep it.
- **Tests:** extend `tests/test_studio.py` or new files; run `PYTHONPATH=source ../.venv/Scripts/python.exe -m unittest discover tests -p "test_studio*.py"`.

## 6. The BPDR research track (runs in parallel; do not let the Studio crowd it out)

From the redesign plan section 5 (**[P]**): learning curve over the number of training singers x three arms x three preselected seeds (only seed 339 has been used; the proposal promises three), singer-level paired bootstrap, harm at matched coverage (not error alone), declared before opening held data, models ~21k parameters so cost is small. Report short-note and stable-note regimes separately and do not present held notes as the rescue (held-note raw error ~12 c did not turn into delivered repairs: 23 of 200; the one earlier matched three-arm run tied 5.84/5.64/5.65 c). Pull the 707 construction-exclusion reasons by category before adding data; choose datasets by independent scoreable singers after the render audit, not hours of audio; the consented 8-adult pilot is the real data fix. Verify licence and singer counts before naming any new dataset (none verified; HumTrans singer count vs the 4 used and the role of the `vocalset_*` folders are still to be checked). Frozen gates are not to be weakened. 11 Dec decision point stands.

## 7. Suggested schedule (today 10 Nov; adapt to the real presentation date)

| Window | Work |
|---|---|
| 11-17 Nov | **Step 0 first:** the pYIN frame-size / onset-only control (section 3.1) and the control-arm fit check (section 3.7), both before any request to Vishwa or any learning-curve run; then ask Vishwa the extractor/sidecar questions; F1a diagnostic; F1b fairness tool; F2a room check + tests; list/verify datasets needed for F4 (ask before downloading) |
| 18-24 Nov | F1c voice profile + threshold experiment; F3a simulated-tap test (decide go/no-go on F3b); speech detector re-validation |
| 25 Nov-1 Dec | F2b live meter (optional); F3b tap UI if F3a is positive; start BPDR three-seed learning-curve runs |
| 2-11 Dec | Finish BPDR comparison; F4 triage if data licence/size allow; refresh evidence JSONs; **11 Dec decision on the research claim** |
| 12-20 Dec | F5 (only if time), F6 menu, UI polish, keyboard/mobile checks, panel script rehearsal |
| 21-31 Dec | Frozen evaluation, listening session with independent listeners, latency on demo hardware, evidence table, final package |

## 8. Panel demonstration script and claim discipline

Flow: (1) noisy sung phrase -> status, located feedback, before/after piano rolls with sound, F1 vs human annotators (say "controlled mix", one file, study average is on the evidence tab); (2) clean expressive hum left alone, and show the honest example where PESTO is worse; (3) clipped file and silence -> located re-record / technical failure, no accepted path; (4) reading aloud -> "this is speech", optional creative mode -> melody and chords from the team's models (say "composed, not repaired"); (5) evidence tab (UPDATED 12 Nov): the frame-size control, i.e. a 2048 frame gives +0.18 F1 over the shared 4096 frame (singer-level interval), PESTO adds only a modest extra, cleanup adds ~0, repair review-only, BPDR three-arm result as measured.

Say: "measured on 21 development files from 14 singers", "most of the gain comes from the extractor's analysis frame length; we recommend it to the extractor's owner", "the research hypothesis is tested honestly; a null result is a valid outcome". Do not say: the prepared audio is cleaner (it isn't, usually), speech is repaired, the Transformer improved the result (unproven), it works for all voices, it generalises to novice phone recordings (untested), any number about how often songwriters lose ideas.

## 9. Open questions that only the owner/people can answer

1. Is ethics approval in place and are the 8 pilot participants and 2 independent reviewers available? (needed for real novice recordings, tap/take features, any accessibility claim)
2. Will supervisors accept a headline of "do-no-harm preparation component + pitch-evidence hand-off + an honestly reported BPDR comparison (possibly null)"?
3. Will Vishwa agree to read the sidecar and which extractor is primary for January?
4. Exact presentation date; which of F1-F6 the owner wants first (recommendation: F1, F2, F3).
5. Do Vocadito IDs 2,4,5,7,9,11,12 stay "unopened" after the 10 Nov probe read them (section 4.7)?
6. Permission to download datasets (speech, noise, instruments) once licence/size are checked.

## 10. Verify before building (things that may have changed or were never checked)

- `git status` and that `studio/` still passes `unittest tests.test_studio` (11 tests).
- Teammate branches: Vishwa's remote branch had a 30 Oct "live melody analysis" commit I did not read; check whether it already consumes `accepted.wav` and whether the demo bridge still uses saved notes (`pp1_demo/orchestrator.py`).
- Pasan/Rahul checkpoints still load via `pp1_demo.adapters` (`chords.harmonize` smoke test on any phrase).
- Whether a copy of the Studio server is still running on port 8504 (stop it before restarting).
- Streamlit version behaviour for custom components and `st.audio_input` (v1.66 used).
- No listening review of speech melody or harmony has happened; the browser upload and microphone paths were never exercised in a real browser.

## 11. Known defects in the built Studio (fix or disclose before the panel)

- `examples.build()` writes the example WAVs once and reuses them, but its first run needs the LibriSpeech clip (`librosa.ex('libri1')`) cached or network access.
- `Run-Studio.ps1` was never executed; the Streamlit server was started directly with the same arguments (HTTP 200, health ok). Run it once.
- The harmony stage uses the PROPOSED pitch-evidence notes, not what the shared extractor would produce from the prepared WAV; the UI labels this, or change it to use the extractor's notes.
- Microphone and upload files accumulate in `results/studio/sessions/` with no deletion rule. The proposal requires a consent/deletion procedure and restricted storage; add retention/deletion before any participant data is used.
- Upload and microphone paths were never exercised in a real browser; no listening review of the speech melody or the harmony has happened.
- All Studio code, plan files, tests and tools are **untracked** in git, and `results/studio/` (evidence JSONs) is git-ignored by `/results/*`. Their survival depends on the local disk until the owner decides to commit (not done; ask first).
- A copy of the Studio server may still be running on port 8504.

## 12. Confirmation data (decide before tuning anything further)

Every Studio number so far comes from the same 21 Vocadito training-role files, reused across many looks (PESTO threshold, selector thresholds, speech threshold, examples). They are development evidence. Before the January evaluation: freeze all thresholds and the primary metric in a dated declaration, then open ONE confirmation set not used for tuning: ideally the pilot recordings (8 adults x 6 phrases, needs ethics) and/or HumTrans development singers M02/M03 with fresh noise clips, with singer-level intervals and every attempt counted. Reserved sources (HumTrans M04/M05, Vocadito files outside the 21) stay closed until then; check existing locks before access.

## 13. Sources used for the problem statement (all **[W]**, 10 Nov 2026)

- Viitaniemi, Klapuri, Eronen 2003, melody transcription: https://www.ee.columbia.edu/~dpwe/papers/ViitKE03-melodies.pdf
- ISMIR 2014 query-by-singing/humming evaluation: https://archives.ismir.net/ismir2014/paper/000147.pdf
- Microsoft Research, user-specific training for vocal melody transcription: https://www.microsoft.com/en-us/research/?p=326648
- Boris (PoliMi), speech-impaired music production agent: https://re.public.polimi.it/bitstream/11311/1197654/1/3464385.3464713%20%283%29.pdf
- Single-switch composing: https://equalentry.com/composing-music-with-a-single-switch/
- Babbled Notes (project page, self-reported): https://brookehoward2008-droid.github.io/babbled-notes/docs/
- Pitch tracking and breathiness: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8659512/ ; pitch contours across audio domains: https://arxiv.org/html/2503.19161v1 ; voice source features / atypical speech: https://eprints.whiterose.ac.uk/id/eprint/241369/2/ICASSP_2021__Voice_Source_Features.pdf
- Speech-to-singing (2020): https://arxiv.org/pdf/2002.06595
- Voice-memo capture (vendor guidance, weak): https://scorecloud.com/learn/capture-musical-ideas-fast/
- Gap check: BERT-APC https://arxiv.org/abs/2511.20006 ; reference-free correction preprint https://arxiv.org/html/2610.11524v1
Web searches returned little 2025-2026 material for the transcription problem; the strongest evidence is old. Re-search before citing in the thesis.

---

## 14. Declaration of the step-0 frame-size control (written 12 Nov 2026 BEFORE running; this is attempt 1)

Script: `tools/studio_frame_control.py` -> `results/studio/evidence/frame_control.json`. Same 21 training-role sources, first 8 s, clean / 10 dB / 0 dB (same ESC-50 clip), same nearest-semitone segmenter, hop fixed at 256, singer-level bootstrap seed 339. Shared-extractor files untouched; `evidence.pyin_track` gained optional `frame_length` and `fmin_note` arguments whose defaults reproduce today's extractor.
- **Why fmin is a second variable:** C1 (32.7 Hz) has a period of about 674 samples at 22,050 Hz; a short frame cannot hold such lags, so shortening the frame may silently change the searchable range. Hence arms: `pyin_4096_C1_today`, `pyin_4096_C2` (fmin alone), `pyin_2048_C2`, `pyin_1024_C2` (frame + fmin). Reference: PESTO + peak normalisation.
- **Metrics:** onset+pitch F1 (50 ms / 50 c, primary), onset-only F1 at 50 ms and at 100 ms (secondary; separates timing smear from pitch error). Mean of annotators A1/A2.
- **Decision rule, fixed now.** Gap G = F1(PESTO) - F1(pYIN today) on the primary metric, per condition. Let the best pYIN arm's gain over today be g. "Frame/fmin explains most of the gap" iff g >= 0.5 G in the clean condition AND the singer-level 95% interval of g excludes 0. Otherwise "the difference is not explained by frame size or fmin" (which still does not prove PESTO is the cause; other segmenter/voicing differences remain). Any outcome is reported. If the rule is met, the ask to Vishwa becomes a parameter change and the "tracker" claim wording changes; if not, the sidecar ask stands.
- Not allowed after seeing results: adding arms, changing hop, changing the rule. Additional arms count as a new, numbered attempt.

## 15. 12 Nov 2026: step-0 result, readiness verdict, additions

**15.1 Frame-size control result [V], attempt 1, rule from section 14 applied as written** (`results/studio/evidence/frame_control.json`; 21 files, 14 singers, onset+pitch F1 @50 ms/50 c, singer-level bootstrap):

| Arm | clean | 10 dB | 0 dB |
|---|---|---|---|
| pYIN 4096, fmin C1 (today's extractor) | 0.288 | 0.273 | 0.276 |
| pYIN 4096, fmin C2 (fmin alone) | 0.288 | 0.272 | 0.277 |
| pYIN 2048, fmin C2 | 0.467 (+0.180, CI 0.100-0.212) | 0.453 (+0.180) | 0.396 (+0.120) |
| pYIN 1024, fmin C2 | 0.454 (+0.166) | 0.426 (+0.154) | 0.360 (+0.084) |
| PESTO + normalisation | 0.492 (+0.204) | 0.501 (+0.229) | 0.481 (+0.206) |

- Rule met: best pYIN arm closes 0.180/0.204 = 88% of the clean gap, CI excludes 0. **Most of the "PESTO gain" is a frame-length effect, not a tracker effect.** fmin alone does nothing (so the two variables were separable). Onset-only F1 @100 ms is about 0.68-0.70 for every arm: the 4096 frame loses 50 ms-level timing, not coarse onsets.
- **Consequences.** (1) Do NOT say PESTO is a big lever; the honest lever is a shorter analysis frame, with PESTO a modest extra (+0.02 clean, +0.05 at 10 dB, +0.09 at 0 dB over pYIN-2048; the PESTO-minus-pYIN-2048 interval is NOT computed yet, compute from the per-file JSON before stating it). (2) The ask to Vishwa is now a **one-parameter change** (`frame_length` 4096 -> 2048 in `extract_features`; note it also sets the mel `n_fft`, so Vishwa must check the downstream mel use), not a sidecar. This is a recommendation from development data on the same 21 files with 4 arms: confirm on the confirmation set before it is called general. (3) The sidecar/PESTO path is demoted to optional. (4) Fix the Studio claims table, the evidence tab text and the panel script (section 8) accordingly. (5) Do not edit `src/melody_understanding`.
- Limits: hop fixed at 256, only 2048/1024 tried, one noise clip, same 21 files used for every look so far.

**15.2 Readiness verdict (12 Nov): NOT yet ready for the panel; ready for a supervised internal walkthrough.** Done and verified: Studio builds and its 11 tests pass; validation (0 of 96 sung/hummed cases rejected); located guidance; honest examples; BPDR three-arm result recorded with its gate; teammate check (no remote commits since 30 Oct; `origin/main` extractor unchanged, still pYIN 4096/hop 256). Blocking before a panel: (a) claims text still credits PESTO (see 15.1); (b) control-arm fit check (3.7) NOT run, so any BPDR-vs-control statement is blocked; (c) Studio upload/microphone paths and `Run-Studio.ps1` never exercised in a browser; (d) no listening review; (e) no retention/deletion rule for session audio; (f) nothing committed, evidence JSONs git-ignored; (g) no confirmation set opened (section 12).

**15.3 Additions to the plan (all inside the BPDR component; none touch teammates' code)**
1. **Re-centre the headline** on what the data now supports: a do-no-harm preparation component with located feedback, an evidence-based finding that the shared extractor's analysis frame limits note timing (+0.12 to +0.18 F1 in development data), and an honestly reported BPDR comparison. Hand the finding to Vishwa as a measured recommendation with the table above; they decide.
2. **Run the same frame control for the segmenter's other lever:** hop and min-duration at 2048 (new numbered attempt, declare first) and check pYIN 2048 on HumTrans training hums.
3. **Control-arm fit gate** (3.7) is the next compute step for the research track; look at `tools/` balanced-anchor fitting scripts, repair control fit as a declared matched change, then the three-seed learning curve.
4. **Studio defects to close before any panel:** run `Run-Studio.ps1` once, exercise upload in a browser, add session-audio retention/delete, replace PESTO-centric wording, keep the honest-worse example.
5. **Process (RETRACTED 12 Nov):** I earlier reported a second process running the frame-control script. `.venv/pyvenv.cfg` shows the venv's home is the Codex runtime, so that was the venv launcher's child process, not another agent. No action needed.

**15.4 Done later on 12 Nov [V]:** evidence tab now shows the frame-size control table and the honest summary; the piano-roll label for the PESTO row now reads "Illustration: finer pitch evidence (PESTO); a shorter pYIN frame gives most of the same gain". `Run-Studio.ps1` executed once on port 8510: health ok, HTTP 200 (closes the "never executed" defect). Headless UI test: no exceptions, frame table rendered. Notion logbook row added. Still open: control-arm fit check, real-browser upload/microphone test, listening review, session-audio retention rule, commit decision, section 8 demo script wording (PESTO is no longer the lead).


## 16. Transformer assessment and control-fit plateau diagnostic (12 Nov 2026)

**16.1 Is the Transformer usable / accurate? [V from saved results, nothing retrained]** No. Run `2026-11-07_balanced_short_objectives` (21,666-parameter Transformer, 4 source-held-out folds, 12 fits, 1,200 updates each): held-out positive-target error singer-macro BPDR 21.39 c, augmentation-only 23.25, frame-only 24.91, against about 24.8 with no correction; every arm fails the raw gate; `advance_to_delivery = false`. Earlier delivered repair: 23 verified of 200 attempts. BPDR removes about 3.4 of 24.8 cents (14%) and applies about 0.75 c false correction to originals.
- **The arm comparison is not interpretable [V from the 12 fit receipts].** The balanced-anchor loss stays at the zero-output value (about 0.375-0.47) for the whole run in 9 of 12 fits. Only augmentation-only fold 1 (0.211) and BPDR folds 1-2 (0.20, 0.21) clearly left it. frame-only never left it (held-out outputs about 1e-4 cents). So "BPDR is lowest" mostly counts which fits escaped a plateau, and the controls were never trained in practice.
- **Why the fixed-batch fit worked but the real run did not:** the balanced-anchor fixed-batch probe sat at 22.7 c for 100 updates and fit only by update 200, and it ran one repeated batch with the augmentation-only objective. Fresh batches of the real run did not reproduce that escape in 9 of 12 fits.
- **Other limits already on record:** input-only blind detection of a 25 c error is near chance (AUROC .32-.64); GRU/TCN/ExtraTrees give similar verified-repair counts (18-20 vs 23 of 200), so architecture is not the bottleneck; about 8 scoreable independent singers.

**16.2 Declared diagnostic (attempt 1, training batch only, no held data, no new checkpoint kept for evaluation).** Script `tools/probe_control_fit_plateau.py` -> `results/transformer_usability/2026-11-12_control_fit_plateau/results.json`. Fold 1, first frozen 36-query batch (`physical_trace.npy` row 0), seed 339, declared optimizer (AdamW lr 5e-4, wd 1e-3), balanced anchor, 600 updates (200 warm-up without response terms for BPDR, then response on), metrics logged every 25 updates, first update where positive error falls below 80% of its starting value ("escape update"). Three arms x two variants: `as_declared` and `gate_open` (activity-head bias set to +2.0 after `new_model()`, applied identically to all three arms; this revisits plan decision 1 only for the stated reason that the zero output is a saddle of a product of two small terms; no other change). Frozen files are not edited. **Decision rule fixed now:** `gate_open` is adopted for a re-run of the 12 fits only if all three arms fall below 80% of starting error within 600 updates on this batch AND known-zero output stays below 2 c; otherwise report that the plateau is not removed by this change and stop (no further variants without a new numbered attempt). Even if adopted, a held-out improvement is not promised: information, not optimisation, may be the limit.

**16.3 Plateau diagnostic result [V], attempt 1 (`results/transformer_usability/2026-11-12_control_fit_plateau/results.json`).** On the frozen fold-1 batch ALL arms fit even `as_declared` (escape update: augmentation-only 125, frame-only 150, BPDR 125; final positive error about 1 c, signed response 0.95-0.96). With the gate open all three escape at update 75 and end at 0.72-0.93 c, signed response 0.965-0.986, known-zero error 0.18-0.30 c. The declared adoption rule (all three arms below 80% of start within 600 updates AND zero error < 2 c) is met. **Caution:** because `as_declared` also fits a repeated batch, this probe does not explain why 9 of 12 real fits (fresh batches) stayed at the zero output. A repeated batch can be memorised; fresh batches require generalisable signal. Only the real run can show whether opening the gate helps.

**16.4 Declaration of attempt 2: rerun of the 12 matched fits with the gate open (written 12 Nov 2026 BEFORE running).** Tool `tools/compare_gate_open_objectives.py` imports the frozen comparison unchanged, redirects its output to `results/transformer_usability/2026-11-12_gate_open_short_objectives/`, and sets the activity-head bias to +2.0 after `new_model()` in every arm. Same folds, traces, seed 339, 1,200 updates, 200 warm-up updates, balanced anchor, gates and selection rule; no sweep; attempt 1 (historical run) is kept. Reported regardless of outcome: (a) fit health = number of the 12 fits whose final logged anchor is below 0.3 (zero-output value is 0.375-0.47); (b) the frozen gates and `advance_to_delivery`; (c) positive error per arm against about 24.8 c no-correction, original-view false correction in cents. **Interpretation rule:** the arm ordering is interpretable only if at least 11 of 12 fits leave the zero output. If fewer, report that the controls still did not train and that no arm ordering is claimed. A held-out improvement is not promised. No claim of "BPDR beat the controls" unless fit health passes AND the frozen selection rule passes.

**16.5 Attempt 2 status (12 Nov 2026): INCOMPLETE, stopped by the system for low memory, not restarted.** 4 of 12 fits finished (the four augmentation-only folds), then the run was killed during fold 1 frame-only. Nothing was interpreted from a partial run. Finished fits are bound by receipts, so the frozen tool can resume them. Restart only when the machine has free memory (close other programs first), using `python tools/compare_gate_open_objectives.py --mode fit` from `tools/` with `PYTHONPATH=../source`, then `--mode verify`. The 11-of-12 fit-health rule in 16.4 still governs interpretation.

**16.6 Resume note (12 Nov 2026).** The resumed run failed an exact-equality check on the frame-only duplicate views (difference 4.7e-10, about 1 float32 ULP; the original 7-8 Nov run hit the same roundoff). `compare_gate_open_objectives.py` now applies the same documented change as `resume_balanced_short_objectives.py`: duplicate-view OUTPUT may differ by at most 4 ULPs; checkpoints, traces, gates and the saved-prediction replay stay exact. The 4 finished augmentation-only fits are reused. Nothing about the fits or gates changed.

**16.7 Second memory stop (12 Nov 2026).** The resumed run was stopped by Claude Code again for low system memory; it was not restarted. Completed fits are kept as receipts. Resume only on request with free memory (the same command as 16.5).

**16.8 Attempt 2 outcome on fit health [V, `fit_receipt.json` histories; 13 Nov 2026].** The run was stopped by low memory a third time with 11 of 12 fits saved (only fold 4 BPDR missing); it was not restarted. Fit health under the rule in 16.4 (final anchor < 0.3; zero-output value 0.375-0.47): **6 of 11 saved fits left the zero output** (original run: 3 of 12). By arm: BPDR 3 of 3 saved (folds 1-3; fold 4 missing), augmentation-only 2 of 4 (folds 1-2), frame-only 1 of 4 (fold 1). Even if the missing BPDR fold 4 had left, the maximum is 7 of 12, below the declared 11 of 12. **Rule not met, so no arm ordering is claimed and no held-out comparison was computed** (no `results.json`). The change helped (3/12 -> 6/11) but did not make the controls train.
- **Observation, a hypothesis not a finding:** BPDR left the plateau in every saved fit while the anchor-only controls did so in 3 of 8. BPDR's counter-detuning term compares the clean and shifted views of the same sample, a pairwise signal that anchor-only arms lack, so this may be the response terms making the problem trainable. It could equally reflect weighting. Testing it needs controls that actually train (e.g. a declared, matched longer-update or different-schedule attempt 3), and held-out error must then be compared at matched fit health.
- **Do not claim** BPDR beat the controls. State that the matched controls did not train in most folds and the comparison is therefore unresolved.
- **(SUPERSEDED 13 Nov by the feasibility plan section 3.2: no schedule/learning-rate attempt 3.)** Earlier note: **Next, if wanted (attempt 3, must be declared first):** a control-fit repair that targets the anchor-only arms specifically (more updates or a higher learning rate), checked on the fixed batch AND on fit health across the folds before any held evaluation. Needs about 1 h per full run and free memory; three low-memory stops occurred on this 15 GB machine.

## 17. Pointer (13 Nov 2026)

Feasibility review, the diagnosis of why the Transformer fails, the reformulated BPDR candidate (R1) and the re-prioritised improvement list are in `documentation/BPDR_Feasibility_and_Component_Plan_2026-11-13.md`. It supersedes the priorities in sections 5-7 above and the statement (redesign plan section 2) that BPDR output does not reach teammates: on `origin/main` the orchestrator passes `accepted.wav` to Vishwa's live pYIN. New no-fit evidence: `results/transformer_usability/2026-11-13_loss_redundancy/results.json` (tool `tools/probe_loss_redundancy.py`).
