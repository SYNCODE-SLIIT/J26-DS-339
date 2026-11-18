# BPDR component redesign plan (10 Nov 2026)

Labels: **[V]** verified today by reading code/results or a short run; **[R]** taken from the project's own records, not re-derived today; **[P]** proposed, untested.
Supersedes only the *priority order* of `Continuation_Method_and_January_Plan_2026-11-07.md` (cleanup first). Its hygiene rules, frozen gates, reserved-data rules and 31 Dec target stay in force. Nothing here is a result; no model was trained or changed.

## 1. Bottom line

1. The Transformer is not the bottleneck [R]: GRU 18, TCN 19, ExtraTrees 20 vs Transformer 23 verified repairs of 200; exact-reference repair works (24.9 to 8.6 cents), blind detection is near chance (AUROC .32-.64).
2. **No preparation step has yet been shown to improve the real downstream consumer [V, see section 3].** The component's defensible value is therefore (a) never harming usable audio, (b) telling the user what is wrong and where, (c) delivering the audio plus trustworthy evidence, and (d) the BPDR research comparison, reported honestly.
3. The research novelty still stands [V]: BERT-APC tests shifted boundaries only; the 2026 reference-free preprint (arXiv 2610.11524) uses SOME notes, assumes the note sequence is correct and fixes onsets/durations. Neither tests false splits/merges. This is a gap statement from two papers, not a worldwide priority claim.

## 2. What teammates own (do not duplicate)

| Owner | Component | Evidence read |
|---|---|---|
| Vishwa | Melody understanding: `src/melody_understanding` pYIN baseline, nearest-semitone note segmentation; its own warning says it "may fragment vibrato or merge repeated notes"; no key/beats yet | [V] code |
| Pasan | Harmonic planning (temporal Transformer on notes/beats/key) | [R] `pp1_demo` plan |
| Rahul | Chord refinement (multi-task Transformer) | [R] README |

[V] In `pp1_demo/orchestrator.py` the BPDR stage produces `accepted.wav`, but Pasan and Rahul consume saved dataset notes ("Vishwa stage deferred"). On this branch BPDR output does not reach the teammates. Vishwa's remote branch has a "live melody analysis" commit (30 Oct) that I did not inspect. **First impact item: confirm with Vishwa that the accepted WAV is the extractor's input and that its hash is checked.**
BPDR must not output notes, key, beats, tempo or chords.

## 3. Measured impact today (the table to show supervisors)

All use the real consumer `analyze_vocal` (pYIN), note F1 vs annotator A1, 50 c / 50 ms; 21 training-role Vocadito sources, 14 singers.

| Step | Result | Label |
|---|---|---|
| Neural noise removal vs format-only original: singer-macro F1 change | clean -0.010; 10 dB -0.009; 0 dB +0.003 (95% CI -0.017 to +0.024) | [V] read from `2026-11-08_primary_cleanup_development/results.json` |
| Same, case counts (F1 change beyond 0.02) | 0 dB: 32 better / 35 worse of 84; 10 dB: 26 / 40 of 84; clean: 2 / 4 of 21 | [V] |
| **Upper bound of any original-vs-cleaned selector** (perfect choice per case) | +0.033 (0 dB), +0.026 (10 dB), +0.003 (clean). Baseline F1 is only about 0.28-0.30, even on clean audio (0.293) | [V] computed from the same records |
| Global tuning-offset normalisation (f0 shifted 0/25/50 c, truth shifted with it) | Flat: 0.303-0.306 raw; re-centred 0.313 (about +0.01, within noise); estimated/reference notes 1.27-1.31 raw vs 1.22 re-centred | [V] 21-file probe; analytic f0 shift, not physical audio, one annotator |
| *(Superseded by section 10 and the build plan: a level-normalisation confound and a possible pYIN frame-size confound were found later, and the script cited here was a scratchpad file.)* **Pitch tracker feeding the same segmenter: PESTO (mir-1k_g7, confidence > 0.5) vs the consumer's pYIN**, 21 files, first 8 s, F1 averaged over A1/A2 | clean 0.288 -> 0.434 (+0.147, bootstrap 95% [+0.064, +0.220], better/worse 15/3); 10 dB 0.273 -> 0.496 (+0.223, [+0.159, +0.298], 19/1); 0 dB 0.276 -> 0.478 (+0.203, [+0.146, +0.260], 20/1) | [V] `scratchpad/pesto_vs_pyin.py`; one pre-chosen threshold, no tuning; bootstrap over 21 files (14 singers), not singer-level; real ESC-50 clip 1-100210-A-36 mixed digitally |
| Short-note pitch repair (V71) | 22 beneficial, 89 harmful of 445 | [R] |
| Balanced anchor, fixed training batch | positive error 22.7 to 3.06 c, signed response 0.84 | [R] |
| Corrected 3-arm run, 12 checkpoints | replay receipt exists (12 reloads, 4,482 arrays); `advance_to_delivery` = false; first-pass macro error BPDR 21.39 / aug-only 23.25 / frame-only 24.91, input about 24.8; BPDR jitter sensitivity worse (0.258 vs 0.109) | receipt/flag [V]; numbers [R] |

Reading: the largest measured lever is the pitch tracker, not cleanup (+0.15 to +0.22 F1 vs at most +0.03 from any cleanup selector). With a 0.29 clean-audio pYIN F1, **the extractor, not the audio, is the limit for the current pipeline**, and at most about 3 F1 points are recoverable by cleanup. Do not promise a big downstream gain; do not make DeepFilterNet or tuning normalisation the lead.

Data hygiene note: my first probe also read Vocadito IDs 2, 4, 5, 7, 9, 11 and 12, which are not among the 21 training-role sources. It read audio and A1 annotations only, trained nothing, and was discarded; the reported probe uses training-role IDs only. Record this in the evidence log and have the owner decide whether those files remain unopened for reserved-data purposes.

## 4. Redesigned component

```
solo hum -> A validity + located guidance -> B candidates (original / conservative cleanup)
         -> C evidence sidecar -> D do-no-harm selector -> E optional BPDR local repair (review-only)
         -> F decoded-WAV audit + user review -> G accepted WAV + sidecar to Vishwa
```

- **A. Validity and located guidance** (exists): clipping, level, voiced coverage, noise-only reference, time ranges, rerecord text. Keep the four public statuses.
- **B. Candidates**: original always available; the proposal's conservative spectral path; DeepFilterNet only as a clearly labelled optional candidate. No declipping, dereverb, separation or generative restoration.
- **C. Evidence sidecar [P]**: seconds-based JSON with per-frame pitch reliability, voicing, octave-jump flags, SNR estimate, clipped spans, and the singer's tuning-centre offset **as metadata only**. Metadata, not notes; Vishwa decides whether to use it. Needs Vishwa's sign-off on the schema. **Tested [V]: a PESTO f0 track gives much better note F1 than pYIN in the same segmenter (section 3), so the sidecar is worth building.** The swap belongs to Vishwa; BPDR supplies the evidence and the measured comparison. Next confirm on a fresh source/annotator split and with a threshold chosen on development data only.
- **D. Do-no-harm selector [P]**: choose original vs candidate from input/output acoustics only. Fixed rule first; a regularised linear model or small ExtraTrees only on independently grouped data. Report coverage and harm together. The measured headroom above (about 3 F1 points) sets the honest expectation.
- **E. BPDR repair**: stays the research arm and a review-only, user-accepted edit. Spec of the proposal's three matched arms is unchanged (augmentation-only, frame-only, full BPDR; same data, four views, budget).
- **F/G.** Existing audit, protected samples, rollback, hashes. Acceptance check: the user-selected WAV's hash equals the extractor's actual input.

## 5. The research comparison, redesigned for evidence not capacity

Keep the loss and the three Transformer arms as the novelty. Change only how the question is tested:

1. **Learning curve over number of training singers**, three arms by three preselected seeds (the proposal promises three; only seed 339 has been used [R]). The proposal says the response losses should help "when the available data is limited", so test that directly. Models are about 21k parameters, so cost is small [R]. Pre-declare budget and the singer-level paired bootstrap before opening held data. **Pass criteria must include harm at matched coverage**, not error alone.
2. **Report the regimes separately**: short notes (information-limited, near input error) and stable/held notes (earlier raw error about 12 c [R]). I previously suggested held notes as a likely fix; that was too optimistic: held-note raw error did not turn into delivered repairs (23/200) and the one earlier matched three-arm run was a tie (5.84 / 5.64 / 5.65 c) [R]. Held notes are a second regime to report, not the rescue.
3. **Stop** architecture, loss-weight and activity-gate sweeps. The balanced anchor stays as the one declared correction.

## 6. Data

Choose by number of **independent scoreable singers after the render audit**, not hours of audio [R]: today about 8-12 scoreable singers, 4 humming singers (HumTrans), and 707 of 1,152 attempts excluded; augmentation does not add people. Order of action:
1. Pull the 707 exclusion reasons by category; fix the commonest (rendering/coverage) before adding data.
2. The consented 8-adult pilot is the real data fix; ask whether ethics approval and reviewers are in place.
3. Check how many HumTrans singers exist vs the 4 used, and what VocalSet contributed in `vocalset_*` folders, before naming any new dataset. Verify licence and singer count before download. I have not verified any new dataset.

## 7. Next steps (small, in order)

1. Confirm with Vishwa: accepted-WAV input, optional sidecar schema, and which extractor is primary (pYIN vs GAME).
2. DONE: PESTO-vs-pYIN check (section 3). Next: hand Vishwa the table; replicate on a fresh split before claiming it; choose the voicing threshold on development data only.
3. Exclusion-reason audit and singer inventory (read-only).
4. Declare the learning-curve x three-seed comparison, then run it; report whatever comes out.
5. Replace the "cleanup is the strongest candidate" statement in the 7 Nov plan with the section 3 table.

## 8. Sign-offs and dependencies only people can supply

Supervisors: scope of the final claim, and that the headline may be a measured null with a do-no-harm component. Vishwa: handoff metadata and extractor choice. Ethics approval, participants and two independent reviewers. Exact presentation date (assumed January 2027).

## 9. Not claimed

No candidate is qualified; no downstream gain is established; probe numbers are from 21 files, one annotator, an analytic f0 shift, and are directional only.

## 10. Vocal Rescue Studio: built 10 Nov (panel demonstration)

Run: `Run-Studio.ps1` (port 8504). Code: `source/bpdr_input/studio/`, UI `application/studio_app.py`, tools `tools/studio_evidence.py`, `tools/studio_validate.py`, `tools/probe_tuning_offset.py`, tests `tests/test_studio.py`. The frozen review demo is untouched; nothing is committed (`/results/*` is git-ignored here). **[V]** = run today.

What it does: validates (clipping measured on the decoded samples, silence, speech, voiced coverage) with located re-record guidance; PESTO pitch evidence at a standard level; a spectral-gate candidate chosen only when pitch/voicing are preserved; the four public statuses; accepted WAV + pitch-evidence sidecar (with the normalisation gain); before/after piano rolls and synthesized "what the extractor heard"; note F1 against human annotators; a listening-only cleaned preview; harmony through the teammates' Pasan and Rahul models (read-only calls, genres limited to Rahul's vocabulary, major only); optional speech-to-melody mode (labelled not-repair, outside the research claim).

**What the extractor actually receives.** The PP1 adapter runs `analyze_vocal` (pYIN) on whatever WAV it is handed, and the prepared WAV is the original at a standard level. So today's extractor gets no benefit from the Studio; the gain below exists only if the extractor reads BPDR's pitch evidence (proposed, needs Vishwa). The UI labels the rows accordingly.

Measured **[V]** (`results/studio/evidence/pesto_vs_pyin.json`; 21 training-role Vocadito sources, 14 singers, first 8 s, one digital noise clip (vacuum cleaner), mean of A1/A2, bootstrap over SINGERS, development evidence only):

| Condition | pYIN today | PESTO, recorded level | PESTO + level normalisation (gain vs pYIN, 95% CI) | + cleanup always | Studio choice |
|---|---|---|---|---|---|
| clean | 0.288 | 0.435 | 0.492 (+0.204 [0.116, 0.247]) | 0.490 | 0.492 |
| 10 dB | 0.273 | 0.501 | 0.501 (+0.229 [0.134, 0.291]) | 0.501 | 0.501 |
| 0 dB | 0.276 | 0.483 | 0.481 (+0.206 [0.118, 0.248]) | 0.488 | 0.481 |

The gain comes from the pitch-evidence tracker plus level normalisation (source peaks range 0.07-1.01; normalising alone moved PESTO clean F1 0.435 to 0.492). The spectral-gate cleanup adds about nothing and was chosen in 0 of 21 files in every condition, so the prepared audio still sounds noisy. An earlier table credited clean-audio "cleaning" with +0.06; that was a normalisation confound, caught and fixed.

False rejections **[V]** (`validation.json`): 96 sung or hummed cases (21 Vocadito files x clean/rain/vacuum/clock-tick = 84, plus 12 real HumTrans training hums): 95 ACCEPT_UNCHANGED, 1 ACCEPT_PREPARED, 0 rejected.

Four annotated demo examples **[V]**, note F1 today / shared extractor on prepared WAV / proposed PESTO evidence: rain 0 dB 0.261 / 0.261 / 0.358; vacuum 0 dB 0.136 / 0.136 / 0.333; clock ticks 10 dB 0.252 / 0.252 / 0.306; **clean expressive hum 0.302 / 0.302 / 0.201 (PESTO is worse on this file; kept and reported)**. One file each, an illustration, not proof.

Checks run **[V]**: 11 unit tests (including a 48 kHz stereo clipping case); all 8 examples and the Harmonize / Compose buttons in a headless UI test; the real server started and served the page (HTTP 200, health ok).

Not done / limits: no listening review of the speech melody or the harmony; speech detector calibrated on 3 speech clips only (threshold 0.22 is preliminary; the 96 sung/hummed cases were not rejected as speech); no real novice phone recordings; upload and microphone paths untested in a browser; noise is one clip family in the study (rain and clock ticks appear only in the demo examples); no replication on fresh singers; harmony defaults to a user-chosen key and tempo that are not estimated from the hum (key and tempo estimation belong to other components).
