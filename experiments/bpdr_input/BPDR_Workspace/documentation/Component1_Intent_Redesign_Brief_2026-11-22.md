# Component 1 redesign brief: from "repair the pitch" to "find out what the singer meant" (22 Nov 2026)

Author's reading of the request: the owner asked for (a) a user flow that matches how the component will really be deployed, in plain HTML rather than Streamlit, (b) a stronger research novelty that makes Component 1 matter to the whole project, and (c) a clear statement of what the next components receive. This file is both the plan and, in section 12, a paste-ready build prompt.

Read first: `BPDR_Feasibility_and_Component_Plan_2026-11-13.md` (verdicts and frozen rules) and the proposal `IT23141506_J26_DS_339.pdf`. Labels: **[V]** verified in the repo files named; **[R]** from project records; **[P]** proposed, untested. Nothing in sections 3-9 is built.

## 0. Three things to fix before anyone reads the numbers

`Research_Justification_and_Methodological_Pivot.md` (untracked) is the newest write-up, but several headline claims in it do not survive the project's own records. Do not circulate it, and do not copy its numbers into the UI, the panel deck or a build prompt.

| Claim in the pivot document | What the records say |
|---|---|
| "Note F1 0.301 -> 0.631, +110%" | The 0.301 baseline is `pyin_4096`. About 88% of the old "PESTO gain" is a frame-length effect (a 2048 frame gives +0.18 on clean audio, `results/studio/evidence/frame_control.json`). The fair comparison for the learned boundary model is `pesto_notes_from_track` 0.486 -> 0.631, **+0.145 [+0.087, +0.201]** (plan section 14.3). [V] |
| Selected model is a clean win | `studio_v1` was chosen by comparing v1-v4 on the same 39 training-role Vocadito windows it is scored on: selection on the evaluation set. The seed-340 replication gives 0.579, not 0.601 (plan section 14.1). Development evidence only. [V] |
| "BPDR Boundary-Pitch Disentangled FastTNet" | The best arm is `frame_only`. BPDR's loss terms add nothing: +0.008 [-0.004, +0.022] (TCN), -0.006 [-0.033, +0.021] (lean Transformer). Call it "a learned boundary model", never "disentangled". [V] |
| Preparation improves the extractor | `pyin_4096_prepared` is +0.000 over `pyin_4096`. Today the shared extractor gains nothing from this component; any gain needs Vishwa to read the sidecar. [V] |
| "0% harmful waveform distortion, guaranteed" | Never measured as a rate; no confirmation set was opened. Delete the sentence. |

## 1. Where the project actually stands [R]/[V]

- **Reference-free repair of small pitch errors is not identifiable.** Natural singing sits 19-25 c from a note target; the injected error is 25 c. Blind detection AUROC 0.62-0.64 against a 0.85 bar; Transformer, GRU, TCN and ExtraTrees all land at about 10% verified repairs; with the exact reference supplied, the same short-note path goes 24.9 -> 8.6 c (294 of 445 verified beneficial, not independently replayed). The limit is missing information, not the model.
- **The weight-matched three-arm control (R2, 12/12 healthy fits) finds BPDR no better than controls and fails the delivery gate in all arms.** That is the honest negative result for the original hypothesis and it stays in the thesis.
- **What does work:** 0 of 96 sung or hummed inputs falsely rejected; located clipping/silence/speech feedback; a learned boundary model that beats the extractor's own boundaries on onsets and is stable under false split, merge and jitter (development data).
- **The one gap nobody fills:** nothing in the pipeline asks the *singer* which of two plausible readings is right. Yet the proposal already treats the singer's judgement as evidence (section 3.5: "the singer's own judgement of whether the musical idea was preserved will be considered") and FR5 already lets the user compare playback and choose another take.

## 2. Research direction: Active Intent Confirmation (AIC) [P]

### 2.1 One-sentence idea

Because intent cannot be read from the waveform, treat it as a *latent variable with a measured posterior*, and spend the singer's attention only on the few ambiguities whose answers change the downstream melody the most.

### 2.2 Why this is the BPDR idea, not a replacement

BPDR separates two factors: **boundary** (is this one note or two?) and **pitch** (is this deviation an error or an inflection?). AIC asks one question type per factor.

| Factor | Ambiguity the system detects | Question the singer hears | What the answer fixes |
|---|---|---|---|
| Boundary | Extractor boundary with no learned-model peak (possible false split) or learned peak with no extractor boundary (possible false merge); peak probability in an uncertain band | A/B: the same 2-3 s rendered as one held note vs two notes, original audio first | A **confirmed boundary constraint** for the span |
| Pitch | Steady voiced span whose tuning-adjusted deviation is 18-45 c (the repair-candidate band, `boundary.analyze_boundary_spans`) | A/B: "as sung" vs "centred on the nearest note" | "As sung" = intended inflection: **preserve**. "Centred" = a **reference now exists**: repair becomes well-posed (this is the oracle-reference condition, 24.9 -> 8.6 c) |

So the user's answer is exactly what the information-limit finding says was missing, supplied at the cheapest possible cost.

### 2.3 What is novel, and what is not

- **Not claimed:** the hear-back loop (already Tier 1 item 1 of the feasibility plan), a Transformer, tool integration, consistency losses.
- **Claimed as the research contribution [P]:** (1) *which* question to ask next, chosen by expected information gain over boundary and pitch hypotheses and weighted by downstream impact, and (2) **calibrated per-span uncertainty** that travels with the audio. Neither is given by the hear-back loop alone.
- **Prior work to cite and differ from:** human-in-the-loop melody transcription exists. Tony (Mauch et al., 2015; pYIN-based, co-authored by Bittner, who also co-authored the Vocadito dataset; citation details to verify) lets an expert correct pitch tracks and notes by hand. Here a *novice* answers a few value-chosen A/B questions and calibrated uncertainty is passed downstream. **Task before the supervisor meeting:** search "interactive melody transcription", "active learning music transcription" and "human-in-the-loop pitch correction" and add what is found. Consistent with the proposal, no worldwide priority claim is made.
- **Framing:** an extension of proposal sections 3.5 and FR5, and of plan section 3.1 (R1: "repair only toward a supplied reference"). It changes the headline hypothesis from "repairs more stably" to "finds out what to repair and what to leave alone". **That wording change is the supervisors' decision**, as plan section 3.1 already says. The original hypothesis is reported unchanged as the honest negative.

### 2.4 Hypotheses (declare before running; frozen rules and reserved data unchanged)

- **H-I1a boundary questions (budget curve).** With K answers (K = 0..5), onset F1 and onset-and-pitch note F1 rise faster under EIG-ordered boundary questions than under (a) no questions, (b) random order, (c) lowest-confidence-first. Equal K, singer-level 95% intervals. *Expected risk:* for binary questions lowest-confidence and max-entropy nearly coincide, so the EIG advantage may be small or null; impact weighting is where a difference could come from. Report a null as a null.
- **H-I1b pitch questions (separate metric).** Note F1 **cannot** score pitch questions: the scorer calls `mir_eval` with its default 50 c pitch tolerance (`studio/scoring.py` passes none; library default, not re-run in this shell), and a 18-45 c centring almost never flips a note label (plan section 2.3: 10-15% of labels at 20-30 c). So pitch questions are scored on the proposal's primary metric, **absolute correction error in cents** against the known compensating shift, plus **harmful-edit rate at matched coverage**, with the same four strategies and K grid.
- **H-I2 calibration.** Per-span uncertainty is calibrated (ECE and Brier against the simulated-user answers) better than raw model confidence or raw boundary probability, on singers not used to fit the calibrator.
- **H-I3 do-no-harm.** At equal K, applying answers never reduces note F1 by more than a declared margin on spans the oracle marks "keep as sung", and the harmful-edit rate is reported at matched coverage (proposal Table 3).

### 2.5 Evaluation design

- **Simulated singer (no new human data), by question type.**
  - *Boundary questions:* answers come from note annotations. Vocadito has two independent annotators (`datasets/vocadito/Annotations/Notes/*_notesA1.csv`, `*_notesA2.csv`, present on disk), so A2 acts as a *noisy* user whose disagreement with A1 sets a realistic error rate; a perfect oracle is the upper bound. HumTrans has one DTW-derived label set, so the noisy-user variant **does not exist** on the reserved HumTrans material: report the oracle variant only there.
  - *Pitch questions:* Vocadito and HumTrans labels sit on the semitone grid and cannot say whether a 25 c deviation was intended. Simulate on the controlled-detuning set (`2026-11-04_v64_reliable_melodic_pairs`: 53 originals, 445 detuned): an injected shift means "centred", an untouched original means "as sung", by construction. Label it development data; it says nothing about natural inflection.
- **Data discipline.** The 39 Vocadito training-role windows were used to select the model, so they serve development and calibration only, never a headline. Fit and calibrate policies leave-singer-out on training-role and development material (HumTrans training persons, M02). The confirmatory run on the reserved material is opened **once** after thresholds freeze; HumTrans M03-M05, VALID and TEST stay untouched until then. HumTrans labels are DTW-derived, so any HumTrans number is described as inflated.
- **Declaration first:** write `results/studio/aic/declaration.json` with the question types, K grid, baselines, metrics, seeds, split rule and win condition, hash it, then run.
- **Real users need ethics approval.** A human pilot (proposal: 8 adults x 6 phrases) is not assumed. Until approved, teammate walk-throughs are usability checks only and are labelled "not research evidence".
- **Known limit to state in the thesis:** for free humming there may be no fixed intended melody; an answer then records a *preference between offered readings*, not recovered ground truth.

## 3. Non-overlap with the other components

| Owner | Owns | Component 1 does NOT |
|---|---|---|
| Vishwa (melody understanding) | Notes, pitch, beat, tempo, key, phrase boundaries | Output notes, a key grid, tempo or beats. Boundary evidence and answers are *hints and constraints*; the hear-back synthesis is a read-only preview of the shared extractor |
| Pasan (chord progression) | Functional chords from melody and beat context | Choose chords |
| Rahul (chord refinement) | Detailed chord labels | Refine chords |
| Praveen (this component) | Validation, preparation, uncertainty, user-confirmed constraints, traceable hand-off | Separate voices, recover clipping, make melody from speech |

## 4. What the next components receive (hand-off contract v0.2 [P])

**Verified today.** On `origin/main` the PP1 orchestrator (`pp1_demo/orchestrator.py`) copies BPDR's `accepted_audio` to `accepted.wav` and calls Vishwa's `predict(accepted.wav)` (live pYIN plus the weak-score classifiers). Pasan then reads the notes and a saved or user-set beat grid (`pp1_demo/contracts.py`: tonic, bpm, meter, first beat). **Only the WAV is consumed; the sidecar is not.** This branch is behind `origin/main`.

| Item | Status | Needs |
|---|---|---|
| `accepted.wav` (SHA-256 equals the file read downstream, contract-tested) | exists | nothing |
| `sidecar.json`: PESTO f0/confidence/voiced track, quality report, tuning centre, advisory boundary peaks, span diagnostics | exists | **Vishwa adopts** it (e.g. frame-length 2048 or boundary hints) before any gain is real |
| `intent_map`: per span `{t0, t1, p_split, p_merge, p_inflection, state: confirmed_boundary / confirmed_as_sung / unresolved}` | to build | **Vishwa adopts** `confirmed_boundary` as a constraint on note segmentation |
| `trust_curve`: calibrated 0-1 melody trust per second | to build | **Proposal to Pasan/Rahul:** weight chord confidence per bar, or offer alternatives where trust is low; their decision |
| `user_answers` log with times, versions, policy hash | to build | nothing (traceability, proposal FR6) |
| Four statuses `ACCEPT_UNCHANGED`, `ACCEPT_PREPARED`, `RERECORD_REQUIRED`, `TECHNICAL_FAILURE`; the last two carry no audio path | exists | nothing |

First step with teammates: show Vishwa the frame-length table and `confirmed_boundary`; ask Pasan whether a per-bar trust value is usable at all before building it.

## 5. User flow (what users see is what we deploy)

One path, one decision per screen, plain language, nothing the user cannot act on.

| # | Screen | What happens |
|---|---|---|
| 1 | **Welcome and privacy** | One-line purpose, "one voice, up to 20 s, no backing track", the 24 h deletion rule, consent tick. |
| 2 | **Room check (optional, 3 s)** | Stay silent for 3 s; result is good / noisy / too loud with one tip. Also supplies the voice-free noise reference the proposal requires. |
| 3 | **Record or upload** | Large record button, live level meter with a text label, clipping warning during recording, 20 s counter, upload alternative, playback before sending. |
| 4 | **Verdict** (about 10-30 s, progress bar) | One of four states in words and an icon, never colour alone. A re-record request names the observation and the action with time ranges ("clipping at 3.2-3.9 s: move back from the microphone"). A long file offers its best <=20 s phrase and trims silence only after the user confirms. |
| 5 | **"Here is what I heard"** | Waveform plus pitch ribbon, play the original and the synthesised hear-back, flagged moments marked. Plain question: "Does this sound like your tune?" Yes goes to 7. Not quite goes to 6. |
| 6 | **Clarify (at most K = 3 cards, hard cap 5)** | Each card: one moment, two playable options, "I can't tell / skip" always available. Progress "2 of 3" and a trust meter that moves with each answer. Keyboard: 1, 2, S. |
| 7 | **Review changes (only if the user chose "centred")** | Original vs prepared A/B per span, the post-render check result, per-span accept or revert. Rolled-back spans say why. Nothing is applied silently. |
| 8 | **Hand-off receipt** | Status, accepted-audio hash, exactly what travels onward (collapsible JSON), "record another take", **Delete my recording now**. |

**Separate "Research view" (`/research`)** for the panel, kept apart from the user flow: question-budget curve with the three baselines and intervals, calibration plot, ablation table (including the BPDR-vs-controls negative result), the claims ledger (verified / unverified, section 11), and provenance hashes. Users never see it.

## 6. Architecture and requirement changes

- **Stack:** static HTML + vanilla JS (no build step) served by a small **FastAPI** app that wraps `studio.pipeline.prepare` and the new question engine. One launch command for Windows (`Run-Component1.ps1`); a Dockerfile is optional.
- **API sketch:** `POST /api/sessions`, `POST /api/sessions/{id}/audio`, `POST .../analyze` (returns a job id; poll or stream progress), `GET .../questions`, `POST .../answers`, `POST .../repair`, `GET .../handoff`, `DELETE /api/sessions/{id}`. Sessions use 32-hex ids so the existing retention code applies.
- **Recording:** browsers record webm/opus, which SoundFile cannot decode. Capture PCM with an AudioWorklet and encode WAV in the browser (or run server-side ffmpeg). Request `getUserMedia` with `echoCancellation`, `noiseSuppression` and `autoGainControl` all **off**: they alter the pitch evidence. Microphone access needs `localhost` or HTTPS.
- **Resources:** one bounded phrase per request (proposal NFR scalability); 15 GB machine, CPU torch; one job at a time with a queue message.
- **Carried over unchanged:** 24 h retention purge and delete button (`studio/retention.py`); response target of 20 s phrase within 30 s, reporting median and 95th percentile; keyboard-operable controls, text labels, no colour-only meaning, text explanation for every audio finding (NFR table 5).
- **Requirement change to record in the decision log:** **FR1 names Streamlit for capture and playback. Replace with static HTML + FastAPI.** Reason: the deployed experience needs low-latency A/B audio cards, keyboard shortcuts, client-side WAV capture and a flow that is the same in the demo and in deployment; Streamlit's rerun model is a poor fit for that (design judgement, not measured). All six current apps in `source/bpdr_input/application/` import Streamlit, so this replaces the delivery layer, not the pipeline. Behaviour requirements FR2-FR7 are unchanged; FR5 is extended with the clarification step.

## 7. Question engine (technical outline [P])

1. **Candidates.** From the sidecar and boundary hints build hypotheses per ambiguity: boundary (split / merge / keep) and pitch (as sung / centred).
2. **Prior.** Calibrate hypothesis probabilities (isotonic or temperature scaling, fitted leave-singer-out) from boundary probability, extractor-vs-model disagreement, span length, vibrato depth and deviation.
3. **Value of a question.** `score = EIG(question) x impact(question) / cost`, where cost is listening time and impact depends on type. *Boundary:* the change in the note stream or beat-aligned bars if the answer flips. *Pitch:* **not** note-stream change (about 0 at a 50 c tolerance, so the engine would never ask one); use the expected reduction in absolute correction error (cents) times the probability the span is an error. Keep a **separate budget for pitch questions** (declared, e.g. at most 1 of K) so impact weighting cannot starve them. Pick greedily; stop at K, or when remaining value is below a declared floor.
4. **Update and constrain.** Each answer updates the posterior and writes a constraint into `intent_map`; "skip" leaves the span `unresolved` and the original audio untouched.
5. **Repair.** Only spans answered "centred" go to `span_repair.verify_and_apply_span_repairs` (bounded <= 35 c, crossfade, post-render PESTO check, rollback). The proposal's thresholds stay frozen.
6. **Baselines in the same code path:** none, random, lowest-confidence. Same interface so the curve is one switch.

## 8. Plan to 11 Dec (one heavy job at a time, owner's own terminal)

| Window | Work |
|---|---|
| 22-25 Nov | Supervisor message on the changed headline wording; write the AIC declaration; extract ambiguity sets on development data (no training) |
| 26 Nov-1 Dec | Question engine and simulated-singer curves on development data; in parallel API skeleton and screens 1-4 |
| 2-6 Dec | Calibration, then freeze thresholds; screens 5-8 and the research view; walk-through with 3-5 teammates (usability only) |
| 7-10 Dec | Confirmatory run once on reserved material; hand-off test against the `origin/main` orchestrator (merge is the owner's decision); write the result |
| 11 Dec | Decision point, unchanged |

## 9. Decisions only the owner or supervisors can make

1. Adopt "Active Intent Confirmation" as the headline, with the original BPDR hypothesis reported as the honest negative (supervisors).
2. Authorise a one-time confirmatory run on the reserved HumTrans material (owner and supervisors).
3. Ask Vishwa to consume `confirmed_boundary` and the frame-length change; ask Pasan whether a trust curve helps (teammates).
4. Ethics approval for a human pilot (institution).
5. Merge `origin/main` into this branch (owner).

## 10. Honest risks

- The question engine may show **no advantage over lowest-confidence-first**; the contribution would then be the calibration and the hand-off contract, and the thesis says so.
- Simulated answers are cleaner than human ones; human perceptual limits for a 20-30 c difference inside a melody are untested.
- Users may not know what they intended; fewer, better questions matter more than many.
- Gains for the other components are conditional on their adoption; without it, the measured effect on the live pipeline is zero.
- Boundary model is trained on humming and checked on singing; development evidence only.

## 11. Claims ledger (put this in the Research view and keep it current)

**Do not claim (verbatim from feasibility plan section 6):** That the Transformer repairs pitch; that BPDR beat the controls; that R1 works (untested); that the frame-length gain holds beyond 21 development files; that the component rescues any input.

Also do not claim: that BPDR's loss terms caused the boundary gain; that AIC improves the live pipeline (needs hand-off adoption); that the question engine beats simple baselines before H-I1 is run; "0% harmful distortion"; "+110%"; any human-user benefit before an approved pilot.

**May claim, with the stated limits:** do-no-harm acceptance and located re-record guidance (0 of 96 falsely rejected); the Studio's boundary-hint note path scores +0.145 note F1 over the PESTO note track on development data, singer-level interval [+0.087, +0.201], model selected on the same windows. **Confound:** that path (`boundary.notes_from_track_with_boundary_hints`) also takes one *median pitch per segment*, a 4-frame minimum and a 100 ms unvoiced-gap split, whereas the baseline (`evidence.notes_from_track`) splits at every semitone change. So part of the +0.145 may come from the note-formation rule, not the learned boundaries. **Open ablation:** the same rule with extractor boundaries only, and the rule with the gap split only. Until run, word it as "boundary hints plus a segment-median note rule". A separate, cleaner statement is that the learned boundary model beats the extractor's own boundaries on *onset* F1 (+0.087, development); reference-free blind repair is information-limited (the negative result, four model families, matched controls).

## 12. Paste-ready build prompt

```
You are building the deployable user interface and service for Component 1 of research project
J26-DS-339 (vocal input validation and minimal repair; owner: Praveen, IT23141506). Work in
experiments/bpdr_input/BPDR_Workspace. Do NOT commit; the working tree has unrelated uncommitted
changes. Do not download datasets. Run heavy jobs one at a time (15 GB RAM).

READ FIRST: documentation/Component1_Intent_Redesign_Brief_2026-11-22.md (this plan),
documentation/BPDR_Feasibility_and_Component_Plan_2026-11-13.md, the proposal PDF in documentation/,
source/bpdr_input/studio/pipeline.py, boundary.py, span_repair.py, retention.py, room.py, and
tests/test_studio.py. Do not use Research_Justification_and_Methodological_Pivot.md for numbers.

GOAL: replace the Streamlit front end with static HTML/JS plus a FastAPI service that wraps
studio.pipeline.prepare, implementing the eight-screen flow in section 5 of the brief, a
question engine (section 7) and a separate /research view. The flow users see is the flow we deploy.

REQUIREMENTS
1. Layout: web/ (index.html, ES modules, CSS, no build step) and source/bpdr_input/web/ (FastAPI app,
   session store, question engine). Launch with Run-Component1.ps1 (one command, opens the browser).
2. Recording: AudioWorklet PCM capture, WAV encoded in the browser; getUserMedia with
   echoCancellation, noiseSuppression, autoGainControl all false. Upload alternative. Level meter,
   clipping warning, 20 s cap.
3. API: POST /api/sessions; POST /api/sessions/{id}/audio; POST .../analyze (job id + progress);
   GET .../questions; POST .../answers; POST .../repair; GET .../handoff; DELETE /api/sessions/{id}.
   Session ids are 32 hex characters so studio/retention.py applies. Keep the 24 h purge and
   a visible Delete my recording button.
4. Statuses exactly: ACCEPT_UNCHANGED, ACCEPT_PREPARED, RERECORD_REQUIRED, TECHNICAL_FAILURE; the
   last two never carry an accepted-audio path. Failures never overwrite the original.
5. Question engine: ambiguity candidates (boundary split/merge, pitch as-sung vs centred), calibrated
   prior, greedy expected-information-gain x downstream-impact / cost, K = 3 default, cap 5, skip
   always allowed. Strategies none / random / lowest_confidence / eig behind one interface.
   Repair only for spans answered "centred", through verify_and_apply_span_repairs with rollback.
6. Hand-off v0.2: accepted.wav, sidecar.json, intent_map, trust_curve, user_answers (see brief
   section 4). Mark every field that needs a teammate to adopt it as "proposal" in the JSON and the
   UI. Component 1 NEVER outputs notes, key, tempo or chords as its product; the hear-back audio is
   a read-only preview of the shared extractor.
7. Accessibility: keyboard-operable, text labels, no colour-only meaning, text explanation for every
   audio finding. Performance: a 20 s phrase processed within 30 s on this machine; report median
   and 95th percentile.
8. Research view at /research, visually separate from the user flow: question-budget curve vs three
   baselines with intervals (render from saved results files only), calibration plot, ablation table
   including the BPDR-vs-controls negative result, the claims ledger, provenance hashes.
9. Tests: extend tests/ so existing tests still pass (python -m unittest tests/test_studio.py
   tests/test_analysis.py); add tests for the API contract, retention, the hand-off SHA-256
   equality, the four statuses, and the question engine on synthetic cases. Verify the UI in a real
   browser (record, upload, answer, repair, delete) and report what you could and could not check.

DO NOT CLAIM (word for word): "That the Transformer repairs pitch; that BPDR beat the controls; that
R1 works (untested); that the frame-length gain holds beyond 21 development files; that the
component rescues any input." Also never write: "+110%", "0% harmful distortion", that BPDR's loss
terms caused the boundary gain, or that the question engine beats baselines (unmeasured). The only
gain figure allowed in the UI is "+0.145 note F1 over the PESTO note track on development data,
model selected on the same windows, boundary hints plus a segment-median note rule (ablation
of that rule not yet run)". Show claims through the claims ledger, not marketing copy.

DATA RULES: the 39 Vocadito windows are development only. Leave HumTrans M03-M05, VALID and TEST
untouched. Any experiment is declared first (hashed declaration file) and reports negatives.

FINISH by listing: what runs, what was verified (with commands), what is untested, and the
decisions that remain with the owner or supervisors (brief section 9).
```
