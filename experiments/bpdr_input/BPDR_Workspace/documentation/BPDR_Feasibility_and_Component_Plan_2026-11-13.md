# BPDR feasibility review and component improvement plan (13 Nov 2026)

Read after `BPDR_Studio_Build_Plan_2026-11-10.md` (sections 14-16 hold the 12-13 Nov runs). This file supersedes that plan's *priorities* (its sections 5-7) and its statement that BPDR output does not reach teammates. Hygiene rules, frozen gates, reserved-data rules, the 11 Dec decision point and the 31 Dec target are unchanged.
Labels: **[V]** verified by reading or running on 12-13 Nov; **[R]** from project records; **[P]** proposed, untested. Nothing in sections 3-5 is built.

## 1. Feasibility verdict

| Question | Answer |
|---|---|
| Can the current Transformer (reference-free repair of small pitch errors) be made usable by more tuning? | **No.** The limit is information, not the model. [R]+[V] |
| Is the research (the BPDR concept) still feasible? | **Yes, in a changed form** (section 3), plus an honest negative result for the original form. Needs supervisor agreement. [P] |
| Is a usable, impactful component feasible by 31 Dec? | **Yes.** Its measured levers are restraint, feedback, hand-off and one extractor finding; not silent repair. [V] for the parts named in section 2.5 |

## 2. Why the Transformer does not work (four separate limits)

**2.1 The target cannot be recovered from the input [R].** The model sees one recording and must output "how far is this note from what the singer meant". Natural singing sits about 19-25 cents from a note target; the injected error is 25 cents. A note that is 25 c off looks the same whether intended or not. Evidence: blind detection AUROC .62-.64 against a declared .85 bar; Transformer, GRU, TCN and ExtraTrees tie (23 / 18 / 19 / 20 verified repairs of 200); with the exact original as reference the November short-note path goes 24.9 -> 8.6 c (294 of 445 verified beneficial; not independently replayed). So it works when told the answer and fails when it must guess it. No architecture or loss fixes that.

**2.2 Delivery depends on having a reference [R].** With an exact reference the November short-note path delivered 294 of 445 verified repairs, so rendering is not a standing limit. Without one, the reference Transformer delivered 23 of 200 (October path, a different pipeline; its record says that figure is not a coverage ceiling). Review-only repair toward a supplied reference is therefore the workable route.

**2.3 The stake is small [R].** A 20-30 c error changes only about 10-15% of nearest-semitone note labels with per-recording tuning. A perfect repair could change at most that share of downstream notes.

**2.4 The three-arm comparison is not a fair test yet [V].**
- In the 7 Nov run 9 of 12 fits never left the zero output; with the gate opened in all arms, 6 of 11 (BPDR 3/3, augmentation-only 2/4, frame-only 1/4). Declared bar 11 of 12: not met.
- New, 13 Nov (`results/transformer_usability/2026-11-13_loss_redundancy/results.json`, frozen fold-1 batch, no fit): at initialisation the counter-detuning gradient has **cosine 0.92** with the positive-anchor gradient. Early in training the BPDR term is mostly extra weight on the same supervised target. Measured with the gate open and response terms on; the real run keeps response terms off for the first 200 updates, so this is **consistent with**, not proof of, a weighting advantage being why BPDR leaves the plateau more often. (At the fitted BPDR checkpoint the cosine is 0.003, so later the term does differ.) A fair control should match that positive weight.
- **Boundary sensitivity where a fit actually produces output [V, saved predictions, frames with |output| > 5 c, held folds; descriptive, tiny n]:** change under altered boundaries divided by response size: BPDR 1.0-2.5% (four folds, 0.18-0.51 c); augmentation-only 3.1-9.1% (three trained fits across both runs, 0.65-1.22 c); frame-only 0 by construction. So where the control trains, BPDR looks 2-4 times less boundary-sensitive, which is the direction the proposal predicts, but the absolute effect is under 1.3 c and rests on three control fits. The run-level aggregate (0.10-0.46 c for augmentation-only) understates it because fits that output zero are trivially invariant. This is the one early sign for the concept; it needs trained controls to be a result.
- Checked and fine: the split/merge/jitter corruption does reach the model (altered-boundary predictions differ by up to 19 c across kinds). The identical raw aggregates across kinds are expected, because raw scoring uses the unaltered-boundary row.

**2.5 What does work, measured [V].** 0 of 96 sung/hummed inputs falsely rejected; located clipping/silence/speech feedback; a 2048-sample pYIN frame gives +0.18 note F1 over the shared 4096 frame on clean development audio (88% of the earlier "PESTO gain"); cleanup adds about 0. On `origin/main` the PP1 orchestrator now passes `accepted.wav` to Vishwa's live pYIN inside the audible crop, whose notes go on to Pasan; so the BPDR WAV does reach teammates there (this branch is 11 commits behind).

## 3. Keeping the BPDR concept in a form that can work

The concept = **disentangle pitch evidence from boundary errors**: the output for a stretch of audio should not change when note boundaries are falsely split, merged or shifted (boundary invariance), and should move by exactly -e when a local detuning e is added (counter-detuning). Every candidate below must pass four tests before any build: (i) the target is computable from what is available at inference; (ii) a plain baseline leaves room to improve; (iii) no notes/key/tempo/chords are output as BPDR's product; (iv) it fits 15 GB RAM and the 11 Dec date.

**3.1 Candidate R1 (recommended): BPDR as a boundary-robust deviation estimator with review-only repair [P].**
- Task: for each sung span, estimate its pitch centre and its deviation from a *supplied* reference (the user's confirmation, the notes owner's notes, or a key grid from its owner). Repair is offered only for spans the user accepts, using the existing bounded renderer, rollback and provenance.
- Why it passes test (i): detuning equivariance "shift the audio by e, the estimate moves by e" needs no knowledge of intent, so it can be trained and checked on any audio without anchors. That is also where the response terms carry information the anchor does not (2.4).
- Where the novelty lives: the estimate takes boundaries from the extractor, which fragments vibrato and merges repeated notes (its own warning). The claim becomes: *a BPDR-trained estimator's per-span deviation changes less under false split/merge/jitter than (a) a span median, (b) an augmentation-only model, (c) a frame-only model, at equal accuracy.* The three matched arms and the loss stay; the model family is free (a small Transformer, GRU or TCN; pick by a declared rule).
- **Go/no-go before building (test ii), no training, minutes:** on the 498 existing arrays, measure how far the plain span-median deviation moves under the saved split/merge/jitter variants. Declared rule: per-span metric = absolute change of the span-median deviation (cents) between the original and the altered boundaries, singer-macro mean over the three kinds; **go if it is at least 5.0 c, no-go below 5.0 c** (5 c is about the size at which the downstream note label starts to be at risk, see 2.3). Report the value either way.
- Honest cost: this changes the hypothesis wording from "repairs more stably" to "measures more stably, repairs on review". **Supervisor question**, not ours to settle. Read proposal sections 1.3 and 3 with the supervisor before adopting.

**3.2 Candidate R2 (fallback, always done): report the original hypothesis as an honest negative [P].** Finish one fair comparison: a control whose positive-anchor weight matches BPDR's early effective weight, declared, same data/seed/budget; require the 11-of-12 fit-health rule; report harm at matched coverage. Whatever comes out, sections 2.1-2.4 are the result: the proposal allows a negative outcome. Do not run further gate/schedule variants.

**3.3 Not candidates.** Reference-free silent repair of short notes (fails test i); more architectures or loss weights (2.1); semitone-snap as a reference (falsely flags 33% of natural spans).

## 4. Making the component strong: what to add (by what it needs)

Principle: a **useful outcome for every input**, not usable audio from every input. The proposal excludes melody from speech, clipping recovery and multi-voice separation.

| Input | Outcome the component must give |
|---|---|
| Clean sung/hummed phrase | Accept unchanged, evidence attached |
| Noisy but trackable | Accept; optional cleanup only if pitch evidence is preserved; say what was measured |
| Too quiet / too loud / clipped | Located re-record guidance with time ranges; no accepted path when clipped |
| Longer than 20 s, leading silence | Offer the best <=20 s phrase and trim silence (user confirms) |
| Speech | "This is speech"; optional creative mode, labelled outside the research claim |
| Silence / corrupt file | Technical failure with reason |
| Several voices / music behind | Warn; do not separate |

**Tier 1: on disk now, no training, no new data (do first)**
1. **Hear-back confirmation loop (new) [P].** Play back what the shared extractor heard from the accepted WAV (synthesised, read-only call; the Studio already makes this audio) and ask "is this your melody?". A yes/no tells us *where* the user disagrees, not the correct pitch, so a rejected span leads to a located re-record request; a repair target needs the user to pick the note or a reference from its owner (R1). Limit: with today's extractor at about 0.29 note F1, most correct takes would be played back wrong, so this is only useful after the frame-length change (item 3) or with the pitch-evidence track. Test with a few listeners before claiming value.
2. **Evidence-based re-record guidance.** Test whether `quality_report` measures (SNR estimate, voiced ratio, confidence, octave jumps, level) predict per-file extractor F1 on the existing 63 development cases. Keep only measures that do; declare the test first. Then the guidance has evidence behind it.
3. **Frame-length hand-off to the notes owner.** Give Vishwa the section 15.1 table as a recommendation. First check, through the read-only adapters, whether the changed notes change Pasan's/Rahul's chords on the demo phrases; if chords do not change, say so and do not lead with it.
4. **Phrase selection and trimming** (table above), **room check** (build plan F2a; also supplies the voice-free noise reference the proposal requires), **hand-off contract test** (accepted-WAV SHA-256 equals the file the extractor reads), **session-audio retention/delete rule**.
5. **R1 go/no-go measurement** (3.1) and **merge `origin/main`** into the branch (owner's decision) so the Studio is tested against the live orchestrator.

**Tier 2: needs recordings, ethics or downloads (ask the owner first)**
6. Confirmation set: pilot recordings (8 adults x 6 phrases) and/or HumTrans M02/M03, opened once after thresholds are frozen.
7. **Best-take selection:** record 2-3 takes, hand on the one with the best predicted extractability (uses item 2); two-take agreement as a confidence signal (build plan F5).
8. Tap-the-rhythm boundary hints (F3): attacks the same split/merge problem from the user side; simulated-tap test first.
9. Voice-memo triage (F4); speech-detector re-validation on more speech.

**Tier 3: needs a teammate or supervisor**
10. Reference contract for R1: notes back from the notes owner and/or a key grid with confidence from its owner; BPDR only consumes them.
11. Pitch-evidence sidecar for noisy input. PESTO minus pYIN-2048, onset+pitch F1, singer-level 95% interval [V, from `frame_control.json`]: clean +0.024 [-0.016, +0.057] (no clear difference); 10 dB +0.049 [+0.005, +0.086]; 0 dB +0.086 [+0.042, +0.112]. So after the frame change the sidecar still helps in noise, which is this component's core case; offer both to the notes owner.
12. Supervisor decision on the R1 wording and on a headline of "do-no-harm preparation + confirmation loop + boundary-robust deviation measurement + honest negative on blind repair".

## 5. Order of work to 11 Dec (15 GB machine: one heavy job at a time, run from the owner's own terminal)

| Window | Work |
|---|---|
| 13-17 Nov | R1 go/no-go (no training); quality-predicts-F1 test; chord-change check for the frame finding; supervisor question on R1 wording |
| 18-24 Nov | Hear-back loop in the Studio; phrase trim; room check; retention rule; hand-off contract test; browser test of upload/microphone |
| 25 Nov-4 Dec | If R1 is go: declare and run the three-arm estimator comparison (small models). In any case: R2 weight-matched control run once |
| 5-11 Dec | Freeze thresholds; open the confirmation set once; write the result; **11 Dec decision** |

## 6. Do not claim

That the Transformer repairs pitch; that BPDR beat the controls; that R1 works (untested); that the frame-length gain holds beyond 21 development files; that the component rescues any input.

## 7. Results of the first Tier 1 checks (13 Nov 2026, development data, declared before running)

| Check | Result | Verdict |
|---|---|---|
| R1 go/no-go (`tools/probe_r1_span_median_shift.py`, `results/studio/evidence/r1_span_median_shift.json`) | Plain span-median deviation (cents from nearest semitone, A440 grid) moves 1.76 (split), 0.90 (merge), 8.65 (jitter); macro mean **3.77 c** over 63 development records, 12 groups. Raw pitch-centre variant 6.97 c. | **NO-GO** by the declared 5.0 c rule on the deviation metric. Driven almost entirely by jitter; the plain median is already near-invariant to split/merge. Not run on the 498-record set (not located); metric choice decides the verdict, so treat as provisional. |
| Quality measures predict F1 (`tools/probe_quality_predicts_f1.py`, `quality_predicts_f1.json`) | 63 cases (21 sources x clean/10 dB/0 dB), mean F1 0.44. Only `voiced_ratio` met the rule (rho -0.40, interval [-0.59, -0.01]) and its sign is wrong for guidance (noise artefact). SNR, confidence, level, noise floor, clipping: not kept. Octave jumps constant 0. | Do not present SNR/confidence as accuracy predictors; show them as plain measurements. Clipping/level guidance rests on physical facts, not on this test. |
| Chord change from frame length (`tools/probe_chord_change_frame.py`, `chord_change_frame.json`) | 21 phrases: chords differ in 40.3% of samples between pYIN 4096 and 2048 notes. Agreement with chords from reference notes 47.0% -> 56.4%, difference +9.4 points, singer-level interval [-0.5, +21.7]. | Chords do change; benefit direction favourable but interval includes 0, so do not lead with it. Reference chords come from the same models fed reference notes, not annotated chords. |
| Hand-off contract test (`tests/test_studio.py`) | The file the extractor is called on equals the reported accepted WAV and its SHA-256. Passes. | Done (pipeline contract; the extractor's own file reading is not covered). |

Consequence for the headline: with R1 NO-GO on this measurement, R2 (honest negative on blind repair) is the safer headline. The R1 wording is still the supervisor's decision.

## 8. Retention rule and room check (13 Nov 2026)

**Retention (closes defect 15.2(e)).** `studio/retention.py`: session folders (32-hex names only, direct children of `results/studio/sessions`, symlinks skipped) older than 24 h are deleted each time the Studio loads; upload and microphone modes show the rule and a "Delete my recording and results now" button (input, output, speech and room folders). Test added (expired removed; fresh, non-session and outside paths untouched). Not yet exercised by clicking in a real browser; the existing 13 test session folders (13 MB, generated audio) will be purged on the next start once older than 24 h.

**Room check (build-plan F2a).** `studio/room.py` measures a 2-3 s silent clip (level, flatness, 50/60 Hz hum, sudden events) and returns good / noisy / too_loud / unusable with one tip; a usable clip becomes the cleaner's noise reference (`clean.spectral_gate(noise=...)`), and `noise_profile_policy` is recorded in the result and sidecar. Tests: synthetic quiet, 50 Hz hum, fan noise, knock, very loud, too short; policy recorded. Thresholds (-55/-40 dB, event range 12 dB, hum 12 dB) are preliminary and untested on real rooms.

**Declared comparison (`tools/probe_room_profile.py`, `room_profile_comparison.json`):** 21 sources x 10/0 dB, room-check clip vs quietest-frames profile. F1 difference after cleaning +0.002 [-0.002, +0.006]; F1 the Studio hands on 0.000 [-0.001, +0.001]. The cleaned version is chosen in 3/42 cases with the room clip and 0/42 without, with no F1 change. The room clip was synthetic (same noise file, same gain), the best case for the room check.

Consequence: describe the room check as located advice about the room plus the proposal's voice-free noise reference, **not** as an accuracy gain. Still open: real-browser upload/microphone/delete test, listening review, hear-back loop, phrase trim, best-take selection, commit decision, Notion logbook row.

## 9. R1 go/no-go rerun on the real 498 records, with non-learned baselines (13 Nov 2026)

`tools/probe_r1_baselines_498.py`, `results/studio/evidence/r1_baselines_498.json`. Correction to section 7: the 498 records are in `2026-11-04_v64_reliable_melodic_pairs` (53 originals + 445 pairs, 20 groups, 0 skipped); the 63-record figure came from a different folder. Pitch = `fine`, frames = `reliable`.

| Baseline (mean abs change in cents under the saved boundary variants, group-macro) | split | merge | jitter | macro |
|---|---|---|---|---|
| Plain median, deviation from semitone | 1.75 | 1.15 | 5.90 | **2.93** |
| Plain median, pitch centre | 2.14 | 3.00 | 9.65 | 4.93 |
| Trimmed +-60 ms median, deviation | 1.99 | 1.84 | 7.74 | 3.86 |
| Confidence-weighted median, deviation | 1.75 | 1.21 | 5.97 | 2.98 |

By the declared rule (macro >= 5.0 c) this is still **NO-GO** on both metrics (2.93 and 4.93). Trimming and confidence weighting do not reduce jitter sensitivity (trimming makes it worse), so the headroom is not removed by a simple baseline. The only headroom is under **jitter** (about 6 to 10 c); split and merge sit at about 1 to 3 c. A learned estimator can therefore at most show a stability benefit under boundary jitter, of a few cents. No fit was started.

## 10. R1 note-centre estimator: three-arm result (13 Nov 2026, development data)

Declared first in `results/studio/r1/declaration.json` (TCN substituted for a GRU before any fit, for CPU time). `tools/fit_r1_estimator.py`, `tools/analyze_r1_fits.py`, `results/studio/r1/fit_results.json`. 35 training-role Vocadito windows, 14 groups, 7 leave-group-out folds, 3 seeds, 63 fits, all healthy (100%). A first run lost its results to a JSON-save bug of mine and was rerun unchanged (seeded).

| Arm | MAE (cents, group-macro) | change under split / merge / jitter |
|---|---|---|
| frame_only | 25.56 | 0 / 0 / 0 (by construction) |
| augmentation_only | 25.30 | 0.17 / 0.36 / 2.33 |
| bpdr | 25.13 | 0.09 / 0.17 / 0.94 |
| Non-learned: frame pitch itself | 24.1 | |
| Non-learned: plain median, extractor boundaries | 23.8 | |
| Non-learned: rolling median 0.2 s | **17.9** | |
| Oracle true-boundary median (not available at inference) | 6.9 | |

bpdr minus frame_only: -0.43 c [-0.71, -0.14]; bpdr minus augmentation_only: -0.17 c [-0.42, +0.05]. **The declared win condition is not met**: no learned arm beats the rolling-median baseline (it is about 7 c better than all of them) or even the raw frame pitch. BPDR's jitter change is 40% of augmentation-only's (meets the 70% criterion) but at an accuracy no better than doing nothing, so it is not a useful result. The differences between arms (0.2 to 0.4 c) are negligible next to the 7 c gap to the simple baseline.

Reading: with 35 windows from 14 singers the model learns almost nothing beyond the input pitch; the headroom to the oracle (about 17 c) is segmentation information the model does not recover. This is development-scale evidence, not proof that no model could; it does show the three-arm design with this data does not make a working estimator. The training-free 0.2 s rolling median is a stronger and simpler result than any fitted arm and needs a check on fresh data before it is mentioned to the extractor's owner.

## 11. More data: HumTrans note-centre headroom check (13 Nov 2026)

Collected 666 HumTrans clips (official TRAIN, plain recordings; F01-F05, M01 training role, 120 each or fewer; M02 development, 100) by HTTP range reads (`tools/collect_humtrans_r1.py`, `datasets/humtrans_r1/`, declaration written first). M03, M04, M05, VALID, TEST untouched. The MIDI labels are score-like (notes start at 0.0 s, many clips transposed by whole semitones), so each clip was aligned by DTW with a transposition search (`tools/align_humtrans_r1.py`); 589 of 666 kept (mean abs error after tuning offset <= 60 c, >= 50% frames matched; 115/120, 120/120, 69/72, 85/88, 35/46, 101/120 train; 64/100 development).

Baselines on the kept clips (mean abs error vs the intended-pitch grid, tuning offset removed, person-macro; `tools/fit_r1_humtrans.py` loader):

| Estimator | train persons | development M02 |
|---|---|---|
| Frame pitch itself | 24.4 c | 38.5 c |
| Plain median, extractor boundaries | 20.4 c | 35.4 c |
| Rolling median 0.2 s | 22.1 c | 36.6 c |
| Oracle median over true note runs (not available at inference) | **20.1 c** | 34.3 c |

**Finding: there is no headroom on this target.** The plain extractor-boundary median is already within 0.3 c of the oracle, and the whole gap from raw frame pitch to the oracle is about 4 c. The remaining ~20 c is the singers' own deviation from the equal-tempered grid, which is the quantity BPDR wants to measure, not remove. A Transformer therefore cannot show a meaningful accuracy gain on note-centre estimation here; the Vocadito headroom (17 c) came from performed-pitch annotation at that smaller scale. The declared three-arm HumTrans fit (`declaration_humtrans.json`, about 3 h) was NOT run because the oracle bound makes any win under the declared bar impossible by construction. Caveat: the reference is DTW-derived and the clips are humming from only 6 + 1 people.

## 12. Boundary model result (13 Nov 2026, development data; declared first in `results/studio/r1/declaration_boundary.json`)

Output is a per-frame note-boundary probability only (sidecar hint); notes stay with the extractor owner. `tools/fit_r1_boundary.py` (resumed after a memory kill: 34 + 8 fits, 42 total, all healthy), `tools/analyze_r1_boundary.py`, `results/studio/r1/boundary_fit_results.json`. Trained on HumTrans training persons only (525 clips, DTW-derived onset labels); decisive test on 39 Vocadito windows from 14 singers with independent annotations.

Onset F1 at 50 ms, Vocadito, singer-macro: extractor boundaries **0.601** -> frame_only 0.680, augmentation_only 0.698, **bpdr 0.688**. bpdr minus extractor +0.087, singer-level 95% [+0.036, +0.137]. Under corruption (split / merge / jitter) the extractor falls to 0.596 / 0.594 / 0.538 while bpdr stays 0.690 / 0.688 / 0.685. All five declared criteria pass. On HumTrans held-out persons: 0.670 -> 0.855 (inflated by label coupling, reported only); development person M02: 0.548 -> 0.782.

**What this does and does not show.** A learned boundary model improves on the extractor's own boundaries on independent labels, and it is insensitive to false splits, merges and jitter. But the BPDR-specific terms add nothing measurable: bpdr minus frame_only +0.008 [-0.004, +0.022] (no difference), bpdr minus augmentation_only -0.010 [-0.019, 0.000] (slightly worse). frame_only is stable under corruption by construction and as accurate, so the result supports "a learned boundary model helps" and does not support "BPDR's loss terms are what helps". Criterion (4) passes mainly because every arm's drop is tiny. Limits: trained on humming, tested on singing; TCN not Transformer; boundary onsets only, not yet passed through the extractor; development evidence; thresholds chosen on training clips.

## 13. Boundary end-to-end and Transformer evaluation (15 Nov 2026, development data; declared first in `results/studio/r1/declaration_e2e.json`, amended with FastTNet)

Follow-up to Section 12 testing three declared questions (`tools/fit_r1_e2e.py`, `tools/analyze_r1_e2e.py`, `results/studio/r1/e2e_fit_results.json` with 18 complete fits across TCN and FastTNet Transformer architectures, full and lean feature sets, on 39 independent Vocadito windows across 14 singers with 1,000 bootstrap resamples):

1. **Part 1: End-to-end Note F1 (Model vs Extractor).**
   - **Clean Audio:** Model boundaries fed into downstream note formation achieved **0.535 [0.460, 0.609]** Note F1 (TCN full bpdr) vs **0.562 [0.489, 0.635]** for the extractor's own boundaries (diff **-0.027** [-0.057, +0.005], 95% CI includes 0). **Primary win bar not met on clean notes.**
   - **Onset Robustness:** On Onset F1 at 50 ms, the model decisively beats the extractor baseline across all conditions: clean **0.688 vs 0.601** (+0.087 [+0.038, +0.140]), split (+0.094), merge (+0.094), and jitter (+0.147 [+0.089, +0.206]).
   - **Corruption Robustness:** Under temporal jitter, extractor note F1 collapses to 0.479, while model boundaries maintain **0.529** Note F1 (diff **+0.050** [+0.018, +0.084], 95% CI strictly excluding 0).

2. **Part 2: Transformer vs TCN — ADOPT TRANSFORMER.**
   - The initial compact Transformer (`TNet`) underperformed due to downsampling without a 20 ms skip highway, rigid absolute positional embeddings, and lack of multi-scale pitch-step deltas.
   - The amended **FastTNet** architecture (stride-4 dilated residual highway + multi-scale pitch step deltas + depthwise conv positional encoding + fast pre-norm attention, width 32, 2 layers, 4 heads, 250 steps) resolved all three bottlenecks.
   - **Vocadito Onset F1 at 50 ms:** FastTNet significantly outperforms TCN across multiple arms with singer-level 95% CI strictly excluding 0:
     - `full frame_only`: **0.719 vs 0.680** (diff **+0.039** [+0.012, +0.062]) -> **ADOPT**
     - `full bpdr`: **0.715 vs 0.688** (diff **+0.026** [+0.006, +0.047]) -> **ADOPT**
     - `lean frame_only`: **0.713 vs 0.687** (diff **+0.026** [+0.001, +0.052]) -> **ADOPT**
   - **Vocadito Note F1:** FastTNet delivers substantial Note F1 gains over TCN across all arms (+0.032 to +0.079; e.g. `full frame_only`: **0.601 vs 0.522**, diff **+0.079** [+0.043, +0.116]; `full bpdr`: **0.577 vs 0.535**, diff **+0.042** [+0.008, +0.074]).
   - **Decision: ADOPT TRANSFORMER.** FastTNet is adopted as the primary boundary model checkpoint (`results/studio/r1/boundary_transformer.pt`).

3. **Part 3: BPDR loss ablation with lean features across corruptions.**
   - Under lean features (pitch, reliable flag, confidence only; forcing reliance on the boundary track without energy or onset shortcuts), average Onset F1 across corruptions (split, merge, jitter) is:
     - TCN: `bpdr` = 0.687 vs `frame_only` = 0.687 (diff -0.000 [-0.015, +0.013]) and vs `augmentation_only` = 0.690 (diff -0.003 [-0.011, +0.006]).
     - Transformer: `bpdr` = 0.707 vs `frame_only` = 0.713 (diff -0.006 [-0.033, +0.021]) and vs `augmentation_only` = 0.706 (diff +0.000 [-0.010, +0.012]).
   - **Verdict: BPDR custom loss terms do not statistically improve onset F1 over controls when the track is needed.** Training-time augmentation or frame-only training achieves equivalent accuracy without the complexity of consistency regularization loss terms.

**Consequence for Studio delivery:**
- **Boundary Hints Integration:** Integrated FastTNet boundary hint inference (`source/bpdr_input/studio/boundary.py`) into the Studio pipeline (`pipeline.prepare(with_boundary_hints=True)`), outputting per-frame onset probabilities and peak locations into `sidecar.json`.
- **Ownership & Usability Discipline:** While FastTNet boundaries achieve 0.719–0.741 Onset F1 and 0.577–0.601 Note F1 (beating extractor on onsets and jitter), final symbolic note formation remains owned by downstream components. Studio maintains Tier 1 interactive usability (user hear-back loop with pitch synthesis, automated silence trimming, and phrase selection) while handing off high-quality boundary evidence in the sidecar.

## 14. Multi-version FastTNet check, Studio parity, and real-audio Vocadito benchmark (16 Nov 2026)

Update to Section 13 following multi-seed/multi-version verification (`v1`–`v4`), Studio training–inference parity fixes, and real-audio end-to-end benchmarking (`tools/analyze_r1_e2e.py`, `results/studio/r1/statistical_analysis_report.md`, `tools/benchmark_studio_vocadito.py`, `results/studio/r1/studio_vocadito_benchmark.json`). Reserved splits (`HumTrans M03–M05`, `VALID`, `TEST`) remain untouched.

### 14.1 FastTNet multi-version & multi-seed results [V]

- **Unreported `v1` Note F1 win over Extractor (`e2e_fit_results.json`, seed `339`):** Section 13 reported the TCN clean Note F1 miss (`0.535` vs `0.562`, diff `-0.027 [-0.057, +0.005]`), but re-analysis in `tools/analyze_r1_e2e.py` shows `FastTNet` `v1` (`full frame_only`) beats the Extractor on Vocadito clean Note F1 (**`0.601 [0.538, 0.665]`** vs `0.562 [0.489, 0.635]`, diff **`+0.039 [+0.006, +0.072]`**) with singer-macro 95% CI > 0 across all four boundary conditions (`split +0.042 [+0.011, +0.073]`, `merge +0.047 [+0.019, +0.078]`, `jitter +0.122 [+0.083, +0.158]`), plus clean Onset F1 (`0.719` vs `0.601`, `+0.118 [+0.067, +0.170]`).
- **`v1` 2-seed replication (`[339, 340]`):** `transformer_v1_rep` (`seed=340`, `results/studio/r1/transformer_v4_results.json`, 2/2 healthy, both hitting `thr = 0.80` ceiling) gives `full frame_only` Vocadito clean Note F1 `0.579 [0.516, 0.646]`, Onset F1 `0.714 [0.645, 0.779]`, and `dev_M02` `0.758 [0.730, 0.785]` (`full bpdr`: Note F1 `0.561`, Onset F1 `0.692`, `dev_M02` `0.766`). Combining seeds `339` and `340` (`transformer_v1_2seed`) yields `full frame_only` Vocadito clean Note F1 **`0.590 [0.528, 0.654]`**, Onset F1 **`0.716 [0.645, 0.781]`**, and `dev_M02` `0.756 [0.728, 0.783]` (`full bpdr`: Note F1 `0.569`, Onset F1 `0.703`, `dev_M02` `0.767`).

| Version | Source File | Fits (Healthy, Seeds) | `thr` Grid Selected | `dev_M02` Clean Onset F1 | Vocadito Clean Onset F1 (`full`) | Vocadito Clean Note F1 (`full`) | Decision vs `v1` |
|---|---|---|---|---|---|---|---|
| `v1` (`1-seed`) | `e2e_fit_results.json` | 6/6 (`339`) | `0.8` (saturated) | `0.754` (`frame_only`), `0.767` (`bpdr`) | `0.719` (`frame_only`), `0.715` (`bpdr`) | **`0.601`** (`frame_only`), `0.577` (`bpdr`) | Default baseline (`V1_CHECKPOINT`) |
| `v1_2seed` | `e2e_fit_results.json` + `transformer_v4_results.json` | 4/4 `full` (`339, 340`) | `0.8` (saturated) | `0.756` (`frame_only`), `0.767` (`bpdr`) | `0.716` (`frame_only`), `0.703` (`bpdr`) | `0.590` (`frame_only`), `0.569` (`bpdr`) | Confirms `v1` across seeds |
| `v2` | `transformer_v2_results.json` | 12/12 (`339, 340`) | `0.55..0.75` | `0.807` (`frame_only`), **`0.812`** (`bpdr`), `0.814` (`lean bpdr`) | `0.699` (`frame_only`), **`0.724`** (`bpdr`) | `0.563` (`frame_only`, `-0.038`), `0.577` (`bpdr`) | Keep `v1` (`frame_only` Note F1 trails `v1`) |
| `v3` | `transformer_v3_results.json` | 12/12 (`339, 340`) | `0.6..0.85` | `0.788` (`frame_only`), `0.794` (`bpdr`) | `0.693` (`frame_only`, `-0.026`), `0.680` (`bpdr`) | `0.537` (`frame_only`, `-0.064`), `0.531` (`bpdr`) | **DO NOT ADOPT** |
| `v4` (`POS_WEIGHT=10.0`, `offset=0.0`) | `transformer_v4_results.json` | 4/4 (`339, 340`) | `0.8, 0.85, 0.9` | `0.788` (`frame_only`), `0.795` (`bpdr`, `+0.028` vs 2-seed `v1`) | `0.698` (`frame_only`), `0.662` (`bpdr`, `-0.041` vs 2-seed `v1`) | `0.544` (`frame_only`, `-0.046` vs 2-seed `v1`), `0.515` (`bpdr`) | **Keep `V1_CHECKPOINT` as Studio default** |

- **Why `v2`/`v3`/`v4` improved `dev_M02` humming (`0.788–0.814`) while trailing `v1` on Vocadito singing:** `v1` adds `eval_logit_offset = 0.9` during eval and picks `thr = 0.80`, which corresponds to an effective raw-logit probability threshold of `sigmoid(logit(0.80) - 0.90) ≈ 0.619` (~`0.62`). With `eval_logit_offset = 0.0` and an expanded grid (`0.20..0.95`), `v4` selected much stricter raw probability thresholds (`0.80–0.90`) on HumTrans `tr[:80]`. Stricter thresholding raised precision on HumTrans `dev_M02` humming (`0.788–0.814`), but under-predicted softer/vibrato Vocadito singing note boundaries relative to `v1`.

### 14.2 Studio training–inference parity fixes and UI wiring [V]

- **`source/bpdr_input/studio/boundary.py` parity fixes:**
  - Enforced the canonical **20 ms (50 Hz) frame grid** (`CANONICAL_FRAME_SECONDS = 0.020`, `canonical_hop_samples`, `align_track_to_20ms` using nearest native-frame lookup to avoid interpolating across unvoiced gaps).
  - Matched `source/bpdr_input/audio/analysis.py` acoustic cues (`_extract_acoustic_cues`: z-scored STFT log-energy in dB + **64-band log-mel positive flux**), fixing the prior feature mismatch (`max_abs_diff = 0.0` vs `analysis.py` verified).
  - Added `provisional_boundary_score` and `provisional_boundary_edges` (`0.0` max abs diff against `analysis.py` `raw[:, 70]` on a matched grid); `extract_boundary_hints` and `predict_boundary_probabilities` now automatically compute `provisional_20ms` edges whenever `boundary_edges is None` and `nin in (4, 6)`, enabling `nin=6` BPDR checkpoints (`V2_BPDR_CHECKPOINT`, `boundary_transformer_v4_bpdr.pt`) seamlessly alongside `nin=5`.
  - Registered `V2_CHECKPOINT`, `V2_BPDR_CHECKPOINT`, and `V3_CHECKPOINT`, while keeping **`DEFAULT_CHECKPOINT = V1_CHECKPOINT`** (`results/studio/r1/boundary_transformer.pt`, `thr = 0.80`, `DEFAULT_EVAL_LOGIT_BIAS = 0.9`) based on Sections 14.1 and 14.3.
- **`source/bpdr_input/studio/pipeline.py` & `source/bpdr_input/application/studio_app.py` UI wiring:**
  - `studio_app.py` invokes `pipeline.prepare(..., with_boundary_hints=True)`, attaching canonical 20 ms `boundary_hints` to `sidecar.json` without modifying `plots.py`.
  - Piano rolls overlay advisory dashed vertical boundary-peak lines (`_annotate_boundary_markers`), and `_boundary_hints_block` renders an expander showing detected boundary count, threshold, `20 ms` grid step, and peak times.

### 14.3 Real-audio Studio Vocadito benchmark [V]

Evaluated the actual Studio pipeline (`evidence.load_mono` 10 s crop -> `clean.spectral_gate` + `evidence.pesto_track` + `clean.choose` -> `boundary.extract_boundary_hints` -> `make_notes` / `evidence.notes_from_track`) on all **39 Vocadito training-role windows** across **14 singers** (`tools/benchmark_studio_vocadito.py`, `results/studio/r1/studio_vocadito_benchmark.json`; 1,000-resample singer-macro 95% bootstrap CIs, seed `339`):

| Method | `nin` | `Thr` | Note F1 (Mean [95% CI]) | Onset F1 (Mean [95% CI]) | Diff vs `pyin_4096` (Note F1) | Diff vs `pesto_notes` (Note F1) | Diff vs `studio_v1` (Note F1) |
|---|---|---|---|---|---|---|---|
| `pyin_4096` | - | - | 0.301 [0.256, 0.352] | 0.419 [0.360, 0.483] | - | **-0.185** [-0.239, -0.131] | **-0.330** [-0.395, -0.257] |
| `pyin_4096_prepared` | - | - | 0.301 [0.256, 0.352] | 0.419 [0.360, 0.483] | **+0.000** [+0.000, +0.000] | **-0.185** [-0.239, -0.131] | **-0.330** [-0.395, -0.257] |
| `pesto_notes_from_track` | - | - | 0.486 [0.431, 0.546] | 0.617 [0.572, 0.664] | **+0.185** [+0.131, +0.239] | - | **-0.145** [-0.201, -0.087] |
| **`studio_v1`** | 5 | 0.80 | **0.631 [0.585, 0.676]** | **0.767 [0.732, 0.799]** | **+0.330** [+0.257, +0.395] | **+0.145** [+0.087, +0.201] | - |
| `studio_v2` | 5 | 0.65 | 0.591 [0.534, 0.643] | 0.728 [0.689, 0.766] | **+0.289** [+0.209, +0.367] | **+0.104** [+0.032, +0.176] | **-0.040** [-0.072, -0.010] |
| `studio_v2_bpdr` | 6 | 0.60 | 0.617 [0.561, 0.673] | 0.746 [0.706, 0.787] | **+0.315** [+0.233, +0.394] | **+0.130** [+0.057, +0.200] | **-0.014** [-0.036, +0.010] |
| `studio_v4` | 5 | 0.85 | 0.576 [0.529, 0.628] | 0.736 [0.702, 0.770] | **+0.275** [+0.200, +0.349] | **+0.090** [+0.032, +0.146] | **-0.055** [-0.096, -0.012] |
| `studio_v4_bpdr` | 6 | 0.90 | 0.534 [0.494, 0.579] | 0.705 [0.674, 0.736] | **+0.233** [+0.170, +0.298] | **+0.048** [-0.010, +0.095] | **-0.097** [-0.132, -0.061] |
| `studio_v1_pre_parity_100hz_thr05` | - | - | 0.587 [0.541, 0.638] | 0.751 [0.719, 0.785] | **+0.285** [+0.232, +0.336] | **+0.100** [+0.081, +0.121] | **-0.044** [-0.083, -0.003] |

- **Benchmark takeaways:** Under the real Studio audio path, `studio_v1` improves Note F1 by **`+0.330 [+0.257, +0.395]`** over `pyin_4096` (`0.301`) and **`+0.145 [+0.087, +0.201]`** over `pesto_notes_from_track` (`0.486`). Restoring `20 ms` / `64-mel` / `thr=0.80` parity (`studio_v1`) beats the pre-parity 100 Hz / `thr=0.50` path (`0.587`) by **`+0.044 [+0.003, +0.083]`** Note F1. No newer checkpoint (`v2`, `v2_bpdr`, `v4`, `v4_bpdr`) beats `studio_v1`, confirming **`V1_CHECKPOINT` remains the Studio default**.

### 14.4 Test status [V] and remaining open items [P]

- **Full verified test suite [V]:** `PYTHONPATH=source;. ..\.venv\Scripts\python.exe -m unittest tests/test_studio.py tests/test_analysis.py` -> **37/37 tests passing (`Ran 37 tests in 12.635s — OK`: `29` in `tests/test_studio.py`, `8` in `tests/test_analysis.py`)**, with exact byte-for-byte parity verified between `tools/analyze_r1_e2e.py` and `results/studio/r1/statistical_analysis_report.md` and protected declaration/result/reference SHA-256 hashes unchanged.
- **Remaining open items [P]:**
  1. Real-browser manual check of microphone recording, audio upload, and immediate session deletion flows.
  2. Candidate R2 (Section 3.2): execute one declared weight-matched control run (matching BPDR's early effective positive-anchor gradient weight) and report the honest negative at matched coverage.
  3. Merge `origin/main` into this branch and test the Studio end-to-end against the live PP1 orchestrator.
  4. Teammate/supervisor handoff (`accepted.wav` + `sidecar.json` contract with advisory `boundary_hints`, plus honest-negative framing).
  5. Freeze all thresholds before opening the confirmation set (`5–11 Dec` window -> **`11 Dec` decision**).

## 15. Candidate R2 weight-matched control result (18 Nov 2026) [V]

Final honest-negative control check requested by Section 3.2 (`tools/run_r2_weight_matched_control.py`, declared first in `results/transformer_usability/2026-11-17_r2_weight_matched_control/declaration.json`, SHA-256 `359a77f662938f12e4403f7d38f88ef1ac2a7d874b8a9bc38dc924a3eb1ada14`; previous directories `2026-11-07_balanced_short_objectives/` and `2026-11-12_gate_open_short_objectives/` untouched).

### 15.1 Setup, weight-matching calibration, and 12/12 fit health [V]

- **Setup & schedule parity:** Same 4-fold singer/group split (20 groups: 14 Vocadito + 6 HumTrans training groups), `seed = 339`, `updates = 1200`, `warmup_updates = 200`, `lr = 0.0005`, `weight_decay = 0.001`, `activity_bias_init = 2.0`, 4-ULP `frame_only` duplicate-view check, and `11-of-12` fit-health rule (`final unweighted balanced anchor < 0.30`).
- **Weight-matched positive-anchor multiplier (`1.5855x`):** Measured in `results/transformer_usability/2026-11-13_loss_redundancy/results.json` (`init_gate_open`), `1 + (0.206681 * 0.920677) / (0.5 * 0.649983) = 1.5855`. Controls (`augmentation_only` and `frame_only`) apply `pos_weight = 1.0` during warmup (updates `1..200`) and `1.5855` after warmup (updates `201..1200`) to `pos_loss` in `balanced_anchor`, while `bpdr` stays at `1.0` with its response terms added after warmup.
- **Fit health (`fit_receipt.json` & `integrity_receipt.json`):** **12/12 healthy fits (`fit_health_passed = True`)**, up from **3/12** on 7 Nov (Section 2.4) and **6/11** on 12 Nov (`2026-11-12_gate_open_short_objectives/`). All 12 folds reached final unweighted anchor `< 0.30` (`augmentation_only`: `0.0680, 0.2305, 0.1755, 0.1353`; `frame_only`: `0.1110, 0.2262, 0.2330, 0.1060`; `bpdr`: `0.1637, 0.1707, 0.1959, 0.1024`; runtime `164.0–211.0 s` per fit; `12` checkpoint reloads and `4,482` prediction arrays replayed deterministically, `results.json` SHA-256 `5119245d7420af1e522eeb5503ef75112d286c257a2a934225355c51f267b7e4`). Matching post-warmup positive-anchor gradient weight made all control fits train fairly without plateauing at zero output.

### 15.2 Matched-coverage results, frozen gates, and honest negative [V]

Evaluated from `results/transformer_usability/2026-11-17_r2_weight_matched_control/results.json` (`frame_only` zero-output / no-correction reference on `2026-11-07_balanced_short_objectives/results.json`: `24.906 c`, ~25 c injected error):

| Arm | Raw Positive MAE (c, 17-grp macro) | Raw Original False Correction (c, max grp) | Raw Gate (`<25 c`, `signed >= 0.25`, `orig <= 3 c`) | Pos. Response Error (c: split / merge / jitter / mean) | Pos. Boundary Shift (c: split / merge / jitter / mean) | Known-Zero Boundary / Counter-Detuning Error (c, 3-kind mean) |
|---|---|---|---|---|---|---|
| `augmentation_only` (`1.5855x` post-warmup) | `19.917` (`7/17` pass) | `1.153` (max `5.037`, `S29`) | **FAIL** (`False` x3) | `20.426 / 20.453 / 20.477` (mean **`20.452`**) | `0.896 / 0.076 / 0.437` (mean `0.470`) | `0.040 / 0.159` |
| `frame_only` (`1.5855x` post-warmup) | `19.234` (`9/17` pass) | `1.193` (max `4.730`, `S29`) | **FAIL** (`False` x3) | `18.747 / 18.747 / 18.747` (mean **`18.747`**) | `0.000 / 0.000 / 0.000` (mean **`0.000`**) | `0.000 / 0.153` |
| `bpdr` (`1.0x` + response terms) | `18.986` (`7/17` pass) | `1.002` (max `4.732`, `S29`) | **FAIL** (`False` x3) | `19.087 / 18.945 / 18.958` (mean **`18.997`**) | `0.410 / 0.039 / 0.284` (mean `0.245`) | `0.016 / 0.177` |

- **Frozen gates & selection rule (`advance_to_delivery = False`, `production_eligible = False`):**
  1. **Raw gates fail across all 3 arms and all 3 boundary kinds (`raw_gate_passed = False`):** On the raw uncorrupted view, positive-target MAE drops only modestly from the `24.906 c` no-correction baseline to `18.986–19.917 c`, with only `7/17` (`bpdr`, `augmentation_only`) or `9/17` (`frame_only`) scoreable groups meeting `< 25 c` and `signed_response >= 0.25`, while un-detuned original audio suffers `1.002–1.193 c` false-correction MAE (max group `4.730–5.037 c` on `S29`, exceeding the `<= 3.0 c` limit; outside-target `outside_mae_cents` mean `0.880–0.900 c`).
  2. **Matched-coverage arm comparison fails (`bpdr` loses to `frame_only`):** Once `frame_only` trains reliably (`4/4` healthy), its corrupted-view positive response error (**`18.747 c`**) is *lower* (better) than `bpdr` (**`18.997 c`**), failing the selection criterion `response['bpdr'] < response['frame_only']` even though `bpdr` cuts positive boundary sensitivity roughly in half vs `augmentation_only` (`0.245 c` vs `0.470 c`) while `frame_only` is `0.000 c` by construction. On known-zero segments, `bpdr` shows no accuracy advantage (`0.177 c` counter-detuning error vs `0.153 c` `frame_only` and `0.159 c` `augmentation_only`).
- **Conclusion (Candidate R2 complete):** At fair 12/12 fit health, blind reference-free repair of small pitch errors fails both raw delivery gates and matched-control superiority (`advance_to_delivery = False`). This confirms the Section 2.1 information limit and Section 3.2 honest negative: no further blind-repair gate/schedule variants will be run, and delivery value remains focused on Studio input preparation (`studio_v1` Note F1 `0.631` / Onset F1 `0.767`), hear-back confirmation, and teammate handoff.

### 15.3 Updated open items

- **DONE `[V]`:** Candidate R2 weight-matched control (`results/transformer_usability/2026-11-17_r2_weight_matched_control/`, `12/12` healthy, `advance_to_delivery = False`, honest negative documented).
- **DONE `[V]`:** `FastTNet v5` cumulative ablation ladder and real-audio clean/noisy Vocadito benchmark (Section 16).
- **Remaining open items `[P]`:**
  1. Real-browser manual check of microphone recording, audio upload, and immediate session deletion flows (`Run-Studio.ps1`).
  2. Merge `origin/main` into this branch and test the Studio end-to-end against the live PP1 orchestrator.
  3. Teammate/supervisor handoff (`accepted.wav` + `sidecar.json` contract with advisory `boundary_hints`, plus honest-negative framing).
  4. Freeze all thresholds before opening the confirmation set (`5–11 Dec` window -> **`11 Dec` decision**).

## 16. FastTNet v5 cumulative ablation ladder, noisy-audio benchmark, and promotion evaluation (20 Nov 2026) [V]

Follow-up to Section 14 executing the declared Model 2 improvement plan (`model2_improvement_plan.md`, declared first in `results/studio/r1/declaration_transformer_v5.json`, SHA-256 `5965e6149bb654c7eae71e303f33bbf5e9a760580318b6c08146f6bc456fca7d`). Reserved splits (`HumTrans M03–M05`, `VALID`, `TEST`) remain strictly untouched.

### 16.1 Phase 0 Error Diagnostics on `studio_v1` (`v1_error_diagnostics.json`) [V]

Evaluated `studio_v1` across all 39 Vocadito windows under clean, 10 dB, and 0 dB looped environmental noise (`datasets/environmental_noise/1-100210-A-36.wav`), paired with 64 `dev_M02` humming clips (`tools/diagnose_boundary_errors.py`):

1. **Precision vs Recall:** On Vocadito clean singing, `studio_v1` achieves Precision `0.809` and Recall `0.727`. Under 0 dB noise, Recall stays flat at `0.765` while Precision collapses to `0.619` (false positives triple from 76 to 308, driven by noise-induced spectral flux bursts).
2. **Humming vs Singing Threshold Divergence:** On humming (`dev_M02`), `studio_v1` exhibits the exact opposite balance (Precision `0.680` / Recall `0.850`), confirming that humming onsets have sharper energy discontinuities than singing, and that tuning thresholds on humming risks over-suppressing softer singing note transitions.
3. **Repeated Notes:** 41.9% of same-pitch repeated notes (67/160) are missed vs ~24% for pitch-step onsets.
4. **Oracle Note Formation:** Feeding oracle ground-truth onsets into the standard note formation yields Note F1 of `0.849` (clean) and `0.840` (0 dB), proving that the upstream boundary picker—not the note formation rule—is the true accuracy bottleneck.

### 16.2 Phase 1–2 Ablation Ladder on HumTrans 6-Fold Out-of-Fold Cross-Validation [V]

Evaluated 6 cumulative steps using 6-fold leave-one-speaker-out CV on `HumTrans TRAIN` (525 clips, 6 speakers `F01–F05, M01`), scored by mean person-macro Onset F1 across clean and noise views (`tools/fit_r1_transformer_v5.py`, `results/studio/r1/v5/ladder_results.json`):

| Step | Configuration | Best OOF `thr` | OOF Clean F1 | OOF Noisy F1 | Composite Score | Pre-declared Action |
|---|---|---|---|---|---|---|
| `a_base` | `v1` backbone, OOF grid, `offset=0.0` | 0.90 | 0.8643 | 0.8559 | 0.8601 | ADOPT (baseline) |
| **`b_feats`** | **+ FastTNetV5 16-channel singing deltas** | **0.85** | **0.8806** | **0.8768** | **0.8787** | **ADOPT (+0.0186 gain)** |
| `c_samepitch` | + Same-pitch repeat augmentation | 0.90 | 0.8605 | 0.8570 | 0.8588 | Reject (-0.0199) |
| `d_bridge` | + Synthetic vibrato & legato glide | 0.85 | 0.8769 | 0.8714 | 0.8742 | Reject (-0.0045) |
| `e_noise` | + Noise floor elevation & pitch jitter | 0.80 | 0.8803 | 0.8750 | 0.8776 | Reject (-0.0011) |
| `f_soft` | + Soft Gaussian targets (`sigma=0.8`) | 0.85 | 0.8794 | 0.8752 | 0.8773 | Reject (-0.0014) |

- **Winning configuration:** `b_feats` (`FastTNetV5`, `flags={'feats': 1}`) delivers a solid **+1.86 point composite gain** over `a_base` on out-of-fold validation.
- **Decoding Search:** Grid search over subframe parabolic peak interpolation and timing shifts in `[-30, +30]` ms confirmed `subframe=False, shift_ms=0.0` remains optimal.
- **Development Split (`dev_M02`, 64 clips, 3 seeds `[339, 340, 341]`):** `b_feats` achieved mean Onset F1 of **`0.8044` (clean)** and **`0.8035` (noisy)** vs `studio_v1` baseline of **`0.7538` (clean)** and **`0.7048` (noisy)**, representing a **+5.06 point clean** and **+9.87 point noisy** gain on humming.

### 16.3 Phase 3 Real-Audio Vocadito Benchmark & Promotion Verdict [V]

Evaluated `studio_v5` (`FastTNetV5 b_feats`, seed `339`, `thr=0.85`, `boundary_transformer_v5.pt`) on all 39 Vocadito windows across 14 singers under clean, 10 dB, and 0 dB noise (`tools/benchmark_studio_vocadito.py`, `results/studio/r1/studio_vocadito_benchmark.json` and `studio_vocadito_benchmark_noisy.json`; 1,000 bootstrap resamples):

| SNR Condition | Method | Note F1 (Mean [95% CI]) | Onset F1 (Mean [95% CI]) | Paired Diff vs `studio_v1` (Note F1) | Paired Diff vs `studio_v1` (Onset F1) | Promotion Bar Met? |
|---|---|---|---|---|---|---|
| **Clean** | `studio_v1` | **0.631 [0.585, 0.676]** | **0.767 [0.732, 0.799]** | - | - | Reference default |
| Clean | `studio_v5` | 0.576 [0.526, 0.627] | 0.731 [0.699, 0.763] | **-0.055** [-0.092, -0.021] | **-0.036** [-0.058, -0.013] | **FAIL** (CI < 0) |
| **10 dB** | `studio_v1` | **0.638 [0.586, 0.690]** | **0.783 [0.751, 0.815]** | - | - | Reference default |
| 10 dB | `studio_v5` | 0.595 [0.556, 0.640] | 0.756 [0.728, 0.783] | **-0.043** [-0.081, -0.005] | **-0.027** [-0.045, -0.008] | **FAIL** (CI < 0) |
| **0 dB** | `studio_v1` | **0.590 [0.544, 0.636]** | **0.741 [0.707, 0.775]** | - | - | Reference default |
| 0 dB | `studio_v5` | 0.569 [0.524, 0.617] | 0.730 [0.696, 0.766] | **-0.021** [-0.049, +0.006] | **-0.011** [-0.031, +0.006] | **FAIL** (CI includes 0) |

- **Verdict on Promotion:** Per the pre-declared promotion rule (`Note F1 and Onset F1 95% CI diff vs studio_v1 strictly > 0`), **`studio_v5` is NOT promoted to the Studio default**.
- **Scientific Takeaway:** The cross-domain humming-to-singing gap is asymmetric: optimizing boundary detection on HumTrans humming data (even with multi-scale deltas and noise calibration) produces models that excel on humming (`dev_M02` jumps from `0.754` to `0.804`), but are overly conservative on softer vocal tract transitions in natural polyphonic singing. `FastTNet v1` (`V1_CHECKPOINT`, `eval_logit_bias=0.9`, `thr=0.80`) remains the most balanced detector on real singing audio.
- **Decision:** **`V1_CHECKPOINT` (`boundary_transformer.pt`) remains the frozen Studio default** heading into the Dec 5–11 confirmation window. Candidate `v5` files and receipts are preserved under `results/studio/r1/v5/`.
