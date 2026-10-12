# PP1 phase execution plan

**Baseline:** 27 September 2026. **Rebaselined:** 11 October 2026. **Deadline supplied by user:** 25 October 2026.

Use alongside [the master plan](PP1_Master_Plan.md) and [the evidence tracker](PP1_Evidence_Tracker.md). This file defines build tasks and exit checks. Dates are target windows; passing a calendar date does not establish that a phase is complete. Reuse earlier implementation only after verifying it against the relevant gate.

## Phase map

| Phase | Target dates | Build result | Principal assessment evidence |
|---|---|---|---|
| P0 | 27-28 Sep | Scope, claims, team contracts and a dated baseline. | Problem clarity, design, requirements, risk planning. |
| P1 | 29-30 Sep | Working unchanged/failure paths and acoustic analysis. | Functional implementation and technology knowledge. |
| P2 | 1-4 Oct | Verified data and four-view experiment generator. | Applied ML/MIR, best practices and credible problem demonstration. |
| P3 | 5-9 Oct | Baseline models, rules and reproducible measurement. | Research implementation and fair comparison foundation. |
| P4 | 10-16 Oct | BPDR training, bounded rendering and controller. | Proposed mechanism executing and a usable proof of concept. |
| P5 | 17-21 Oct | Frozen preliminary evaluation and accepted-only integration. | Comparative evidence, preservation, design and actual progress. |
| P6 | 22-25 Oct | Verified demo, evidence pack and presentation rehearsal. | Communication/Q&A, readiness, risk mitigation and user benefits. |

Sequential critical path: **P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6**. Maintain documentation, requirements, risks and presentation examples throughout. Teammates develop their own models against agreed fixtures in parallel. No need to wait for finished planner/refinement models to begin BPDR. P2J, added on 11 October for the second data source, runs beside P3 and is not on the critical path.

**30 September status against that baseline:** The [consolidated P0 scope, claims, draft hand-off, inventory and dated decisions](P0_Scope_Claims_and_Decisions.md) establish the **local P0 gate for independent development**; teammate agreement is still open. P1 engineering is verified. P2's data/crop pipeline, review sheets and one-singer exploratory intervention pilot execute, but all 114 human-review rows remain pending and zero examples are training eligible. Thus P2's trustworthy-target exit gate has **not** passed. P3–P6 are targets, not completed phases. Preserve these dates as the original plan and record actual completion separately in the [evidence tracker](PP1_Evidence_Tracker.md).

The dependency is strict for **research training and evaluation claims**, not for writing code. P3 model/loss/metric implementation and synthetic-fixture checks may proceed while reviewers assess references; fitting on approved anchors and reporting learned comparisons must wait for the P2 review and intervention audits. Do not relabel exploratory examples as reviewed to meet a calendar target.

## 11 October status and corrective schedule

Inspected on 11 October: the current review sheet still has 114 pending rows, zero examples are training eligible, and the component has no model, loss or metric code. P2's trustworthy-target gate is open. The P3 window passed without its deliverables and P4 has not started. The critical path is about one week late with 14 days left.

The phase map above stays as the original baseline. The table below is the proposed corrective schedule; confirm it with the supervisor before treating it as agreed.

| Work | Original window | Revised target | What changes |
|---|---|---|---|
| P2 review gate | 1-4 Oct | No date: no reviewer is available on 11 Oct | Same requirement. It still blocks research fitting. The working path below applies until two reviewers are named. |
| P2 intervention coverage | Not planned | Deferred by the owner on 11 Oct | New. Under audit v0.3 only one training singer can host a shift; see below. The re-audit is automatic and needs no musician. |
| P2J second source and extractor | Not planned | 11-13 Oct | New. Runs beside P3 and is cut first if time is short. Gates J1 and J2 were checked on 11 Oct. |
| P3 controls | 5-9 Oct | 12-14 Oct | Code and fixture tests start now. Fitting waits for approved anchors. |
| P4 BPDR, controller and renderer | 10-16 Oct | 15-18 Oct | One run per learned condition under one equal, predeclared budget. |
| P5 evaluation and integration | 17-21 Oct | 19-21 Oct | Adds the GAME extractor and the natural-boundary condition. Gate J2 passed for a pilot; the melody-extraction owner's confirmation is open. |
| P6 demonstration and rehearsal | 22-25 Oct | 22-25 Oct | Unchanged. |

A new dataset arrived on 11 October: 1,604 vocal-tagged MTG-Jamendo tracks with the team's melody-extraction pipeline. The [extraction check](Jamendo_Extraction_Verification.md) shows it cannot replace vocadito, and the master plan gives it three supplementary roles. The pipeline was confirmed on a two-song sample and the GAME extractor passed its check against vocadito's annotators. It did not cause the delay and must not extend it.

**Intervention coverage.** The same checks found that the v0.3 renderer needs a reference-reliable span of at least 0.86 seconds, a steady note about 1.06 seconds long. Only 3 of the 30 training crops have one, all from one singer, so the multi-singer training subset required by P2 and P3 cannot be generated under the current policy even with every review approved. A 0.5-second requirement would admit 11 crops from 6 singers. The owner deferred this on 11 October. When it is taken up, choose one:

1. Shorten the plateau and guards, record the change as audit v0.4 on training material only, repeat the identity and realized-shift audits, and disclose the revision with every later result.
2. Keep v0.3 and report a one-singer pilot as exactly that.

Reviewers should start with crops that can host a shift under the chosen policy, since approving the others cannot produce training examples.

**Working path without reviewers (owner's position, 11 October).** No musically experienced reviewer is available, and the owner deferred the coverage decision. Zero examples are therefore training eligible, whatever happens with audits, datasets or hardware. Until two reviewers are named, PP1 work is:

1. Model, loss and metric code verified on constructed fixtures (the P3 build order).
2. Engineering runs on the audited exploratory pairs, labelled exploratory in every output.
3. The GAME extractor adapter and the natural-boundary evidence on vocadito.
4. The demonstration paths that need no trained comparison: unchanged input, controlled stops, boundary-only change and local detuning.

This is the narrower pilot the plan already allows. It shows the mechanism executing; it cannot show a reviewed comparison of the three conditions. Ask the supervisor who may count as a reviewer with relevant musical experience, since that single answer decides whether the reviewed comparison is possible before 25 October. Do not train a purported research model on unapproved anchors, from either source.

## P0. Fix the assessment scope and evidence baseline

**Build purpose:** prevent changing the research question while rushing toward a demonstration.

### Tasks

1. Use `experiments/bpdr_input` as a separate component inside the J26-DS-339 repository. The package scaffold and draft contracts now exist; keep all implementation changes inside this component folder. Record environment and configuration; keep raw vocadito files separate from generated outputs.
2. Create a short claims sheet: user problem, technical boundary-error problem, bounded gap, proposed training contribution, main comparison and limits of a controlled pilot.
3. Freeze first-version input scope: solo singing/humming, supported WAV/FLAC, at most 20 seconds, no intended-note/key entry required for BPDR, preserved time origin/rests.
4. Define all four statuses: `ACCEPT_UNCHANGED`, `ACCEPT_PREPARED`, `RERECORD_REQUIRED`, `TECHNICAL_FAILURE`. Non-accepted results have no accepted-audio path.
5. Draft versioned accepted/rejected package examples with IDs, checksums, audio metadata and reasons. Keep seconds-based edit intervals, operation logs and analyzer/policy provenance in diagnostics; model version will be recorded only after a model exists.
6. Review ownership and timing with the melody owner; settle the stale accompaniment-member description and nominate playback/export ownership. Keep an unresolved-decision register if agreement is pending.
7. Record actual work against proposal W1-W8 and establish this PP1 milestone baseline. Do not backfill unverified completion for 16-27 September.
8. Start the dataset/permission and compute inventory. Confirm the framework and analyzer can run before booking larger experiments.

### Outputs and exit gate

The consolidated [P0 record](P0_Scope_Claims_and_Decisions.md) contains the local scope, claims, actual contract examples, resource/data inventory, dated milestone register and unresolved-decision list. Together with implemented `contracts.py`, it makes BPDR's input/output responsibility and comparison question explicit, so the **local P0 gate is met**. Team acknowledgment, live integration, reviewer decisions and data-sharing permission are separate open gates; fixtures permit independent implementation meanwhile.

**Examiner-visible evidence:** one annotated architecture and one problem example, with clear ownership. These are design evidence, not a completed proof of concept.

**Contingency:** leave team adapters behind a stable interface, with pending decisions visible. Do not replace BPDR with the old harmony critic or promise a whole accompaniment research module.

## P1. Build the usable-audio and failure paths

**Prerequisite:** P0 local scope and draft contract.

### Tasks

1. Decode supported files; preserve original bytes/checksum; check empty/corrupt/non-finite data, duration and clipping observations.
2. Select/convert channels carefully; handle cancellation rather than assuming stereo averaging is safe. Create a logged working copy at the configured sample rate, initially 22,050 Hz as proposed.
3. Preserve timing origin, internal rests and duration within the documented resampling tolerance. Do not silently trim or truncate input.
4. Load frozen PESTO once; measure performed F0/confidence and extract log-mel, pitch-change, voiced mask, energy, onset and provisional boundary scores on explicit time coordinates.
5. Build controlled status handling and an accepted-package validator; distinguish a software/decode failure from a need for a better recording.
6. Add a minimal upload/playback or command entry point and aligned measured-pitch plot. Label boundary scores as provisional.

### Outputs and exit gate

Produce a runnable unchanged-audio path, diagnostics JSON, aligned feature sample, original/output playback and saved integrity/contract checks. A reviewed usable fixture can pass unchanged; bad fixtures return controlled non-accepted results; the original remains traceable. All relevant streams share documented frame/timestamp conventions.

Do not present a few accepted fixtures as a validated musical-usability classifier. At this phase, supported behavior is the tested integrity/analysis path.

**Meaningful checks:** corrupt file, silence, too-long input, invalid samples, channel cancellation, repeatable processing and downstream rejection. Confirm feature units and masks; missing pitch must not become a zero-Hz note.

**Examiner-visible evidence:** a live usable upload and a controlled stop, with operation trace. This establishes foundational implementation, not the new training advantage.

## P2. Build the experiment laboratory and trustworthy targets

**Prerequisite:** P1 analysis and time alignment.

### Tasks

1. Join vocadito audio to singer metadata, F0, A1 and A2 note references. Exclude `__MACOSX` and `._*` artifacts. Decode all source files and validate annotation values/order/bounds.
2. Assign source/singer/phrase IDs and partitions before cropping/augmentation. Keep all descendants of a singer in one partition. Keep an untouched PP1 evaluation subset and record any separate final-research holdout.
3. Have two musically experienced reviewers assess acceptable reference material; preserve disagreements/reasons and guard ambiguous transitions. Performed-pitch annotations do not prove intended pitch or justify arbitrary retuning.
4. Create documented phrase crops up to 20 seconds. Retain crop-to-original offsets and correctly transform note/F0 timestamps. For crop-edge notes, lock inclusion/matching rules and flag uncertain partial notes.
5. Implement identity rendering before local detuning. Check sample count, finite audio, clipping and pitch/expression drift.
6. Generate smooth local interventions initially at +/-10, +/-20 and +/-30 cents, restricted to reviewed reliable regions. Define the realized-shift audit tolerance first, supplement the frozen tracker with suitable reference/manual or alternative measurement evidence, and reject unsuitable transformed examples. The nominal target is used only after the waveform passes that audit.
7. Generate nominal/corrupted boundary tracks for false splits, merges and jitter. Define four aligned A/B/C/D views and the common reliable mask; explicitly control pairing across detuned/original audio.
8. Recompute acoustic features from rendered waveforms. Detuning changes both waveform and analysis; modifying only the numerical F0 input creates inconsistent examples.

### Outputs and exit gate

Produce a source/split manifest, crop map, reference-review record, intervention manifests, identity audits, generated A/B/C/D examples and exclusions. A/B have identical waveforms; C/D have identical waveforms; only boundary metadata differs within each pair. The known shift is accurate enough under the locked audit tolerance to support the intended comparison.

As of 30 September, the regenerated v0.3 views meet their identity and exploratory renderer audits, but the reference-review record contains no approvals. The 18 paired views are six detuned renderings crossed with three boundary corruptions of **one** phrase/singer; they are not 18 independent performances and cannot be used as training-ready evidence. The failed v0.1, revised v0.2 and current v0.3 audit policies remain visible in [Phase 2 verification](Phase2_Verification.md).

Declare target masks/tolerances before testing final models. Known detuning and references stay out of runtime prediction inputs. Repeated versions are not new independent recordings or singers.

**Examiner-visible evidence:** select the same audio under a false split/merge, then show a separately detuned audio view. The screen makes the two kinds of change distinguishable even before learned results exist.

**4 October decision checkpoint:** if rendering/measurement cannot resolve the small shifts, fix that chain before training at scale. If reference material is insufficient, narrow the pilot and report its limits; do not compensate with inflated augmentation counts.

At the same checkpoint, record reviewer availability and the number of independently approved **training singers and phrases**, not just review-row or augmented-view totals. Prioritize a declared multi-singer training subset and development examples. If reviews are still pending, continue constructed-fixture engineering and present the exploratory laboratory with its limits; do not train a purported BPDR research model on unapproved zero-change anchors.

## P2J. Add the second source and the team extractor

**Prerequisite:** P1 analysis and the mirrored `data/mtg-jamendo` folder. P2J is not a prerequisite for P3 and never blocks it.

**Build purpose:** obtain a real fixed extractor and a natural boundary source for BPDR, and find out with measurements whether Jamendo vocal stems can widen the training material.

### Tasks

1. Mirror the metadata and the unedited pipeline with `Setup-Jamendo.ps1`, then index the local songs with `Run-Extraction.ps1 -IndexOnly`. **Done 11 October:** 1,604 songs indexed.
2. Install the pinned tools with `Setup-ExtractionTools.ps1 -Stage Full` and run a small named sample with `Run-Extraction.ps1 -First '<ids>' -KeepStems`. Record time per song and every platform substitution. **Gate J1, done 11 October on a two-song sample:** the process works, but the separator needs about 30 seconds of computing per second of music on the development CPU, so a full-length song takes over an hour. Local runs are for short samples only.
3. Run the same frozen GAME on vocadito recordings, which need no separation. Score its notes against A1 and A2 with `mir_eval`, reporting onset-and-pitch and offset-aware results separately. Compare the raw and the published pitch with the annotated pitch and fix which field BPDR reads. **Gate J2, done 11 October** with `extraction/check_game.py` on 28 recordings of training and development singers: BPDR reads `model_midi_pitch`.
4. Cut the kept stems into mapped vocal crops of at most 20 seconds, with offsets back to the full song. Inside GAME notes, measure the share of frames where PESTO is voiced and agrees with YIN. Run the existing identity and realized-shift audits. Put each number beside the same number for vocadito. **Gate J3:** `extraction/check_stems.py` gives a first look on the sample; a decision needs stems from many more artists, produced on a GPU machine or the team's Mac.
5. Partition by `artist_id` with the existing stable-hash grouping, cap the tracks taken per artist, and exclude NoDerivatives tracks before any crop or intervention is made. One artist holds 174 of the 1,604 tracks.
6. If the voices role goes ahead, prepare the two-reviewer sheet for a declared Jamendo crop subset. Predicted notes and tracker agreement are not approvals. **Gate J4.**
7. **Done 11 October: probe of the laptop's Intel Arc GPU.** No Mac is needed. The separator package has no Intel-GPU path of its own, and its DirectML path deliberately keeps this model on the CPU, so the probe used the Intel-GPU build of PyTorch (`torch 2.14.1+xpu`) in a separate `.venv-xpu` environment. The same 10-second excerpt took 16.6 seconds against 163.5 on the CPU, and the stem differed from the CPU stem by 101 dB below signal level, which is the same result. The launcher now has a logged `-Gpu` option. A BPDR training step is about 18 times faster on the same GPU (0.73 against 13.2 seconds), so `training.py` has `--device xpu`. The NPU stays out of scope for PP1: neither model exists in a form it can run, and converting them would be a project of its own.

### Outputs and exit gate

Produce a saved extraction run with its `history.log`, a GAME-versus-annotation table on vocadito, a stem-quality table and a dated decision on each role. The extractor and natural-boundary roles need J2. The voices role needs J1, J3 and J4. Record a failed or skipped gate as failed or skipped.

**Status on 11 October:** the run history, the GAME table and a first stem table exist; see the [extraction check](Jamendo_Extraction_Verification.md). J1 holds for a two-song sample only. J2 passed for a pilot. J3 is open: one normal-length song, separated on the laptop's GPU, gave 40 to 50% reliable frames inside predicted notes against 76% for vocadito. The earlier 17% came from a track chosen for being short. J4 has not started.

Use only `game_vocal`, through `model_midi_pitch`. Never read the instrumental or combined melody as vocal boundaries. Keep the pipeline's files unedited so that a team update can be re-mirrored; platform differences belong in the launcher and in the run history.

**Examiner-visible evidence:** the team's extractor running on a BPDR-accepted recording, and one waveform shown under two model-estimated segmentations. This is integration and problem evidence, not a repair result.

**13 October decision checkpoint:** the voices role stays off the PP1 critical path. The laptop's GPU can now separate a normal song in 10 to 15 minutes, so a declared, artist-capped sample of 20 to 30 songs is possible overnight with `Run-Extraction.ps1 -First '<ids>' -Gpu -VocalOnly -KeepStems`; `check_stems.py plan` names the tracks. Reopen the role only if that sample passes J3. Run it unattended and do not spend P3 or P4 working time on it. Keep GAME on vocadito for the other two roles, once the melody-extraction owner confirms it as the agreed extractor. If that confirmation does not come, keep the contract stub for D6 and report the natural-boundary condition as remaining work.

## P3. Implement and train the controls first

**Prerequisite for learned research runs:** independently reviewed, audited P2 targets and a reliable feature pipeline. Architecture, loss and metric code may be built and checked against constructed fixtures before reviews arrive. Jamendo crops that pass gate J3 may feed engineering runs labelled exploratory; they never stand in for approved anchors in a reported comparison.

### Tasks

1. Build the proposal's compact encoder: starting settings of four layers, hidden size 128, four heads, feed-forward size 512 and dropout 0.1. Use explicit positional/padding masks and cents output.
2. Implement supervised correction anchors and within-voiced-span correction smoothness. Fit normalization on training data only.
3. Implement the boundary-aware augmentation-only model with 71 features and the matched frame-only model with 70 features. All views, source weights and training/selection budgets remain comparable.
4. Test known curves and masks, then fit/reload a tiny subset to debug learning. Keep this separate from independent accuracy evidence.
5. Implement original-audio and conservative-rule reference conditions with common rendering/policy where applicable. Do not add a compulsory BERT-APC reproduction.
6. Train initial matched baselines; log seeds, parameter counts, optimizer/configuration, updates, checkpoint and source manifest. Measure error on detuned and preservation regions separately.
7. Check how strongly the augmentation-only baseline reacts to boundary changes. If it already ignores them, document that and retain the fair comparison.
8. Quantify whether the model's acoustic inputs distinguish the audited 10-, 20- and 30-cent changes on approved training material. PESTO's coarse pitch response in the exploratory audit is a warning, not proof that log-mel features contain or lack the needed fine-shift information.

### Build order (approved by the owner and built on 11 October)

Everything below was built and tested without reviewers, because it uses constructed fixtures first. Code is in `src/bpdr_input`, tests in `tests/test_phase3.py`. All 16 new tests pass, and the 32 earlier ones still do. This is code that executes correctly on fixtures; no model has been trained to a result.

| Step | File | What it contains | Checked by |
|---|---|---|---|
| 1 | `losses.py` | The masked mean absolute difference, the anchor and smoothness losses, and the two response losses exactly as in the proposal [S1, pp. 21-22]. Empty masks are excluded and counted. | Constructed curves with known answers: correct value and sign (`rC - rA = -e`), masks respected, gradients flow. |
| 2 | `model.py` | The compact encoder [S1, p. 21]: four bidirectional Transformer layers, width 128, four heads, feed-forward 512, dropout 0.1, positions, padding mask, one cents value per frame. Built with 71 inputs or with 70. | Output shape, padding has no effect, parameter counts recorded for both sizes. |
| 3 | `examples.py` | The loader. Research mode accepts a pair only through `require_training_example`. A separate engineering mode reads audited exploratory pairs from the training split, writes to its own `runs/engineering` folder and stamps every output as exploratory. Normalization is fitted on training data only. | Research mode refuses exploratory pairs; engineering outputs carry the stamp; no target reaches the inputs. |
| 4 | `metrics.py` | Plateau correction error in cents, boundary sensitivity, alteration outside the detuned support, edit coverage. | A zero-output model fails the detuning measure; the frame-only model gives identical A/B outputs under deterministic inference. |
| 5 | `training.py` | One command for the three conditions with the same budget: supervised warm-up, then AdamW. Saves a checkpoint and a run record with seed, configuration, parameter count, updates and input hashes. | A tiny set is fitted, saved and reloaded to the same predictions. |

The research gate is never loosened to make engineering runs possible. With the audit decision deferred, engineering data is the existing one-singer pilot: six shifts and 18 pairs. That is enough to debug learning and not enough to compare the three conditions.

Checked on the real pilot pairs: research mode refuses them, engineering mode loads all 18, and the two models have 802,433 and 802,305 parameters. A 30-update smoke run of the training command completed on the CPU and on the Intel GPU and saved a stamped run record. After 30 updates the model still outputs almost zero, as expected; the smoke run shows the command works, not that anything was learned. Run it with `python -m bpdr_input.training --pairs-run <run> --condition bpdr --mode engineering`; the command is not yet a console script, because adding one means regenerating the lock file.

### Outputs and exit gate

Produce two reloadable baseline checkpoints, rules/reference outputs, an executed metric pipeline and development results with denominators. Padding/unvoiced frames are handled correctly; the frame-only model's identical-input boundary pairs agree under deterministic inference. Zero-output predictions fail the known-detuning endpoint.

**Examiner-visible evidence:** model predictions execute; training/development history and reproducible inference exist. At this stage there is no BPDR advantage claim.

**9 October decision checkpoint:** if training is unstable, diagnose target/normalization/mask/loss issues before enlarging architecture. If compute limits require one run per condition, lock the same reduced budget for every learned model and disclose the limitation.

## P4. Build BPDR, permission control and audio repair

**Prerequisite:** P3 checks and matched controls.

### Tasks

1. Implement boundary-invariance response loss on A/B and C/D, and local counter-detuning response loss on C-A and D-B. Keep the supervised anchor and smoothness terms.
2. Verify loss sign, time alignment, region masks, gradients and empty-mask exclusions using constructed curves. A perfectly fitted supervised target already satisfies the desired response relations; the additional objective must earn its value on unseen material.
3. Train BPDR under the agreed budget and compare development behavior with both learned controls. Use development data for weights, checkpoint selection and thresholds.
4. Build a controller that checks plausible boundary-view agreement, reliable voicing, protected transitions, correction magnitude and affected fraction. Withhold unsupported edits; do not silently clip oversized proposals. The proposal compares the main prediction with up to two other reasonable boundary versions [S1, p. 26]. GAME's alternative segmentations from P2J can supply those versions offline; a runtime source must first be measured against the 30-second target, because GAME on CPU is slow.
5. Render eligible relative cents corrections into a separate waveform. Retain duration, rests and the agreed timeline; log changed and crossfade intervals. The proposal renders repairs with Praat through Parselmouth [S1, p. 26]; the current librosa renderer only generates interventions. Build the repair renderer with Parselmouth, or record the deviation and its reason.
6. Reanalyze and verify the candidate. Roll back a failed candidate to a previously checked usable waveform; if none exists, return a reasoned non-accepted status.
7. Add original/prepared playback, pitch/correction overlays, edit logs and clear reason codes. Keep optional denoising disabled until these core paths pass; track it as remaining proposal work.

### Outputs and exit gate

Produce a BPDR checkpoint, development comparison, complete controller configuration, rendered examples and rollback evidence. Demonstrate known local repair as well as acceptable no-change input. Report correction predictions separately from actual waveform effects.

Freeze the PP1 evaluation specification and chosen settings by 16 October: metrics, independent subset, coverage denominator, preservation criterion, case-selection rules and training budget. No tuning on the untouched evaluation collection.

**Examiner-visible evidence:** the proposed paired-response mechanism executes, and its output is used by a functioning component. The comparative advantage remains provisional until P5.

## P5. Produce research evidence and integration evidence

**Prerequisite:** frozen P4 checkpoints/settings and untouched evaluation material.

### Tasks

1. Evaluate original, rules, augmentation-only, frame-only and BPDR on the declared collection. Use independently fixed masks, paired comparisons and source/singer aggregation.
2. Report detuned-support cents error, boundary sensitivity, no-change/outside-support alteration, rendered pitch, acceptance/edit coverage and failures. Save raw predictions before the controller and actual delivered results afterward.
3. Include expressive no-change material, identity controls, natural analyzer-error cases where available, and clear exclusion counts. Estimate paired intervals at independent singer/source level, keeping all related views together; label small-sample intervals descriptive. Report seed variation separately when runs support it; one run does not establish training stability.
4. Compare preservation at achievable matched acceptance coverage. Do not remove refusals or render failures from the result table without accounting for them.
5. Evaluate controlled A1-annotation boundary tracks and naturally inferred provisional boundary scores as **separate conditions**. The current A1-derived sharp timing track is not the runtime heuristic score; report any distribution shift and do not infer deployed robustness from synthetic annotation-track invariance alone. If gate J2 passed, add GAME-predicted boundaries and their alternative segmentations as a third, natural-analyzer condition.
6. Connect accepted packages to the fixed melody-understanding adapter. A temporary real extractor supports a preliminary before/after evaluation; a stub supports only contract testing. The planned real extractor is frozen GAME at the team's pinned commit and checkpoint; report note pitch-center change in cents beside note F1.
7. Test rejection propagation and seconds/beat mapping. Record which downstream outputs use learned proposed models, baselines, heuristics or fixtures.
8. Assemble the evidence register, FR1-FR7 acceptance mapping, proposal W1-W8 progress and implemented risk mitigations. Add real user/listening evidence only if it has actually been collected under the applicable protocol.

### Outputs and exit gate

Produce saved result tables/plots, audio outputs, run IDs/configurations, exclusions, a replayable demo and accepted-only integration checks. Every claimed result can be traced to actual inputs and model/policy versions. Show at least one failure or uncertain case.

**Examiner-visible evidence:** what works, how it compares, what it preserves and what remains unresolved. Do not call a one-seed pilot a completed main study or general real-world accuracy benchmark.

**21 October decision checkpoint:** if BPDR is not better, show the measured result and an evidence-based next step. If a model-affecting bug is fixed after examining evaluation results, use a fresh holdout or label the rerun exploratory. Keep research claims separate from UI fixes.

## P6. Verify and rehearse the PP1 evidence package

**Prerequisite:** P5 artifacts or an explicitly documented incomplete-evidence status.

### Tasks

1. Recheck the live application, saved checkpoints, accepted/rejected cases, pitch plots, playback volume and artifact paths on the demonstration machine.
2. Build slides around the master narrative: problem -> existing capability -> bounded gap -> implemented mechanism -> comparative evidence -> progress/requirements/risks -> benefits/remaining work.
3. Prepare technical backup material: exact losses, masks, data partitions, parameter counts, controller, renderer, metric definitions and declared limitations.
4. Prepare an offline replay from a real saved run. Label it when used; a replay is an interruption fallback, not a substitute for actual implementation.
5. Rehearse to the official slot once confirmed. Each member practices their own contribution and the hand-off boundary; do not assume shared marks or a fixed slide allocation.
6. Practice examiner questions from the master plan. Answers cite source pages or real outputs, not unsupported superiority claims.
7. Record demo date/version and completion snapshot. Show baseline versus revised dates, current blockers and executed corrective actions.

### Final acceptance

The examiner can distinguish the problem, gap, proposed novelty, implemented behavior, preliminary measured effects and remaining research. The application demonstrates an unchanged case, a local repair candidate, boundary-only invariance testing and a controlled stop. Every displayed number comes from a saved run or is clearly labelled as a target.

A polished presentation cannot fill a missing research gate. If implementation evidence is incomplete, report its actual state and mitigation rather than claiming that the plan itself proves the contribution.

## Continue after PP1

Complete remaining participant/reference acquisition, repeated-seed comparisons, natural-error generalization, blinded listening, final downstream benefit, optional denoising evaluation and full research reporting. Preserve the PP1 artifacts as a dated version; keep later changes and their evaluation distinct.
