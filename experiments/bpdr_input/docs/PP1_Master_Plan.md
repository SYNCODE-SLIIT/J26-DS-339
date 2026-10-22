# BPDR master plan

**Current baseline: 12 October 2026. Owner: Liyanage L. N. P., IT23141506.** This is the authoritative scope and research plan. The [execution plan](PP1_Phase_Execution_Plan.md) gives tasks; the [evidence tracker](PP1_Evidence_Tracker.md) records actual results. Earlier versions are recoverable in `private/cleanup_2026-10-12/recovery.zip`.

## 1. Research purpose and system boundary

The project turns a short sung/hummed idea into genre-conditioned harmony while preserving musical identity:

```mermaid
flowchart LR
    A[Short solo vocal] --> B[BPDR: validate and preserve or minimally repair]
    B -->|Accepted WAV and status| M[Melody understanding]
    M --> P[Relative-root and family planner]
    P --> R[Context-aware chord refinement]
    M --> R
    B -->|Re-record or failure| X[Stop before generation]
```

BPDR owns audio integrity, assessment, conservative pitch repair, preservation, rollback and an auditable accepted-only package. Melody understanding owns notes, rests, rhythm, key, phrases and embeddings. Planning/refinement own harmony and genre conditioning. Component numbering varies across proposals; use names and owner IDs. Full source mapping and unresolved team contracts are in the [project audit](Project_Audit_2026-10-12.md).

The research question is whether boundary-invariance and local counter-detuning response losses, together with supervised preservation anchors, improve held-out small-detuning repair without damaging acceptable singing, relative to matched augmentation-only and frame-only controls. A Transformer, a separator or a successful pipeline run is not the contribution. A model that always predicts zero does not demonstrate successful repair.

Operating scope: one unaccompanied voice, WAV/FLAC, 0.25-20 seconds. Preserve the original, sample-time origin, internal rests, note order, tempo, expressive variation and duration. Do not infer intended notes by snapping to a scale, replace rhythm, synthesize gap-filling notes, repair heavily clipped recordings or treat multiple singers as supported. Jamendo separation is an offline data-preparation step, not a newly promised production input feature. Whistling appears in one figure but is outside the written singing/humming prototype scope unless separately validated. [S1 pp11,23-28]

## 2. Verified foundation and remaining work

**Owner update, 14 October — no reviewers for now:** experienced musical reviewers are unavailable. [The revised workflow](Reviewer_Deferral_2026-10-14.md) continues automated/constructed/annotation-based engineering evidence now and defers actual musical review until reviewers are available. Gate D/J4 remains open/deferred; existing review records and research-loader guards are preserved. This is not an automatic musical approval or a reason to halt engineering. Full musical acceptability/research eligibility remains unvalidated.

**Current completed step — Gate F, 15 October:** [frozen control diagnosis](Known_Control_Diagnosis_2026-10-15.md) reproduces all **16 guarded audits and 32 transition peaks exactly**. The glide reject is one positive PESTO change of +0.000413 cents at 1.24 s against a -15.6-cent request; its sign remains positive in float64, and native/aligned timestamps agree. YIN measures -17.420747 cents. The reject remains; no epsilon is adopted. Existing transition peaks measure pitch-change magnitude, not target error. Separate known-control center-law residual maxima are 5.422835/7.289706 cents at the fixed 2048/1024 windows; these remain finite-window diagnostics, not auditory thresholds. All **147 regression checks pass**. No WAV or renderer is created or changed; all eight measured and 24 withheld cases remain. **Next — Gate F (open):** Run the declared audio-only source-estimation feasibility on the eight existing constructed sources; withhold construction F0/phase from the estimator and use it only in scoring. No new rendering or vocal trial yet. **Gate D/J4 remains deferred; zero research-ready examples or accepted repair.**

**Prior completed step — Gate F, 15 October:** [known-source analytical audio controls](Known_Source_Control_2026-10-15.md) render the eight preselected strict cases and explicitly withhold 24. Seven of eight full audits pass, matching the baseline count on those same eight cases, but the passing cases differ: constant-880 is gained and glide-220 is lost to one opposing coarse-direction frame. All eight fine-pitch/preservation checks and eight exact identities pass. Maximum transition estimates improve on the matched subset, yet four individual 2048-window cases and glide at 1024 worsen. Source-relative amplitude dips improve across 18 ramps; this uses a different denominator from the older branch-relative diagnostic. All **141 regression checks pass**. These controls require known construction F0/phase and are not a practical vocal renderer or musical accuracy. The subsequent frozen diagnosis is recorded above.

**Prior Gate F update, 15 October:** [source-anchored feasibility](Anchor_Feasibility_2026-10-15.md) retains all 32 constructed cases/36 supports. The fixed sign/magnitude-preserving phase bridge solves nine supports in eight cases; 18 supports fail this model's necessary budget and nine more are unresolved by the fixed basis. This is conditional mathematics, not audio pass rates or universal impossibility. Relaxed solutions permit unwanted boundary interventions and are not approved. All **134 checks pass**, with no new renderer/WAV/accepted package. The subsequent known-source audio control is measured above.

**Prior Gate F polar-join comparison:** [one fixed phase-aware candidate](Phase_Join_Comparison_2026-10-14.md) removes measured fundamental dips, but worsens transition estimates at both fixed windows (maxima 80.81/127.94 cents versus 42.42/58.24). All 32 guarded audit dictionaries remain identical to baseline, preserving 26 passes/six rejects; physical/fine checks and eight identities pass. The unchanged advancement filter refuses it. That stage passed **129 checks** and remains evidence.

**Prior Gate F phase feasibility:** [64 existing candidates reconstructed exactly](Phase_Feasibility_2026-10-14.md), with fixed phase/amplitude measurements on 72 ramps per method. Below-half narrowband amplitude ratios occur in 25 baseline and 45 taper ramps. That stage passed **123 checks**, motivating the subsequent fixed polar-join experiment while retaining all prior failures.

**Current work gate (14 October): Gate F — usable delivered audio, renderer validation; open.** Every next-step update should name the current gate and the gate-specific action. Future prerequisite Gate D/J4 is deferred until experienced reviewers are available; Gate B source/voice support and Gate E matched research evidence also remain open. Working on engineering fixtures in F does not close those prerequisites or authorize research fitting/delivery. Engineering continues now under the owner decision.

**Prior Gate F boundary comparison:** [one declared taper candidate](Boundary_Comparison_2026-10-14.md) gives 27/32 full audits versus 26/32, with all fine-error/preservation checks and eight zero identities passing. It loses an existing pass and worsens the shorter-window maximum transition peak; the advancement filter refuses it. That stage passed **119 regression checks**, proving execution rather than musical accuracy. No renderer is adopted or accepted repaired package created. This is preserved prior evidence; the frozen known-control diagnosis above is the latest step.

**Prior Gate F diagnosis:** [all 32 controlled cases replay](Framewise_Diagnostics_2026-10-14.md), retaining 26 baseline passes and six rejects. The 17 opposing PESTO changes are below 0.001 cent; no sign tolerance is adopted. Full/local/oracle contrasts expose blending effects on estimates and window sensitivity, without proving musical preservation. That diagnostic stage passed 112 checks and remains preserved evidence.

| Area | Present evidence | Remaining gate |
|---|---|---|
| Technical audio path | Original retention, channel handling, resampling, frozen acoustics, checked export and failure paths | Musical suitability and repair permission remain unvalidated |
| Reference laboratory | 40 sources, 29 singers, 57 mapped crops; 114 pending review rows | Actual independent reviews and approved intervals |
| Interventions | Six audited shifts/18 variants of one phrase from one singer | Broader independent-source coverage; variants are not new performances |
| Learned implementation | Encoder, losses, loader, metrics, recorded engineering fits | Development selection, held-out study, no-change endpoint and uncertainty |
| Extractor compatibility | GAME pilot on 28 Vocadito recordings; accepted-audio check on 17 crops | Melody-owner agreement and repeat checks after real repair |
| Jamendo | 1,604 tracks, eight stems; fresh five-artist sample yields six screen crops from three artists and five passing shifts across two artists | Manual suitability, broader audited support, reviewed anchors and J3/J4 |
| Delivered repair | Prediction/rollback fixtures and unadopted framewise candidate tested on controlled signals; no delivered repair | Useful validated model, uncertainty/permission calibration, real-audio renderer/listening validation and accepted delivery |

Use the current tracker for run IDs and regression results. Existing training metrics are measured on training pairs; they are not accuracy on independent data. No trained checkpoint currently establishes BPDR's research benefit.

**Panel notebook requirement (owner, 14 October):** maintain the existing two notebooks exactly in line with completed work. Update affected cells and `notebooks/evidence_index.json` after each step, execute with fresh kernels and verify outputs before presenting. Keep historical engineering fits and planned features explicit; no invented approvals or accuracy. Check redundancy and preserve recovery before notebook cleanup inside BPDR. The [README notebook workflow](../README.md#keep-the-panel-notebooks-current) defines maintained sources versus dated executed evidence.

**Prediction-to-permission preview:** [the adapter step](Prediction_Adapter_2026-10-14.md) now connects the saved smoke model to candidate interval/permission checks. Its one seen-crop preview requests no edits; constructed cases verify grouping and refusal. Runtime uncertainty/expressivity/permission remain uncalibrated, so no repair is authorized. This is execution evidence, not a useful-model or accuracy result. The full suite passes 102 checks and both notebooks expose these limits.

**Framewise candidate tested:** [Praat/Parselmouth controlled evidence](Framewise_Renderer_2026-10-14.md) gives 26/32 full candidate-audit passes, with all 32 fine-error checks passing. Six stricter coarse-direction rejects remain visible. Eight identities, timing/outside/rest preservation and one supplied-fixture transaction verify. This separately versioned candidate is not adopted and has no accepted repaired package. The current suite passes 108 checks; real vocal/listening validation and actual reference/uncertainty gates remain open.

**14 October follow-up:** [renderer diagnostics and the independent review bundle](Audit_Diagnostics_and_Review_2026-10-14.md) reuse the current extracted material. The best aggregate tested setting gives 10/18 audit passes versus 5/18 active, but regresses two existing successes and still yields no all-six crop. Treat it as a calibration candidate, not an adopted repair path. The active v0.3 policy/renderer and zero research eligibility remain unchanged; actual independent reviews and broader expressive-control validation are next.

**14 October wider calibration:** [both datasets and twelve expressive control profiles](Renderer_Calibration_2026-10-14.md) are measured with unchanged gates. Vocadito gives 17/18 passes for all three renderers, but each candidate loses one active case and gains another; its three crops cover two recordings and only S29. Jamendo reproduces 5/18, 10/18 and 9/18. Constructed fine-only checks give 47/72, 70/72 and 63/72; even the strongest candidate fails demanding vibrato. Retain the active renderer, complete real reference reviews and matched listening, and address independent-voice coverage before research fitting. Passing variants do not create eligible data.

## 3. Two datasets, two distinct sources of evidence

**14 October controller fixtures:** [engineering permission and rollback](Controller_Fixtures_2026-10-14.md) now verify nine constructed outcomes, with 93 regression checks passing. The planner withholds invalid proposals; post-render failures return the exact preserved source. Supplied corrections/uncertainties, illustrative bounds and the laboratory renderer are fixture inputs, not adopted production policy. No `ACCEPT_PREPARED` package or research example is created. The notebooks include the measured mechanics and remaining gates.

| Collection | Use now | Use after gates | Limits |
|---|---|---|---|
| Vocadito | Human F0/A1/A2 reference checks, controlled boundary cases, candidate review anchors | Reviewed multi-singer training/development and held-out reference evaluation | Small, with varying recording conditions; note labels do not identify an intended tune |
| MTG-Jamendo | Separated-vocal preparation and model-boundary engineering experiments | Supplementary preservation anchors after license, suitability, review and shift audits | Accompaniment, bleed/reverb/backing voices, no human F0/note reference or verified singer identity |

Both sources stay in this component. Jamendo extends the proposal; it does not replace Vocadito or the proposed participant study. Adult participant recordings, guide-melody cases and free humming remain a later ethics/consent-dependent study. The proposed 8 adults x 6 phrases is a planning estimate, not collected data or a justified final sample size. Determine main-study size from pilot precision/power. [S1 pp16-17]

Vocadito: split by singer before cropping/augmentation. Jamendo: split by source/artist before crops, cap initial sampling at one song per artist, check exact and near-duplicate audio, and retain every descendant in its parent's partition. Artists are proxies; recurring singers/collaborations can still cross artist groups. Keep evaluation/final-holdout sources untouched during geometry design and tuning. Any unfamiliar-melody claim also needs melody-level separation.

References must record `human_annotation` versus `model_estimate`. A GAME prediction or two agreeing trackers cannot count as two reviewers. An acceptable reference has target **zero correction**; a known audited local detuning has target **negative injected cents** on the validated plateau, zero outside support, with ramps excluded until validated. Neither GAME nor dataset MIDI notes are commands to overwrite the singer's pitch. Agreement checks assess reliability, not intended musical correctness.

Two musically experienced reviewers independently approve usable reference intervals, identify ambiguous transitions, and record reasons resolving disagreements. Both sources require that gate before research anchors. For Jamendo, additionally document whether model-estimated pitch is sufficient for the approved experimental role; do not manufacture human F0 files to satisfy the Vocadito loader. If changing the two-reviewer methodology, record and obtain supervisor agreement on the change before research claims. [S1 pp16-17]

**Current execution decision:** the owner has deferred this review stage because reviewers are unavailable. The paragraph above describes the future proposed research validation; it does not prevent the now-authorized engineering work. Continue with explicit provisional labels and actual automatic checks, without rewriting pending records or inventing identities/approved intervals. Later add the real review as a new evidence version.

Dataset terms: Vocadito is described by its [official record](https://zenodo.org/records/5578807). MTG-Jamendo's [official terms](https://github.com/MTG/mtg-jamendo-dataset#license) restrict the dataset to non-commercial research/academic use and provide individual audio licenses. Preserve attribution and license metadata; exclude NoDerivatives and unknown-license tracks from the transformation/release pilot. The local assessment counted 353 NoDerivatives tracks; verify licenses in each new manifest. Do not infer rights from the repository's code license.

## 4. Reuse models; own the preparation path

**14 October direction:** a future macOS/Windows melody/chord pipeline will provide an optional BPDR exporter, following the [saved owner brief and integration decision](Portable_Pipeline_Decision_2026-10-14.md). Preserve its existing outputs/default behavior and validate both platforms before integration. For now, use the already extracted baseline; portable-pipeline implementation is deferred and does not block review or intervention diagnostics.

Retain the team's pinned BS-RoFormer separator and GAME commit/checkpoint, verified by hash. The newer melody/chord pipeline's vocal stage uses those same models. Its instrumental melody, combined melody, chords, key and Roman labels belong to other work and are not BPDR supervision. Its combined output fills vocal rests and resolves note overlaps; do not use it to evaluate vocal preservation. Read unshifted `model_midi_pitch`, retain stems, and save each decode because GAME has no exposed extraction seed.

The [common builder implemented on 13 October](Data_Preparation_Verification_2026-10-13.md) produces immutable manifests with source/artist/singer identity as available, license, partition, crop origin, source/crop/accepted hashes, typed references, model versions, saved boundary decodes, reliability masks, review state and exclusions. For Vocadito: solo source -> mapped crop -> Phase 1 -> GAME boundary adapter, retaining human references. For Jamendo: licensed/grouped song -> kept stem -> mapped <=20-second crop -> Phase 1 -> GAME -> provisional pitch screen and manual single-voice/bleed checks -> review queue. The Vocadito manifest is reused behind a source-specific adapter; Jamendo does not receive fabricated human annotations. The [fresh five-artist sample](Jamendo_Five_Artist_Sample_2026-10-13.md) completes this path: ten unchanged outputs, nine screen candidates, eighteen pending reviews and zero research-ready examples. J3 suitability calibration and J4 approvals remain open.

The initial three-stem plumbing pilot and the declared five-artist, voice-tagged >=60-second sample have run. Three of five new artists reach the provisional review queue; only two artists have any passing v0.3 detuning, and no crop passes all six shifts. Prioritize independent musical review and renderer/support diagnostics before deciding on a 20-30-artist expansion. Inspect fine YIN error and loss of common voiced audit coverage as well as the earlier PESTO quantization issue. Use GPU separation and record model/platform/overlap settings; resumptions must match the declared settings and selection. Full-song separation remains the baseline; excerpt-first separation waits for repeated J5 evidence. Keep the full 1,604-song expansion on hold until quality and independent-source support justify it.

## 5. Repair quality is an acceptance gate

The proposal's runtime is a frozen acoustic analyzer, BPDR predictor and deterministic permission rules, followed by local rendering and remeasurement. Offline controls do not become additional production stages. [S1 pp23-27]

1. Assess technical integrity and musical suitability. Current Phase 1 proves the former only. Calibrate the latter on labelled suitable/unsuitable examples including silence, noise, leakage, multiple voices and acceptable expressive singing.
2. Predict relative cents corrections under the main and reasonable alternate boundary views. Require reliable voicing, sufficient support, bounded uncertainty and agreement; protect onsets/offsets, rests and expressivity. Corrections beyond allowed limits are withheld, not silently clipped.
3. Render only permitted intervals with a duration-preserving framewise method; the proposal names Praat/Parselmouth. The artificial constant-shift generator is not the production renderer. Pin the renderer and audit zero-change identity, realized pitch, transitions and changes outside edit support.
4. Remeasure delivered audio. Preserve sample count/time origin and rests; verify pitch error on supported regions, protected-region drift and audible artifacts. Compare the same frozen extractor before/after and against its repeat variation.
5. Roll back a failed candidate to a previously checked usable waveform; otherwise return a reasoned non-accepted status. Withheld edits may produce `ACCEPT_UNCHANGED` only if the input is suitable. Emit `ACCEPT_PREPARED` only after all applicable checks pass.

Freeze numerical bounds on training/development data before held-out testing. Report abstentions/edit coverage, failures and harmful edits with repair error; accepting everything or correcting nothing cannot alone demonstrate accuracy. Thresholds in current Phase 1 are provisional engineering settings, not calibrated musical guarantees.

## 6. Fair study and output assurance

Compare original input, a declared conservative rule baseline, augmentation-only, frame-only and BPDR. The three learned conditions share data, partitions, capacity as appropriate, source weighting, optimizer/update budgets and selection protocol. Fit normalization on training sources only. Use at least three predeclared seeds where feasible and label a one-seed run exploratory. Select checkpoints/weights/controller thresholds on development data; freeze before evaluation. Disable denoising in the initial comparison. If introduced later, all correction conditions share one fixed noise policy; the original control receives only necessary decoding/format conversion. Conditional denoising requires a reliable nonsinging noise interval, feature recomputation and preservation/rollback checks, with its effects recorded separately from pitch repair. [S1 pp17-22,26]

Keep three boundary conditions separate: controlled human-annotation tracks, the operating audio heuristic, and GAME model-estimated tracks. Training currently uses A1 timings while runtime uses a heuristic; quantify that distribution change rather than claiming synthetic invariance proves runtime robustness.

Evaluate jointly: plateau correction error, boundary-response sensitivity, preservation inside/outside support, dedicated no-change cases (vibrato, slides, acceptable transposition), rendered pitch/artifacts, harmful-edit rate, technical integrity, edit/acceptance coverage, latency and same-extractor note performance. Include onset+pitch and offset-aware results with their conventions. Show raw predictions and delivered/controller outcomes at comparable coverage. Pair comparisons by independent singer/source and give 95% intervals with denominators, exclusions and failures. Repeated frames/variants are not sample size. Blinded listening uses fixed playback conditions; a small listener sample is a pilot.

Use held-out Vocadito references for the first controlled pilot. Keep Jamendo robustness/yield results separate and do not score accuracy against the same pseudo-labels used to select examples. Participant data is needed for broader real-user or intended-melody claims. If no independently reviewed examples exist, continue useful engineering work but report no supervised comparative result.

## 7. Integration and schedule

The accepted package contains request/source IDs, status, WAV path/hash/rate/count/channels, time-origin/crop mapping, reason codes and links to versioned diagnostics, analyzer/model/controller/renderer policies and edit intervals. Consumer validation rejects missing/tampered audio and non-accepted statuses. Do not downsample or recrop without explicit mapping. The current `ProcessingResult` schema 0.1 is a proposed contract; the melody owner must accept the actual format. GAME is a fixed baseline adapter until that owner identifies the agreed extractor.

H1 currently supports technical format compatibility, not repair correctness or team acceptance. Extend H1 after actual `ACCEPT_PREPARED` output, and once on evaluation data with extractor repeat variation. For repaired audio, require preserved timing/rests and no unintended degradation outside permitted edits; inside edited support, measure pitch against the declared correction reference. Do not require repaired pitch to match the deliberately detuned input. Later isolate raw-vocal, prepared-vocal and reference-melody downstream paths under fixed planner/refiner settings to measure error propagation.

The previous user planning target is **25 October**. The supplied proposal's schedule figure lists an initial PP1 prototype on **26-28 October**; neither establishes the official slot here. The proposal places controls in Nov-Dec, BPDR study Dec-Jan, integrated evaluation around Feb 2027 and final submission around Apr 2027 (estimated). Do not promise the whole scientific study by October. The execution plan separates a near-term engineering prototype from the subsequent reviewed study.

## Sources

- S1: supplied IT23141506 proposal, all 43 PDF pages; scope pp11-17, comparisons/losses pp17-23, diagrams pp24-25, delivery pp26-28, schedule p36.
- S2: supplied IT23439078 proposal, all 57 pages; architecture p38, datasets pp29-30, accepted-only contracts pp53-55.
- S3: supplied IT23148086 proposal, all 37 pages; representation/scope pp19-24, upstream-error isolation p18.
- S4: supplied IT23157132 proposal, all 44 pages; data/context pp20-25, architecture p26, refinement mapping ambiguities pp10-11,23.
- Source copies remain in `private/proposals`; exact source hashes and full-page audit are recorded in [Project_Audit_2026-10-12.md](Project_Audit_2026-10-12.md). Historical rubric/work-package details remain in the recovery snapshot.
