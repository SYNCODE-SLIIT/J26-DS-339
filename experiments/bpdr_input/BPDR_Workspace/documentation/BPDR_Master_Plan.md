# BPDR master plan

**Version:** 1.13 — 1 November 2026  
**Owner:** Liyanage L. N. P. / IT23141506  
**Working folder:** `experiments/bpdr_input/BPDR_Workspace`  
**Current position:** **MOD-03 ACTIVE: finish usable scoped Transformer repair before other features**, following the owner's 28 October instruction. Broader data, grouped activity/correction training, versioned held-voice eligibility and guarded BPDR integration are in the [execution record](Transformer_Usability_2026-10-28.md). The [completed iteration](Transformer_Usability_Result_2026-10-28.md) fails all four final singers: 5/48 improvements, 12/12 originals unchanged. The candidate is inactive and MOD-03 remains incomplete. The [joint-context comparison](Transformer_Joint_Context_2026-10-28.md) completed 11 fits: raw CV error improves, but development delivered repairs fall to 6/48 with all 12 originals unchanged. The [frozen curve/renderer study](Transformer_Curve_Study_2026-10-28.md) raises known-development verified repairs to8/48 but still fails the fixed per-singer gain requirement. Both variants remain inactive. Selective UI remains planned.

**Latest model evidence, 29 October:** [Support supervision](Transformer_Support_Study_2026-10-29.md) and [broader actual-audio training](Transformer_Audio_Pool_2026-10-29.md) completed16 new CV fits. Fresh re-audit validates373 unique training variants,265 beyond the current108, from the same50 originals/10 singers. Broader training lowers raw MAE11.901 ->11.322 cents and model-only support68 ->72 cases, but fails female2 original-prediction limits and reduces verified repairs23/200 ->18/200. All50 original WAVs remain unchanged. Both new studies reject adoption. MOD-03 remains ACTIVE; original/final outcomes stay retained and no confirmation audio was accessed.

This is the active component plan. Use the [implementation tracker](BPDR_Implementation_Tracker.md) for task status and the [planning review](Planning_Review_2026-10-27.md) for source authority, conflicts and retained history. The earlier PP1 master, execution and evidence documents remain historical records. Their old next-step paragraphs do not override this plan. Updating documentation does not complete a planned feature or pass a research gate.

**Latest completed checks:** [Fixed ensemble](Transformer_Ensemble_2026-10-29.md) passes prediction gates but yields2/200 verified repairs. [R3 renderer](Transformer_R3_Renderer_2026-10-29.md) yields22/200 versus23/200 current Praat and is rejected. Both preserve50/50 originals;17 combined tests and independent waveform/hash checks pass. No full refits, development access or activation. The [candidate-level preflight](Transformer_Candidate_Preflight_2026-10-29.md) finds177 positive,8 negative and14 unassessed regions.39/108 baseline cases have no region overlap, so retain a dense fallback.96 synthetic unchanged controls add zero measured candidates; this failure remains visible.

**Latest model work,30 October:** [Candidate-head/readout comparison](Transformer_Candidate_Head_2026-10-30.md) completed8 head-only fits with frozen parents. Pooled and evidence-augmented heads fail response gates and verify20/200 and21/200 repairs. Frozen learned-centre readout passes prediction gates and verifies24/200, but delivered error20.706 exceeds reference20.342. None selected. All50 originals unchanged;20 focused tests pass,8 frozen parent lineages and65 encoded outputs verified. No refits, development or confirmation access. MOD-03 remains ACTIVE.

**Latest fixed-readout result:** [V25 distribution readouts](Transformer_Distribution_Readout_2026-10-30.md) pass raw prediction gates but lose actual delivery: median18/200 with21.444c remaining error; upper quartile19/200 with21.992c, versus reference23/200 and20.342c. All50 originals remain unchanged per arm. Twenty tests and independent replay of37 encoded outputs pass;6 protected artifacts remain unchanged. Neither readout selected; no new fits, development/confirmation access or activation.

**Updated owner direction,30 October:** investigate whether Transformer choice causes the difficulty, compare other model families, consult primary research and suggest input-usability improvements within BPDR ownership. [Model choice and input usability](Model_Choice_and_Input_Usability_2026-10-30.md) records verified pipeline bottlenecks, research sources, comparative protocol and component-safe feature extensions. Model family superiority remains unproved.

**Current execution:** [V26 dense residual adapter](Transformer_Dense_Residual_2026-10-30.md) finishes4 head-only fits with exact initial function preservation, passes raw gates but loses delivery21/200 and21.394c versus23/200 and20.342c. All50 originals unchanged. Independent216-view/21-WAV/frozen-parent/protected-artifact replay passes; no selection change. All25 focused tests pass. [V27/V28 model-family comparison](Model_Family_Comparison_2026-10-30.md) finishes all8 GRU/TCN and4 fresh ExtraTrees fits, actual-WAV comparisons and independent replay with matching V12 sampling/source digests.

**Next model work:** complete the [matched absolute-response training](Transformer_Absolute_Response_2026-11-01.md), actual-WAV comparison and independent replay. All4 new squared-response controls must reproduce frozen V12 tensors, predictions and sample digests before the4 absolute-response fits. Change only the counter-detuning residual from square to absolute with coefficient0.25; keep architecture, data, sampling, original losses, budgets, decoder and every permission/output check fixed. This is not completion of the full RES-01 study or a learned boundary-invariance claim. Require a qualified locked pipeline, verified full-fit lineage, known development and untouched confirmation before activation. No selective UI while usable repair remains unproved.

**Current authoritative update,1 November:** owner resumed after pausing on30 October. [FCPE measurement](Acoustic_Measurement_2026-11-01.md) finishes in an isolated sidecar:91.064c relative-shift MAE versus current YIN0.860c on audited references; independent158-track replay passes. No replacement adopted or active-environment change. Privileged local-context diagnosis45->48 permissions,0 losses, independently replays108 cases/97 original-crop queries; not a runtime feature. [V30 binary activity](Transformer_Activity_Decoding_2026-11-01.md) passes raw gates but loses delivery21/200 and21.801c versus23/200 and20.342c;50 originals unchanged, independent216-view/21-WAV replay passes, no adoption. V29/V30 add no neural fits. V31 now completes4 new MSE controls with exact reference reproduction and is running4 matched absolute-response fits.32 focused tests pass. No development/confirmation access, selected-model change or usable-accuracy claim.

**Latest completed comparison:** GRU18/200 and21.626c, TCN19/200 and21.281c, fresh ExtraTrees20/200 and21.217c all lose to reference23/200 and20.342c. All50 originals unchanged per family; independent648-view/57-WAV replay across12 trained alternatives passes,6 protected artifacts match. GRU/ExtraTrees fail raw gates. The tree's terminal post-fit roundoff-check failure is retained and corrected by explicit serial-reload amendment without refitting or gate changes. None selected. A privileged known-label diagnostic permits45/108 cases and verifies35 outputs after10 rollbacks; this is NOT model performance or a general ceiling. Supplementary preservation finds outside-injection sample changes in7/23 reference outputs, with very small aggregate pitch drift; listening remains unassessed. Full diagnostics independently verified. MOD-03 remains ACTIVE.

## 1. Main research goal

The [submitted proposal](IT23141506_J26_DS_339.pdf#page=15), section 2.1, states:

> To develop and experimentally evaluate BPDR for short vocal-input preparation, determining whether it improves local pitch repair and subsequent melody extraction under segmentation errors without increasing harmful alteration of acceptable musical material at matched acceptance coverage.

The research question is whether a shared correction model can **ignore unreliable boundary changes while responding to small physical pitch changes**, with preservation of the original performance. The proposed methodological contribution is the combination of boundary-invariance and local counter-detuning objectives with supervised preservation anchors. A Transformer, continuous pitch correction, vibrato preservation, a collection of libraries or an interactive editor is not itself the claimed novelty. [Proposal pp14–22]

Keep the five proposal objectives visible:

| Objective | Required evidence | Current position |
|---|---|---|
| O1: characterize failures | Taxonomy of recording failures, false splits/merges and local detuning | Technical diagnostics and controlled investigations exist; natural-user characterization remains incomplete |
| O2: reproducible data | Permitted references, audited interventions, source/singer partitions and exclusions | Public-data preparation exists; independent suitability approval and participant study remain incomplete |
| O3: test BPDR against matched controls | Full BPDR, augmentation-only and frame-only under equal data/budgets/selection | Implemented comparisons exist; a consistent held-out BPDR advantage remains unestablished |
| O4: build the input component | Validation, preservation, bounded repair, rerecord feedback and traceable accepted-only handoff | Technical path and experimental previews work; usable new-Transformer repair is the immediate target |
| O5: evaluate usefulness | Delivered repair, fixed-extractor benefit, expression, decision quality and computational cost | Limited controls and interface receipts exist; independent listening, natural-intention evidence and general downstream benefit remain open |

**Three completion statements must remain separate:**

1. **Engineering completion:** supported inputs can pass from a frozen model through permitted local rendering, decoded-WAV verification and a user-reviewed export.
2. **New feature completion:** the user can protect audio and individually review verified repairs, with a final verified combination.
3. **Research validation:** matched held-out experiments support the stated BPDR objective and its preservation/downstream claims. Engineering or UI completion does not imply this result.

The next presentation targets the first two on a declared held-voice scope. The full research study remains on the roadmap and must retain negative results if simpler controls perform as well or better.

## 2. What we are building and who owns it

BPDR prepares a short solo vocal idea for later melody understanding. It validates the recording, measures acoustics, preserves usable audio or proposes a small local repair, verifies the delivered waveform and returns a traceable accepted package or an actionable stop.

| BPDR owns | Other components own |
|---|---|
| Audio integrity and technical preparation | Final musical note transcription and note durations |
| Internal pitch/voicing/onset/boundary evidence | Beats, tempo, key/mode, musical phrases and harmony-oriented melody representation |
| Relative local pitch correction and rendering | Genre-conditioned harmonic planning and chord progressions |
| Repair intervals, protected samples, rollback and review | Chord realization, extensions and refinement |
| Accepted WAV, time mapping, diagnostics and provenance | Musical interpretation of that accepted waveform |

BPDR's internal acoustic segmentation is not a second public note representation. The new objects are **sample/time-based audio edit intervals**, without note names, key, beat or chord labels. Source: proposal table 4, p27.

### Supported scope and user decisions

- Existing technical input: one unaccompanied voice, WAV/FLAC, 0.25–20 seconds. Technical acceptance alone does not establish musical correctness or monophony.
- First selective-repair release: small local shifts in stable held-voice regions. Short/multinote phrases, natural novice errors and general humming accuracy require further validation.
- Preserve source bytes and the canonical mono 22,050 Hz PCM24 working original. Render from that working original, preserving its sample grid, duration, pauses and untouched samples.
- Build in the **BPDR app only**, using the existing `Run-Demo.ps1` workflow. Do not change `pp1_demo`, sibling components or their models/features.
- Add drag selection plus start/end fields. When protection overlaps a repair's complete support, including its blend, withhold that **whole repair** in v1.
- Accepting an edit can never bypass technical permission or verification. A user decision can only restrict the edits available downstream.
- No note/key/scale snapping, invented melody, rhythm replacement, generative voice model or automatic repair of badly clipped/multisinger input.
- Keep the existing optional noise path separate; do not add noise-policy changes to the initial pitch experiment.

The owner indicated more than two weeks before the next presentation. The exact slot is unconfirmed. Relative milestones below are planning estimates, not official dates or a promise to finish the full study in that window.

## 3. Verified starting point

The current working learned demo loads ExtraTrees through guarded model/evidence checks. The historical Transformer and the new context Transformer are distinct checkpoints and training histories.

| Evidence | Actual result | What it permits us to say |
|---|---|---|
| Existing ExtraTrees delivered-WAV pilot | 4/16 interventions scoreable, one scoreable singer; 2 improvements; mean error 27.442 → 14.663 cents; 4 original controls preserved | A scoped experimental preview works on particular controlled examples; broader gates failed |
| New context-Transformer calibration | 10/16 interventions scoreable, 2 singers, 3 scoreable original recordings; BPDR-condition raw MAE 5.839 cents vs matched ExtraTrees 10.031 | Raw correction prediction improved on this development set; not delivered-audio accuracy |
| Matched new neural conditions | Augmentation-only raw MAE 5.644; frame-only 5.652; BPDR 5.839 cents | This seed does not demonstrate a BPDR-loss advantage |
| Original preservation, raw predictions | New BPDR condition 4.066 cents mean absolute request on originals | Preservation still needs work; this value is not a delivered edit |
| New-Transformer permission replay | 7 proposals withheld, 2 identity, 1 nonidentity outside the scored shifted plateau; scored-region error remains 25.833 cents | No useful permitted repair demonstrated on these scored regions; no WAV was rendered in this replay |
| Documentation baseline replay | Saved prediction/target recomputation 5.83948 cents; hashes match; 7/10 cases have zero scope-eligible scored frames | Eligibility and candidate selection are immediate bottlenecks; renderer failure has not been established |
| Completed model-first comparison | 60 CV fits, 7 scoreable held-out singers; baseline correction/original request 9.857/5.312 cents, combined 10.568/3.688 | Preservation improves 30.6% at 7.2% higher correction error; all objectives fail the all-singer prediction gate |
| Fixed-default actual-WAV evaluation | 10/16 calibration attempts scoreable; 0 verified improvements; measured error 25.975 -> 25.975 cents; 3 technically verified encoded outputs include 2 altered original controls | Completed tracing and audio checks do not establish usable repair; keep candidate diagnostic and active demo unchanged |

The two 16-attempt denominators above belong to different evaluation sets. Do not combine them, report raw 83.6%-within-10-cents as general singing accuracy, or reuse previously inspected material as a fresh final test.

Evidence: [Transformer recovery report](Transformer_Recovery_2026-10-27.md), [saved comparison](../results/transformer_recovery/760a6f1745c94263a5b9a51b7dafffd5/evidence.json), [permission replay](../results/transformer_recovery/760a6f1745c94263a5b9a51b7dafffd5/permission_audit.json), [planning baseline replay](../results/planning_review/2026-10-27_1754e056/baseline_replay.json), and [earlier delivered-WAV verification](EXPERIMENT_VERIFICATION_REPORT.md).

Latest evidence: [model-first results and interpretation](Transformer_Improvement_2026-10-27.md), [complete comparison](../results/transformer_improvement/d6e8b933b23546c78a53aa2292c03c1a/index.html), and [after-training accuracy workflow](Transformer_Accuracy_Workflow.md). This broader cross-validation and the earlier two-singer calibration are different protocols. The default calibration checkpoint has 6.670-cent raw error / 79.87% within 10 cents, yet no delivered repair benefit. Seed 340 passes the two-singer prediction check only; it was not selected after observing calibration.

### Proposal deviations that must remain explicit

| Proposal / earlier design | Current development variant | Rule for interpretation |
|---|---|---|
| 71/70 acoustic inputs; initial 4-layer, width-128 encoder | 32/31 own-audio context inputs; 2 layers, width 48; checkpoint window 81 | A declared feature/architecture redesign, not an exact proposal implementation comparison |
| Mean-absolute `d_M` for boundary and counter-detuning response terms | Squared normalized response terms in the 27 October context experiment | Retain this as engineering evidence. Use the proposal's MAE response definition for the proposal-aligned study; report any squared-response alternative separately |
| Explicit correction smoothness term | Initial context experiment lacked it; MOD-01 adds valid adjacent-target derivative SmoothL1 | Continuity alone slightly reduces held-out correction error but does not satisfy preservation; retain the negative result |
| Vocadito and planned consented novice recordings | Supplemental VocalSet, Jamendo and HumTrans investigations | Public-data engineering does not replace participant collection or natural-intention references |
| Matched full BPDR vs augmentation-only is the primary scientific comparison | ExtraTrees is a practical demonstration comparator | Beating ExtraTrees alone cannot establish BPDR's methodological contribution |

## 4. Target workflow and minimum interfaces

**Preserved working original → acoustic evidence → frozen Transformer → multiple repair candidates → versioned eligibility/permission → local rendering → decoded-WAV verification → individual review → composition from original → final verification → accepted package.**

Retain these existing mechanisms rather than duplicating them:

- `context_transformer` and its serialized feature/configuration contract for the new predictor.
- `runtime_views`, `scope_mask` and `correction_from_prediction` as the documented legacy eligibility/selection baseline.
- `plan_fixture`, `transact_fixture`, `render_candidate`, `audit_curve` and `audit_delivered_wav` for permission, rendering and preservation checks; extend through an explicitly versioned selective workflow.
- `demo_app`, `demo_runtime` and `demo_review` for the current Streamlit workflow and reviewed export.
- `ProcessingResult`, `AcceptedAudio`, package validation and accepted-only consumption; retain their public shape and statuses.

Add a separate guarded context-Transformer adapter; do not pass its `.pt` checkpoint through an ExtraTrees/joblib loader. Load the saved 81-frame configuration rather than assuming the model class's default window. A new scoped eligibility receipt must bind checkpoint, feature, policy, renderer and verification hashes. Historical selection pointers and failed research flags remain unchanged.

Persist a small selective session containing source/working hashes, sample grid, model/policy versions, revision, protected intervals, candidate IDs and user choices. Each candidate records its full sample support, correction-array reference, technical disposition, reason and verified-preview hash. Keep technical verification separate from `pending/accept/keep` decisions. Do not fabricate calibrated confidence; log tracker support and boundary-view sensitivity with their meanings.

Protection changes invalidate affected previews and final exports. Selection changes clear final confirmation. Every final composition starts from the same working original. If the combined audit fails, block the prepared export and retain the original; do not silently return a failed repair as successful output.

### Two different kinds of verification

1. **Runtime fidelity/preservation:** did the actual decoded WAV realize the permitted requested movement while preserving sample integrity, protected/outside samples and voice support? This is possible without a reference melody.
2. **Reference-based repair benefit:** did the delivered WAV reduce error against an independent/controlled reference? This is an evaluation measurement, not an available truth label for arbitrary uploads.

User listening/confirmation is an operational choice, not independent research approval. Keep unknown intended-note error unknown.

## 5. Ordered engineering milestones

### A. Diagnosis and data declaration — SVR-01 / SVR-02

Persist a reusable trace of **all** attempted cases through raw prediction, reliability, scope eligibility, candidate selection, permission, rendering and final disposition. Include target/support information only on the scoring side. Reuse the planning replay rather than claiming its observations as newly implemented tracing.

Distinguish model error, scope exclusion, wrong-region selection, fragmented support, correction limits, transition protection, rendering failure and delivery-audit failure. The documented overlapping reason counts are not proof of one universal cause.

Reserve new evaluation sources before training/policy work. Check singer identities, source hashes and prior use; all descendants stay with their source partition. Keep existing calibration/pilot results developmental.

### B. Multiple candidates and eligibility — SVR-03

With the existing checkpoint frozen, compare the legacy single-span baseline with an input-only regionizer that retains all supported candidates. Evaluate sign-consistent hysteresis using activation 3 cents, release 1.5 or 3 cents, and gap allowance 0/40/80 ms. These are declared development candidates, not production-calibrated settings.

Never merge across unreliable voice, silence, opposing corrections or protected samples. Never pad a candidate just to satisfy minimum duration. Keep the original scope/transition rules as one policy arm. A separate held-voice arm may distinguish acoustic onsets from pitch-only boundary cues, but must be specified before fitting/evaluation and pass unchanged/expressive/transition controls before selection. Do not simply remove guards on inspected examples.

Keep maximum correction **30 cents**, minimum support **0.86 s**, total changed-support fraction **25%**, and existing agreement limits fixed in the first engineering comparison. The current long-voice scope (1.10 s / detrended spread <=20 cents) is a provisional eligibility rule under investigation, not a musical truth label.

### C. Focused training — SVR-04

Keep the 27 October architecture, fixed-scale features, optimizer and 300-update budget. Run baseline, stronger preservation, target-gradient continuity, correction-limit penalty and combined conditions. Use preservation multiplier 2; normalized gradient and limit-penalty weights 0.1. Record them as initial engineering choices in the declaration before execution; do not claim they came from a paper or final-test tuning.

Use seeds 339/340/341 and four training-side singer folds: female1+male1, female2+male3, female3+male4, female4. Verify these IDs against the current manifest before execution; an identity/count change requires a new declaration. No descendant crosses a fold. Derivative supervision uses adjacent centers with the same inference windows and valid adjacent targets, never unrelated randomly sampled frames or unknown target gaps.

Select only configurations satisfying preservation requirements; then rank by singer-balanced delivered improvement, then useful coverage. Ties favor fewer changes to the existing method. Freeze the common engineering recipe before the proposal-aligned comparison in section 6. Do not make the model larger in this first iteration.

### D. Delivered audio and live adapter — SVR-05 / SVR-06

Run permitted regions through the existing Praat renderer and **encoded** WAV audit. Check unchanged originals, realized movement, outside/protected samples, pause/voicing preservation, sample grid, clipping and boundary behavior. Preserve each failure and rollback.

Only after permitted candidates exist, benchmark Praat against an optional component-local Rubber Band adapter on identical constant-shift supports. Keep configuration/version/dependency/distribution records. A dynamic-curve comparison requires a genuine time-varying adapter, not independently processed fragments. Keep Praat on ties, unavailable dependencies or failed delivery checks. The research paper BERT-APC itself uses Praat-Parselmouth; the baseline is not obsolete merely because a different library exists.

### E. Selective review and export — SVR-07 / SVR-08

Build a small locally bundled Streamlit waveform component with drag protection and labelled time fields as keyboard-accessible fallback. Preserve existing app styling and audio workflow; no CDN or separate application server.

Display repair intervals and reasons, equal-gain original/repaired playback, and per-candidate Accept / Keep original controls. Protect the whole overlapping candidate including the blend. Expose only verified previews as selectable repairs. Reconstruct accepted combinations once from the working original, recheck the combined 25% edit budget and decoded WAV, reassess the selected waveform's provisional acoustic evidence, then request fresh user confirmation.

Reuse the accepted-only package shape with a separate selective user-reviewed demo policy version. No accepted edits means an explicitly unchanged export. A failed combined repair must not be labelled a completed prepared export.

### F. Fresh evaluation, eligibility and rehearsal — SVR-09 / SVR-10

Target four previously unused VocalSet singers, two from each available singer category, selected deterministically from official metadata after prior-use checks. Predeclare straight held tones on vowels a/e/i where present, +/-20/30-cent shifts and two support positions per source. No singer/source substitution after outcomes. Construction failures, scope exclusions, withholds and unscoreable cases remain distinct and visible. Insufficient coverage means insufficient evidence, not permission to replace difficult cases.

Freeze model, regionizer, policy, renderer and verifier before processing the reserved evaluation. Report raw and delivered error, coverage, harmful edits, original/protected preservation, failures, latency and seed variation. Keep singer as the independent unit; three seeds are not three more singers.

Retain the existing per-singer practical criteria (positive permitted gain, response >=0.1, raw-original mean <=3 cents, permitted-original mean <=1 cent, original edited frames <=5%, all-attempt coverage >=25%). The additional selective-release target is at least one **verified delivered improvement in each correction direction** and >=25% all-attempt verified-improvement coverage for every represented evaluation singer. Exact protected/outside sample and sample-grid checks are mandatory for each delivered file. These new scoped targets do not rewrite historical research gates.

Two independent actual listener reviews assess tune/expression and audible artifacts separately. Reviewer availability is currently unconfirmed/deferred; no automated fixture or user checkbox may fill that evidence gap. Engineering can continue, but missing human validation must remain visible.

## 6. Research validation after the scoped prototype

The engineering ablation above does not replace the main study. Register a matched comparison of **original audio, conservative rule correction, augmentation-only Transformer, matched frame-only Transformer and full BPDR**. Use one frozen common preparation/preservation recipe and rendering/controller policy for correction conditions; original audio receives only required decoding/format conversion. ExtraTrees remains a supplementary comparator.

For the proposal-aligned response test, retain MAE `d_M` for boundary consistency and counter-detuning; anchor/continuity refinements are common across the three neural conditions. Keep four-view data, source weights, comparable capacity, budgets, seeds, model selection and evaluation conditions matched. The primary comparison is full BPDR versus augmentation-only, followed by the boundary-free control. Declare the adapted feature version and do not transfer historical 71-feature results to the 32-feature model.

Evaluate false split, missing/merged boundary and jitter separately; expand to the proposal's +/-10/20/30-cent coverage and natural analyser errors. Boundary invariance without useful detuning response is not success. Compare at matched acceptance coverage and report paired singer/source differences with uncertainty separately from training-seed variation. Report null/negative outcomes if BPDR does not outperform the simpler conditions.

Keep independently reviewed reference suitability, ethics/consent, guide-melody/free-hum collection, and unfamiliar-melody evaluation as remaining work. The proposal's eight adults x six phrases is a **planning estimate**, not completed recruitment or a power calculation. A scoped VocalSet holdout remains an engineering evaluation and cannot silently replace this study.

Evaluate the same frozen melody extractor before/after accepted preparation using independent references and explicit onset+pitch / offset-aware precision, recall and F1. GAME execution/MIDI/CSV is interface evidence, not improvement. BPDR may run evaluation adapters, but final musical interpretation remains owned by the melody component.

The proposal's 20-second-phrase / 30-second response-time target remains a development target; benchmark median and p95 on documented hardware before claiming it is met. Retain accessible controls, labelled plots and no automatic research reuse.

## 7. Presentation and completion rules

Suggested sequence, subject to evidence rather than an invented deadline:

| Milestone | Demonstration |
|---|---|
| Working main feature | A real supported voice with controlled detuning survives prediction, permission, rendering and actual-WAV verification |
| New selective feature | Show multiple candidates, protect a region, accept one repair, keep another original and export the verified combination |
| Conservative control | An unchanged original remains unchanged, or an unsafe/uncertain candidate is withheld with a clear reason |
| Evaluation evidence | Show all attempted/scoreable/improved counts, singer counts, preservation, failures and actual listening status |

Choose the demonstration example using a recorded development-data selection rule; distinguish a curated example from evaluation success rates. Multiple regions must fit the existing total-edit budget. Do not simulate model-generated repairs or prefill positive measurements.

Only after implemented and demonstrated may the presentation say: “BPDR prepares vocal input through selective local pitch repair. A Transformer proposes corrections, BPDR verifies the delivered waveform, and the user can protect expression and individually review verified edits.” Add the supported scope and validation limits. Do not describe the whole research contribution as completed merely because this workflow runs.

## 8. Tracking and resuming work

Use [BPDR_Implementation_Tracker.md](BPDR_Implementation_Tracker.md) as the only current task register. Each completed task needs a run/receipt or reviewable change, checks actually executed, denominators where relevant, remaining limits and the next task. Update the tracker and decision log below in the same work session; do not append an identical chronological log to every plan.

Use `results/selective_repair/<run-id>/` for future immutable declarations, hashes, traces, candidate decisions, verification and final export receipts. Keep diagnostics/targets/references separated from callable inference inputs. User decisions and confirmation must bind to the exact waveform/session revision. No model fitting occurs during presentation.

Task status vocabulary: **DONE** = completion evidence exists; **ACTIVE** = declared work underway; **NEXT** = first unstarted task; **PLANNED** = not started; **BLOCKED** = named missing prerequisite; **DEFERRED** = deliberately outside the current delivery. Read-only findings can support a task without completing its reusable implementation.

### Decision log

| ID / date | Decision | Basis / consequence |
|---|---|---|
| D01 / 2026-10-27 | This master and tracker become the active plan | Owner requested consolidation before implementation; old plans retained as history |
| D02 / 2026-10-27 | Keep the original BPDR research question | Proposal pp14–22; GUI/ExtraTrees improvement cannot substitute for the matched objective test |
| D03 / 2026-10-27 | First scope is stable held voice; BPDR app only | Explicit owner selections; shared PP1 implementation remains untouched |
| D04 / 2026-10-27 | Drag protection + time fields; whole-candidate veto | Explicit owner selections; avoid fragmented edits around protected samples |
| D05 / 2026-10-27 | Diagnose eligibility and region selection before larger models | Reproducible low raw error but zero useful permitted correction; 7/10 scored regions excluded by scope |
| D06 / 2026-10-27 | Retain Praat as baseline; renderer comparison conditional | Current implementation and proposal use Praat; the new checkpoint has not yet reached rendering in the replay |
| D07 / 2026-10-27 | Separate runtime verification, reference scoring and human review | An upload lacks known intended-pitch truth; technical checks and confirmation are different evidence |
| D08 / 2026-10-27 | Preserve source/model/evidence/old-plan history | Documentation-only consolidation; no checkpoint adoption, gate reset or original-data mutation |
| D09 / 2026-10-27 | Clarify feature/loss/data deviations | Context redesign and squared response terms are engineering variants; preserve a proposal-aligned comparison |
| D10 / 2026-10-27 | Prioritize model improvement and an after-training accuracy command | Explicit owner instruction. MOD-01 runs the declared five-objective, four-fold, three-seed training-side comparison with the unchanged legacy controller; no fresh final-data or UI claim. MOD-02 verifies standalone checkpoint evaluation. Broader SVR-02/03/04 workflow remains separately gated |
| D11 / 2026-10-27 | Complete comparison/checking; retain candidate as diagnostic | All 60 CV fits and 3 final refits completed. No objective passed every training-side held-out singer. Combined preservation improves at a repair-error cost; standalone rendered recheck reproduces default calibration with 0/16 verified gains. No adoption. Next SVR-02 reserves fresh sources and declares broader model work; SVR-03 remains a separate frozen-model controller study |
| D12 / 2026-10-28 | Prioritize usable Transformer before other feature work | Explicit owner instruction. MOD-03 declares broader training, a new input-only activity/correction architecture and separately versioned held-voice eligibility. Numerical/output limits remain fixed. Additional actual longer-deviation scope is explicit, not padded runtime support; all prior negative conditions retained |
| D13 / 2026-10-28 | Reserve four new final singers before acoustic access | Prospective amendment assigns female8/female9/male8/male9 to final, existing IDs5/6 to historical development, IDs7 to training. No new evaluation outcomes were used; final target creation/inference is blocked until locked development gates pass |
| D14 / 2026-10-28 | Retain failed independent usability result; no activation | 47 neural fits and a stricter development-calibrated guard produce 5/48 final improvements, with all 12 originals unchanged. All four final singers fail. The genuine step-feature Transformer is a distinct engineering variant; neural superiority over its bare cues and the proposal objective remain unestablished. |
| D15 / 2026-10-28 | Make assessment and freshness explicit | A prospectively documented counting correction leaves wholly unscoreable singers unassessed while retaining every attempt and original. All four final singers are scoreable and fail regardless. The final cohort is now inspected; the standalone repeat is reproducibility evidence. Future model/policy changes require another reserved cohort. |

### Reading and execution references

- [Planning review and precedence](Planning_Review_2026-10-27.md); [source proposal](IT23141506_J26_DS_339.pdf).
- [Latest Transformer improvement results](Transformer_Improvement_2026-10-27.md); [accuracy workflow](Transformer_Accuracy_Workflow.md); [initial Transformer recovery](Transformer_Recovery_2026-10-27.md); [current working-demo guide](Working_Demo_2026-10-19.md).
- [Research status](RESEARCH_STATUS.md); [proposal/implementation matrix](RESEARCH_GAP_AND_IMPLEMENTATION_STATUS.md).
- [Preserved supplied research brief](references/Selective_Repair_Brief_2026-10-27.txt). It contains original unresolved citation markers; use the proposal and verified primary links, not those markers, as citations. Current source facts and this plan override assumptions in that brief.
- Canonical six notebooks: [notebook index](../notebooks/index.html) and [notebook guide](NOTEBOOK_GUIDE.md). Open via `Open-Notebooks.ps1`; verify via `tools/execute_research_notebooks.py`. Retained notebook 07/old-suite references in historical reports do not describe the current notebook layout.
- Primary background: [BERT-APC v3](https://arxiv.org/html/2511.20006v3), [Deep Autotuner](https://arxiv.org/abs/2002.05511), and [Rubber Band documentation](https://breakfastquay.com/rubberband/code-doc/classRubberBand_1_1RubberBandStretcher.html). Their results are not BPDR results or grounds for expanding into symbolic musical inference.
