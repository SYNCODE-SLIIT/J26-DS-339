# BPDR execution plan

**Rebased 12 October 2026.** Follow this plan from the current working foundation; do not restart or delete verified work. [Master plan](PP1_Master_Plan.md) defines the method and [evidence tracker](PP1_Evidence_Tracker.md) separates implementation from research results. Dates below are working targets, not completed gates or supervisor approval.

## Immediate sequence

**Owner direction — reviewers unavailable:** [continue without experienced review for now](Reviewer_Deferral_2026-10-14.md). Proceed with Gate F controls/diagnostics and B/C/E engineering evidence, preserving exploratory labels and pending reviews. Gate D/J4 is deferred until reviewers are available, not an immediate action item that stops engineering. Formal musical validation and research-ready anchors remain uncredited; the future review workflow below is retained.

**Current work gate: Gate F — usable delivered audio (renderer validation), open.** Include this gate label and a concrete gate-specific next action in progress/next-step reports. Gate D/J4 independent reference approvals are deferred until reviewers are available; Gate B voice/support coverage and Gate E matched research fits remain open. Gate F engineering proceeds now on saved controlled evidence. No research or delivery gate is credited from fixtures alone.

**Coverage audit -> common dataset path -> screened multi-source sample -> actual reference reviews -> matched learning -> conservative delivered audio -> independent evaluation -> accepted-only integration.** Model/controller engineering and contract fixtures can proceed while reviews are pending. Research fitting requires the reference gate. A calendar date never overrides a gate.

| Priority | Target window | Deliverable | Exit requirement |
|---|---|---|---|
| 0 | 12 Oct | Clean audited component and one current plan | Source/data retained, recovery archive verified, regression passes |
| 1 | 13-14 Oct | Training-only intervention coverage report and common dataset builder | Source adapters and schema checks; broader audited supports or explicit limited-pilot result |
| 2 | 14-16 Oct | Initial Jamendo pilot, screening and review queue | Five artists first; increase to 20-30 only after quality/yield inspection; report exclusions |
| 3 | 13-18 Oct | Usability policy, accepted-only contract adapter and repair-controller/renderer fixtures | Identity, bounded edits, timing, rejection and rollback checks; numerical settings remain provisional until calibration |
| 4 | 17-21 Oct | Three-condition engineering runs and end-to-end pilot | Matching budgets, saved predictions and reloadable checkpoints, exploratory labels |
| 5 | 22-25 Oct | Prototype evidence pack and rehearsal | Working unchanged/stop paths; label any candidate repair and unpassed research gates |

Reviewed fitting/evaluation joins this window only if approved multi-source anchors and separate development data become available. The proposal's Oct 26-28 PP1 figure and prior Oct 25 planning target are distinct; the official slot is unconfirmed. Full controls/study/integration remain the proposal's Nov 2026-Apr 2027 estimated work. Preserve the original Sept 27 baseline in the recovery snapshot; this schedule does not imply earlier planned phases completed.

## Gate A: preserve and verify the foundation

- Keep implementation, tests, locked dependencies, launchers, maintained notebooks, both raw datasets, checkpoints, reviews, source manifests, relevant failure history and extraction evidence.
- Remove only dependency-checked disposable files within this component, after archive/hash verification. The 12 October cleanup is recorded separately.
- Run all component regressions. Confirm current manifests, review pointers and notebook pointers resolve; retain historical snapshots as historical.
- Use existing `Run-Phase1.ps1`, notebook launch and extraction launcher for current behavior. Do not rebuild a dataset run containing reviews in place.

Output: current inventory, retention check, test log and [cleanup report](Cleanup_2026-10-12.md). No new component behavior is implied.

## Gate B: fix intervention coverage before counting training material

**13 October measured update:** [the first coverage audit](Coverage_Audit_2026-10-13.md) finds potential support in six singers for shorter candidates, but only three singers with any passing shift and one with all six in a six-singer probe. Two -10-cent rejections involve PESTO's coarse pitch-bin stepping despite fine YIN agreement. Keep v0.3 active; Gate B remains open. Diagnose the coarse-consistency rule and inspect further crops while Gate C builds the common source path. No review or quality threshold was bypassed.

**Fresh Jamendo update:** [the five-artist sample](Jamendo_Five_Artist_Sample_2026-10-13.md) finds v0.3 support in three crops/two artists. Five of eighteen variants pass across two crops/two artists, and none passes all six. Fine shift error and loss of common voiced audit coverage are additional failures. Diagnose renderer/tracker behavior and support selection under the existing error/preservation requirements before expanding or counting training pairs. Artist proxies and model masks do not replace reviewed reference voices.

**14 October diagnosis:** [all eighteen decisions replay](Audit_Diagnostics_and_Review_2026-10-14.md); local-versus-full comparisons rule out crossfade contamination of the tested audit interiors. Controlled vibrato exposes phase-vocoder fine errors. A declared FFT/hop comparison gives 6/18, 10/18 and 9/18 passes, but loses some active successes and no crop passes all six. Keep the active renderer/policy; extend versioned calibration and audible checks on existing training material before adopting a candidate. A nine-crop, two-sheet independent review bundle is ready; actual human decisions remain pending.

Current audit v0.3 admits suitable supports in only three Vocadito training crops from one singer. That is insufficient for a general multi-singer learned claim. Review effort alone cannot fix this geometry.

**Wider calibration completed 14 October:** [108 real-crop and 216 constructed-control measurements](Renderer_Calibration_2026-10-14.md) retain those limits. Both FFT-1024 candidates lose an active Vocadito success; hop 128 still fails two wide/fast vibrato controls. No renderer is adopted. A labelled 36-clip comparison page is ready, and 81 regression checks pass. Next, collect the actual independent original-reference decisions and audible observations, then address independently identified voice support under a declared policy. Keep model/controller fixtures exploratory while reviews are pending; do not resume extraction or research fitting from aggregate candidate gains alone.

1. On training sources only, enumerate reference-reliable spans and stable pitch/transition masks; count singers, recordings and crops independently.
2. Evaluate shorter plateau/guard candidates as a new versioned policy. A 0.5-second support is a coverage hypothesis, not an approved setting. Keep existing fine/coarse realized-shift and outside-drift tolerances unless measured justification supports revision.
3. Re-run zero-change identity, outside-support equality, plateau coverage and realized-shift audits for each candidate. Inspect artifacts/transitions; record why candidates fail.
4. Freeze a justified geometry before development/evaluation. If no candidate admits multiple independent sources, retain v0.3 and explicitly limit learning to an engineering pilot; collect suitable approved references rather than relaxing quality checks to force eligibility.

Output: policy comparison with independent-source denominators and exclusions; new audit version only if justified. A planning target is >=5 independently sourced training voices with audited supports for an initial multi-source pilot, not a statistical sufficiency threshold. Main-study size comes later from precision/power.

## Gate C / P2J: build the component's data path on the pinned models

**13 October implementation:** the shared builder, typed source adapters, pinned GAME adapter, screen/masks, pending review queue, integrity verifier and listening index are available. After the initial six-crop pilot, [five new training artists](Jamendo_Five_Artist_Sample_2026-10-13.md) completed full-song GPU separation and GAME with individual receipts. The current common run has ten unchanged outputs, nine screen candidates, eighteen pending reviews and zero research-ready examples. Resume guards reject different separator settings or changed frozen selection metadata. Keep the working unedited melody mirror as the shared-model/format reference. The table remains the full exit criteria; measured preliminary yield does not satisfy single-voice/leakage calibration or reference approval.

| Planned responsibility | Required behavior | Meaningful check |
|---|---|---|
| Source adapters | Reuse Vocadito loader with human references; load Jamendo metadata/audio without invented annotations | Human/model reference kinds stay distinct; missing files/licenses fail explicitly |
| Grouping and selection | Freeze singer/source/artist partitions before crops; deduplicate audio; initial one song per artist | No source or descendants cross partitions; unknown singer identity recorded |
| Separator adapter | Verify pinned BS-RoFormer hashes/settings, use GPU, preserve vocal stems | Resume checks source/model/settings hashes; failed separation creates no eligible example |
| Crop preparation | Mapped <=20-second windows; preserve source offsets/rests; run Phase 1 | No silent source truncation; rate/sample-count mapping and accepted hash validated |
| Boundary adapter | Pinned GAME on accepted crops; retain fractional `model_midi_pitch` and raw seconds | +12 shifted field cannot be read accidentally; empty notes excluded; every decode saved |
| Suitability screen | Voicing/reliable support, leakage/overlap indicators, boundary ambiguity and exclusions | Silence/choir/leakage/weak pitch fixtures are withheld; screening is not reviewer approval |
| Manifest/review packaging | One immutable record shape for both datasets, versioned masks/references, lineage and independent reviews | Source changes invalidate derived record; pending/rejected examples cannot become research eligible |

The existing-stem plumbing stage and frozen five-artist sample are complete. Seven Jamendo crops were prepared from four artists; six crops from three artists enter review. One song has no five-predicted-vocal-second window; one crop is withheld for low voicing/reliable pitch. Three crops/two artists support v0.3 probes; five of eighteen variants pass and no crop passes all six. Next, review the candidates and diagnose renderer/audit coverage, then decide whether an independently sourced 20-30-artist expansion is justified. Keep song/artist/crop/frame counts separate and predeclare expansion criteria. The observed fractions are descriptive screening yield, not population suitability or accuracy.

J1 (execution) and J2 (GAME on Vocadito) are existing pilot evidence. J3 (representative stem suitability) remains open. J5 (excerpt/full-song equivalence) is optional; keep full-song separation as baseline until repeated across windows/artists. Cut J5 first if short of time.

## Gate D / J4: actual approved reference anchors

**Deferred by owner, 14 October:** no musically experienced reviewers are currently available. The numbered steps below are for future execution when reviewers are available. Current engineering work uses automatic checks, known controlled targets and existing typed references, without fabricated approval or a research-loader bypass. Research-ready examples remain zero. See [the staged no-reviewer workflow](Reviewer_Deferral_2026-10-14.md).

1. Prioritize sources that can host audited detuning, then send independent review sheets to two musically experienced reviewers through the owner's research workflow. Do not fabricate or infer their approval.
2. Review acceptability, stable support, uncertain transitions and permission to preserve the reference. Record distinct identities, decisions, agreed intervals and reasons resolving disagreement.
3. Preserve human annotations for Vocadito. For Jamendo retain `model_estimate` provenance even after a recording is judged acceptable; any additional human pitch annotation is a separately documented task.
4. Generate reviewed A/B/C/D pairs only through verified hashes, interval intersection, source partition and realized-shift audits. Keep failed/empty gate outcomes traceable.

Exit: enough independently approved training/development sources for the declared pilot, with actual audit-passing pairs. Otherwise engineering mode continues and research mode must refuse the data. Participant collection waits for ethics/consent; Jamendo cannot be silently substituted for the proposal's real-user study.

## Gate E: fair models and measurements

Retain current `model.py`, `losses.py`, `examples.py`, `metrics.py`, `training.py`. They are a tested scaffold, not a completed study.

1. Add source-weighted sampling/aggregation so long songs or many variants do not dominate; record dataset role and counts.
2. Add development checkpoint selection and separate evaluation loading. Freeze seeds, budgets, weights and normalizer fit on training data only.
3. Compare augmentation-only (71 features), frame-only (70 features) and BPDR under matched budgets; retain original and declared conservative-rule references. Warm-up and common anchor/smoothness terms are controlled. Initially disable denoising; any later conditional noise policy is fixed/shared across correction conditions, while the original control gets only necessary decoding/format conversion. Log noise preparation separately from pitch repair.
4. Add a dedicated no-change metric path; acceptable singing without injected-detuning masks must remain in preservation reporting.
5. Save raw predictions plus checksums, selected checkpoint and settings. Engineering runs stay labelled `exploratory_not_research`. Research runs require Gate D and a new recorded collection version.
6. Separately evaluate annotated controlled boundaries, audio heuristics and GAME estimates. Record decode variation; do not select the most favorable GAME decode.

Exit: reloadable conditions, fair protocol and independently executable metrics. Training-fit improvement alone passes no research-effect gate.

## Gate F: usable delivered audio

**Current completed step — Gate F, 15 October:** [frozen control diagnosis](Known_Control_Diagnosis_2026-10-15.md) reproduces all **16 guarded audits and 32 transition peaks exactly**. The glide reject is one positive PESTO change of +0.000413 cents at 1.24 s against a -15.6-cent request; its sign remains positive in float64, and native/aligned timestamps agree. YIN measures -17.420747 cents. The reject remains; no epsilon is adopted. Existing transition peaks measure pitch-change magnitude, not target error. Separate known-control center-law residual maxima are 5.422835/7.289706 cents at the fixed 2048/1024 windows; these remain finite-window diagnostics, not auditory thresholds. All **147 regression checks pass**. No WAV or renderer is created or changed; all eight measured and 24 withheld cases remain. **Next — Gate F (open):** Run the declared audio-only source-estimation feasibility on the eight existing constructed sources; withhold construction F0/phase from the estimator and use it only in scoring. No new rendering or vocal trial yet. **Gate D/J4 remains deferred; zero research-ready examples or accepted repair.**

**Prior completed step — Gate F, 15 October:** [known-source analytical audio controls](Known_Source_Control_2026-10-15.md) render the eight preselected strict cases and explicitly withhold 24. Seven of eight full audits pass, matching the baseline count on those same eight cases, but the passing cases differ: constant-880 is gained and glide-220 is lost to one opposing coarse-direction frame. All eight fine-pitch/preservation checks and eight exact identities pass. Maximum transition estimates improve on the matched subset, yet four individual 2048-window cases and glide at 1024 worsen. Source-relative amplitude dips improve across 18 ramps; this uses a different denominator from the older branch-relative diagnostic. All **141 regression checks pass**. These controls require known construction F0/phase and are not a practical vocal renderer or musical accuracy. The subsequent frozen diagnosis is recorded above.

**Prior completed step, 15 October — source-anchored feasibility:** [conditional phase/pitch constraints](Anchor_Feasibility_2026-10-15.md) retain all 32 constructed cases/36 supports. Nine supports across eight cases have strict fixed-basis mathematical solutions. Eighteen supports fail the sign/magnitude-preserving necessary budget; nine more are unresolved by this basis, rather than proven universally impossible. Relaxed +/-30-cent solutions can reverse a requested correction and are diagnostic only. All **134 checks pass**. The subsequent known-source audio control is measured above.

**Prior phase-aware polar-join comparison:** [the fixed 32-case candidate](Phase_Join_Comparison_2026-10-14.md) removes measured fundamental amplitude dips, but increases transition peak counts/maxima at both fixed YIN windows. Guarded audits are identical to baseline (26/32); all fine/physical checks and eight identities pass. The unchanged advancement filter refuses it. That stage passed **129 checks**. Retain its failures without retuning or inferring delivered quality from amplitude improvement.

**Prior phase/amplitude feasibility:** [fixed diagnostics](Phase_Feasibility_2026-10-14.md) reconstruct all 64 existing waveforms, measure 72 ramps per method and pass eleven known controls. Destructive fundamental mixing is observed; it is not an auditory rating or proof of a sole cause. That stage passed **123 checks** and remains evidence.

**Prior completed step — boundary-taper comparison:** [the fixed 32-case experiment](Boundary_Comparison_2026-10-14.md) increases aggregate full passes from 26 to 27, but loses vibrato-220/crossing and worsens the shorter-window maximum transition peak. All fine/preservation checks and eight exact zero identities pass; the declared advancement filter refuses the candidate. That stage passed **119 checks**. Preserve both windows and all baseline decisions; do not tune the refused taper setting or proceed as if it validated real vocals. A bounded training-only vocal/listening comparison waits for justified controlled evidence. No renderer adoption, research pair or accepted repaired package is credited.

**Prior completed direction/transition diagnosis:** [frozen evidence](Framewise_Diagnostics_2026-10-14.md) reproduces all original audit values and local samples. The 26/32 baseline pass count is unchanged; all 17 opposing coarse changes in six rejects are below 0.001 cent, with no acceptance-rule change. Transition contrasts demonstrate a blending contribution to measured peaks and estimator-window sensitivity. That stage passed 112 checks and closes no research/delivery gate.

**Framewise candidate tested 14 October:** [the controlled Praat/Parselmouth experiment](Framewise_Renderer_2026-10-14.md) passes 26/32 full audits and all 32 fine-error checks; six per-frame coarse-direction rejects remain. Eight exact identities and one supplied changing-curve transaction verify, with 108 regression checks passing. Count/time/outside/rest preservation is measured on controls only. The candidate and new per-frame audit remain separate from active v0.3. Next, diagnose the rejects and predeclare a bounded existing-training-vocal/listening comparison; no production adoption or accepted repaired package is credited.

**Prediction adapter completed 14 October:** [frozen checkpoint to permission preview](Prediction_Adapter_2026-10-14.md) preserves 70/71 inputs, constructs sign/gap-separated candidate intervals and checks worst alternate-view disagreement. One seen S29 crop gives 940-frame curves, maximum about 0.17 cents and no edit request; six constructed adapter cases and 102 regression checks pass. Missing/uncalibrated uncertainty, expressivity protection and production policy still prevent runtime edits. No audio is rendered, model fitted or accepted package created. Next, evaluate a duration-preserving framewise renderer candidate; permission/uncertainty calibration and actual reference reviews remain open.

**14 October engineering progress:** [permission/rollback fixtures](Controller_Fixtures_2026-10-14.md) verify nine constructed outcomes and 93 regression checks. Invalid proposals are withheld without clipping; physical/post-pitch failures roll back exactly. This is a separate fixture harness with no accepted package. Runtime uncertainty/proposal generation, calibrated limits, the framewise renderer, expressive preservation and accepted-only delivery remain open; the active data path and v0.3 renderer/policy are unchanged.

1. Specify provisional permission policy: reliable support, correction magnitude/affected fraction, uncertainty and alternate-boundary agreement, protected transitions, rests and expressive cases. Calibrate values on development data; over-limit proposals are withheld rather than clipped.
2. Implement the proposal's duration-preserving local renderer (Praat/Parselmouth candidate), independently of the artificial detuning generator. Record its version and decisions.
3. Audit identity, exact sample-count/time origin, plateau realized shift, protected/outside drift, transitions and audible artifacts. Measure latency on the demo device.
4. Remeasure every candidate; rollback failures. If the original is already suitable, withheld edits may return unchanged. Otherwise return re-record/failure; never expose failed audio as accepted.
5. Emit `ACCEPT_PREPARED` only after verified permission and post-render checks, with edit intervals/provenance. The four statuses require independent acceptance tests.

Exit: declared controller/renderer behavior and measured preservation. Constructed tests prove mechanics; human musical suitability and independent effect remain separate.

## Gate G / H1: independent study and accepted-only hand-off

- Agree the versioned package with melody owner IT23439078. Validate status, hash, WAV format, rate/count, timing/crop origin and diagnostics. Rejected packages stop downstream. Retain warnings and operation log.
- GAME remains a declared baseline until the melody owner accepts it or supplies their extractor. H1 currently passes only for unchanged technical preparation on 17 crops.
- Extend H1 to actual prepared audio on train/development first, then on the held-out evaluation partition once after settings freeze. Require preserved timing/rests and unchanged non-target extraction within repeat variation; evaluate edited pitch against the declared correction reference/direction, not equality with the detuned input. Compare before/after against human references and GAME's own repeat variation.
- Measure correction error, boundary sensitivity, no-change harm, outside/transition preservation, rendered artifacts, edit/acceptance coverage, failures and latency jointly. Report onset+pitch and offset-aware extractor metrics with conventions.
- Paired source/singer-level analysis and 95% intervals; at least three seeds where feasible. Keep controlled synthetic detuning, unmodified expressive cases and Jamendo robustness separate. Blind listening has fixed rendering/playback conditions.
- Claim advantage only if BPDR improves the declared repair endpoint and satisfies a predeclared preservation margin at comparable coverage against both learned controls. Otherwise report inconclusive or negative results.
- Later test downstream error propagation with raw, prepared and reference melody under fixed planner/refiner settings. Their chord-family/function contract conflicts are team-owned blockers, not BPDR tasks.

## Demonstration and longer study

For October, show original preservation, accepted unchanged input, reasoned stops, the boundary-only experiment and accurately labelled candidate/engineering repairs if ready. Keep a traceable offline replay for hardware failure. Show unpassed gates and independent-source counts; downloads, tests and variant counts are not completion percentages.

After the prototype: obtain approved references/participant permissions, freeze a feasible protocol, train all controls, perform the held-out study, assess real-user/unknown-melody scope, and integrate with the owners' actual models. Follow the proposal's later estimated windows, revised with the supervisor when evidence and official deadlines are available.
