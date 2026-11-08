# BPDR implementation and research tracker

**8 November balanced comparison, provisional first pass.** All twelve
fits exist. Saved aggregation reports BPDR raw positive singer-macro MAE
21.387088 cents versus augmentation-only 23.246071 and frame-only
24.905712 cents; 11/17 scoreable groups
fail a required BPDR raw criterion. `advance_to_delivery=false`. Independent
checkpoint/prediction replay remains live in session 36082, with no final
receipt yet. Jitter boundary sensitivity is worse than augmentation-only
(0.257566 versus 0.108847 cents). No delivery qualification or activation follows. The new
stratified reporter is syntax checked and awaits that receipt before
execution; identity controls are also predeclared and await parent replay.

**8 November request/permission/output readout.** The selected version now
shows measured decoded sample changes separately from model requests and
permitted preview requests. Original selection is zero-change even with a
valid preview. Cleanup is labelled separately; withheld files gain no
permission from existence. Export checks the summary's selected role/hash.
Eight focused tests and all 43 affected regressions pass. Six UI state/role
pairs independently replay with exact summaries/export hashes (25090 exit 0).
See [the result record](Request_Delivery_Check_2026-11-08.md).

**8 November time-specific feedback.** The review UI now shows measured
quiet ranges, near-full-scale peaks and uncertain pitch-measurement spans.
Pauses and expression remain neutral; acceptance/repair gates are unchanged.
Eight focused tests pass, and a four-state UI check independently replays
(35926 exit 0), including malformed-source handling. Feedback metadata
survives export while original/prepared WAV hashes remain unchanged.
See the [result record](Recording_Feedback_Check_2026-11-08.md).
The full updated regression suite passes: 35 tests, session 5430 exit 0.

**8 November real-hum workflow.** Upload, simulated recording-widget bytes,
controlled-rain preview/protection/undo/export and silence refusal pass
through the actual review UI. Fresh-process replay matches all consequential
results (session 21082, exit 0): eight route executions and six actual pYIN
calls on one already opened training hum and its derivative. The fixed
spectral example delivers +2.45525 dB paired SNR with preservation checks
passed. All selected/consumer hashes match; exported confirmations are
automated fixtures, never human approvals. See the [result record](Real_Hum_Workflow_Check_2026-11-08.md).
No live microphone, novice/human review or general cleanup qualification.

**8 November local-edit review.** The opt-in review UI now supports
selection of existing permitted local edits, rejection of all edits,
undo and fresh final-combination PCM24/acoustic auditing. Failed audits
retain the previous usable version. Twenty-seven backend/UI tests pass.
The fixed two-edit synthetic control passes eight real decoded-WAV audits
across create/fresh-process verify; all hashes and acoustic results match.
Its combined export retains selection metadata and reaches frozen pYIN in
two matching-hash calls. See the [handoff result](Accepted_Pyin_Handoff_Engineering_2026-11-08.md#local-edit-selection-and-final-combination-verification).
This is engineering evidence, with no participants or trained-model
accuracy claim. Real-hum qualification and shared PP1 orchestration remain open.

**8 November full cleanup extension.** Session 98279 completed all
189 opened development cases (378 owning-consumer calls). Independent
saved-metric aggregation reproduces all twelve summaries (error <1e-12).
At 0 dB onset/pitch F1 gains are +0.00333/+0.01293 for A1/A2, with both
95% singer intervals crossing zero and 35/84, 36/84 harmful cases.
Clean harms are 4/21, 3/21. The fresh-process 378-call replay completed
in session 88786 (exit 0), matching every metric/preprocessing record
exactly and independent aggregates within 6.94e-18. It retains all raw
consumer outputs. Inputs match the
earlier audit/export receipt. No reserved data or shared code changed.
Always-on neural cleanup remains inactive; see the full cleanup result.

**8 November selected-WAV handoff engineering.** New opt-in
`repair/primary_handoff.py` calls the frozen owning pYIN consumer in a
bounded child process from an explicitly reviewed package. Synthetic
original/prepared exports have different hashes; four real consumer
calls (two repeated per selection) consume exact selected bytes and
match results excluding runtime. Receipt:
`results/demo_sessions/2026-11-08_primary_selected_handoff/integrity_receipt.json`.
No human review or improved-audio quality is claimed. Twenty
review/handoff/protection tests pass, including actual UI selection,
protection veto and undo. The new opt-in `Run-Review-Demo.ps1` uses a
separate review UI; historical locked Python sources are restored exactly.
Current-source original/prepared/protected-original exports and six real
consumer calls replay under `2026-11-08_primary_selected_handoff_v3`.
This is local demo integration, not completed shared PP1
orchestration or qualified pitch repair. No owning extractor was edited.

**8 November consumer check.** Three fixed development cases and both
note annotators were evaluated through the actual frozen pYIN consumer.
Six calls replayed. RPA improves while onset/pitch note F1 declines
on all three cases for both annotators. Cleanup remains unqualified.
See the [consumer result](Usable_Input_Plan_Status_2026-11-05.md#8-november-actual-consumer-compatibility-check).

**8 November corrected comparison.** All three matched arms passed
split/merge/jitter training smoke checks and held inference. Twelve
new fits and checkpoint replay were started in session 35400 under
`2026-11-07_balanced_short_objectives`. The balanced anchor is common
to all arms. The full BPDR response losses are unchanged. No historical
failed result or qualification criterion is replaced.

That session stopped after five intact fits on a two-ULP frame-only
duplicate prediction difference (2.33e-10 cents, one endpoint frame).
The no-fit diagnosis reproduces identical tokens and exact repeated
outputs. Session 36082 now uses the separately declared numerical resume
wrapper: four-ULP duplicate-output assertion, exact token and checkpoint
prediction replay, unchanged raw predictions, fits and scientific gates.
All twelve fit receipts are now hash-verified (four per arm). Final held
aggregation and complete independent replay remain;
no candidate is qualified.

The eight-case identity-resynthesis extension is implemented and predeclared
for correct held checkpoints after the parent replay completes. It compares
all arms/perturbations with no fits and no cohort selection. Source roles,
unsupported cases and numerical input equality remain explicit.

**Training prerequisite improvement.** The equal-class anchor diagnostic
reaches 3.062 cents MAE and 0.8406 signed response on the fixed training
batch, compared with 22.727 cents under the original weighting.
Zero-target mean request is 1.184 cents. Exact replay matched.
No new generalization or delivered-output qualification is claimed.
Split, merge and jitter now have audited input support on all 498
records, with actual acoustic-token equality and meaningful changed
boundary contexts. The corrected comparison now has its own frozen
declaration and implementation and is running as recorded above.

**Complete PCM24 rescore.** All 189 candidates and 567 estimates finished,
with independently reproduced aggregates. One estimator score changed
versus FLOAT. The old neural gate still fails PESTO preservation,
including an accepted clean-control harm. Export safety is verified
for these files. Cleanup qualification remains incomplete.

**E02 diagnostic, 7 November.** A fixed training batch assigns one sixth
of anchor mass to nonzero targets. Its 200-update fit remains near zero,
matching the stopped augmentation-only arm's failure on that batch.
Independent replay matched exactly. Further full comparison fits await this
diagnostic. The full PCM24 cleanup export rescore is also active.
The original full BPDR terms are now being tested on the same fixed
batch after the declared warm-up.

The 400-update full-objective diagnostic completed. Warm-up matches the
control, but positive response stays near zero after enabling the terms.
The six historical comparison fits remain retained and stopped.
The 21-clean-control waveform alignment check shows 103/105 zero-lag
peaks, with two nonzero local peaks requiring further inspection.

**7 November delivery update.** All 189 neural candidates have new verified
PCM24 exports. The seven former over-full-scale candidates reproduced their
21 FLOAT scores before rescoring. One delivered PESTO score fell by 0.222
percentage points and the other 20 matched. This is bounded export evidence.
Full inference replay, delay, listening and qualification remain open.
The independent reviewer brief is ready in `output/pdf/`, with trial audio
and actual ratings pending.

**Updated:** 7 November 2026  
**Active specification:** [BPDR master plan v1.29](BPDR_Master_Plan.md)  
**Current position:** **MOD-03 INCOMPLETE — no qualified short-hum repair candidate.**
The owner targets January 2027, keeps free humming essential and permits optional
guide mode. Follow the [7 November continuation recommendation](Continuation_Method_and_January_Plan_2026-11-07.md)
and [finite experiment protocol](Next_Experiment_Protocol_2026-11-07.md). Qualification,
independent review/collection, useful preparation and accepted-WAV integration
remain required; broader component progress is not synonymous with model accuracy.

**Latest audit:** five current result files match their replay receipts; both
noise-study aggregate tables reproduce; 189 enhanced WAVs pass shape/finite checks
but seven exceed full scale. Neural gating harms one clean PESTO control. Fifty
existing tests pass. The original all-tracker neural gate remains failed. The
objective comparison has six intact fits out of twelve and no final comparison.
No runtime/model/default was changed. Earlier “active job” statements below are
historical and are superseded by the current continuation record.

## 1. Current evidence baseline

V70 raw/query replay completes:coverage guarantees all original queries, but
correction error21.529c versus20.995c and original means nearly unchanged;raw gates
fail. V71 WAV checks22672 and positive/head diagnosis69538 are active. No sweep.

V68 raw replay is COMPLETE:8 parent/new checkpoints,996 held prediction arrays
and4 exact traces. The tail-loss correction error20.995372c versus19.701240c is
worse; original means.766431c versus1.051955c are lower. Both raw gates fail.
V69 independently replays498 new plans,178 candidate audits,152 encoded outputs,
445 reference records and53 original controls;498 baseline records are inherited
under verified identity, not rerendered. Tail/uniform32/24 beneficial repairs,
109/148 reference harms,101/143 known-zero harms,11/11 changed originals. Remaining
23.902523/24.028204c from24.896935c input. This is a genuine measured delivered
tradeoff/improvement, but not usable qualification.80654 is terminal0.

V67 query audit2752 also completes:26836/40899 reliable training frame/model
positions directly queried as originals,29734 with exact C-zero token equivalents.
782/1548 activation-tail positions and440/748 proposed original positions lack
those queries.115200 direct-original draws and57600 C-zero draws are accounted
for across folds. Repeated model/frame positions are not independent people;
similar-input gradients and late optimization drift are not measured.

V70/V71 are declared and active in session22672:4 coverage-sampler fits -> raw
checkpoint/query replay -> fixed actual-WAV evaluation/replay/report. Three sampler
tests pass, including full-pass-before-repeat and budget refusal. First fit reaches
1200 updates; its general-original trace queries all10552 eligible frames within
14400 draws. Queries are not guaranteed nonzero gradient updates under top4 loss.
No new model score/qualification result yet. V71 frozen tail baseline reuses V69
only after data/policy/prediction identity. No extra seed/epoch/threshold sweep.


V68's4 fits finish with initial correction-error/original-preservation tradeoff
(20.995c/.766c versus19.701c/1.052c); raw gates fail, replay/V69 remain active
in80654. No further parameter sweep or activation.

V66/V67 independent replay completes; both candidates remain rejected. V68/V69
are now live in80654 under the loss-only specification linked below. No activation.

V65 raw comparison is independently VERIFIED:8 checkpoints,996 held arrays,
8 sampler traces and2988 training-only teacher arrays reproduce all summaries.
Uniform/mined overall error19.701240/19.958749c, HumTrans16.923916/18.708845c,
Vocadito20.858458/20.479542c; both raw gates FAIL. Within10c singer-macro fractions
are.262964/.241584; unchanged-command baseline error24.905844c. Original means
are1.051955/1.243259c, hiding activation tails in33/37 original windows and maxima
27.580/27.500c. Raw known-zero activation frames8079/10159 are repeated derived
frames, not independent people or rendered harm counts. Paired singer bootstrap
for mined-minus-uniform MAE is+.257509c, descriptive95% interval[-.678951,1.307409];
this does not establish population superiority for either sampler. No mining adoption.

Pipeline44844 is terminal1 AFTER successful checkpoint replay:the raw reporter
assumed every technical working clip had reliable pitch frames. `vocadito_8` has
zero such frames; it is retained with null pitch statistics, not scored as zero
error.52/53 original windows have measurable reliable-frame statistics; all53
remain actual-WAV identity controls. `raw_report_failure.json` preserves the failure.
No new fit was needed. V67 prepared diagnosis/replay also preserve this distinction.

Current pipeline session83651 is RUNNING from corrected raw reporting onward:
raw report COMPLETE -> V66 delivered evaluation ACTIVE -> independent output replay
-> domain/duration/direction report. Check the durable pipeline status and poll83651;
do not rerun training or replay because old44844 is terminal. V66 remains fixed and
no actual-output result or activation is claimed yet. V67 diagnosis/replay is prepared
for AFTER V66 replay; it is not queued in the already-running driver.


V63 metadata verification is complete:72 slots/53 working windows,27 unchanged source regions and26 changed, with audio/context bytes and partitions unchanged. V64 construction is terminal0 (session95750):445 passing examples/1152 attempts versus V58's349,17 contributing groups,53 valid original windows.432 case records exactly reuse unchanged geometry,171 valid examples reused;274 are newly measured at the new support geometry. Exclusions:332 no source region,304 short-window,71 three-tracker audit failures. These construction results now pass independent replay; they are not learned accuracy. All audit/renderer limits remain fixed.

V64 independent replay is COMPLETE:340 new decoded audits,1360 refused wrong-request controls (1240 with adequate common coverage),340 refused outside-sample tampers,274 labels and6096 arrays recomputed. All445 pairs are verified. Coverage comparison:127 newly constructible keys,31 lost at the changed locations,137 old-and-new valid keys with changed support; net+96 is a reference-coverage gain, not matched repair accuracy. S10/S15/S20 are newly represented.

Initial pipeline15131 is terminal1:the sampler placeholder used n*441 samples, giving n+1 controller frames for n feature frames. No fit/checkpoint or model score existed at failure. The fix uses (n-1)*441+1 samples; protection exactly matches actual-waveform policy on3 representative lengths/cases. Original failed declarations and failure receipt are retained under V65/V66; the training-recipe hash was reissued before fitting, with no data/model/loss/threshold change.

Historical completed-fit pipeline44844 (terminal at reporter failure after replay):8 fixed-budget uniform/mined fits -> independent checkpoint/sampler replay -> detailed raw report -> V66 delivered-WAV evaluation -> independent delivered replay -> domain/duration/direction report. Every nonzero phase stops later work. Check `results/transformer_usability/preservation_pipeline_status.json` and poll44844 before restarting. All8 fits are complete; initial raw gates fail for both arms and checkpoint replay is complete. Corrected output pipeline83651 is active; no activation. The prepared raw reporter passes singer-weighting/empty-subset checks;3 mining unit tests and the refusal of unverified data also pass. V66's fixed output protocol is locked before new scores.

**Authoritative current state:** [continuation record](CONTINUATION_STATE.md). V52 delivered replay completes:6 beneficial/34 preservation harms/4 changed originals for added hums, still rejected. V56 independent4-checkpoint/112-array/112-plan replay confirms2/84 fitted and4/28 held unwanted proposals. V57/V58 independently verify53 working windows and349 pairs/1152 attempts from the same33 training recordings; no new people, held evaluation access or fits. The current data/geometry limits and objective audit are in [short-phrase plan v1.8](Short_Phrase_Transformer_Plan_2026-11-02.md). The64-attempt pilot and V64 uniform scaling/replay are complete under unchanged audit limits. Preservation-mined sampling has3 passing helper tests but no trained candidate. Model qualification remains incomplete.

**Pilot/trace/timing completion:** V60 now verifies50 physical passes/64 attempts from the3 previously unscoreable training groups;224 negative controls refuse (196 assessable),56 sample tampers refuse and800 arrays replay. V61 traces all34 harms to actual outside-injection sample changes and separates guard-label gaps from existing-zero failures. V62 reproduces4 known full-path outputs in5.61–12.31s, with explicit small-sample/input/startup limitations. Uniform scaling/replay now pass, and the bounded preservation contrast is active; no new model accuracy gain or activation is claimed.

**Historical stage notes:** all earlier running/next-stage statements below are retained history, superseded by the authoritative current state above.

**Latest verified result:** [V51 transfer](HumTrans_Transfer_Result_2026-11-02.md) completes4 fits,8-checkpoint/408-array/4-trace replay. Added hums improve hum error19.365->17.845c, regress Vocadito21.201->22.652c and fail all represented-group gates. V52 delivered evaluation runs under prospectively fixed policy. V53 app-frontend probe accepts5 previously excluded training crops technically;4 meet target duration. Existing gain protection is unchanged; independent replay recreates all5 working files exactly. Only2 recovered sources have short-context eligibility. Uniform whole-cohort frontend alignment and physical-pair replay must precede a new fit. MOD-03 ACTIVE, no activation. Older fitting status below is historical.

**Latest execution:** V50 is independently verified:61/192 short HumTrans pairs,4 contributing singers,9 valid originals;64 decoded audits,128 controls,1,441 arrays and61 labels reproduced. V51 matched transfer comparison is running4 new fits against4 frozen V46 parents. It adds real-hum training data with the same Transformer/loss/budget, preserving all528 attempts and33 attempted originals. No comparison result, activation or usability claim yet. F01/M01 remain unassessed on positive repair. [Details](Local_HumTrans_Recovery_2026-11-02.md).

**Current correction:** [local HumTrans recovery](Local_HumTrans_Recovery_2026-11-02.md) verifies24 existing recordings/references and preserves12 training/12 development roles. The V48 proposed split is superseded; its acquisition commands now refuse conflicting reservations. Actual official composition IDs are separate across splits, singers are shared. Training-only short construction/replay are complete (61/192 valid); matched new fitting is underway. Thirteen focused preparation tests pass. No network blocker for existing data; no new model qualification. Earlier acquisition-stage claims below are historical.

**Preparation update:** the [HumTrans plan](HumTrans_Data_Plan_2026-11-02.md) now has a bounded ZIP64 range reader and guarded pilot acquirer. Twelve local tests pass; neither test fixtures nor code completion establish real acquisition. A fresh direct metadata retry still fails and browser acquisition times out. No real reservation/audio/model fit is added. Dataset access must recover before broader humming training can proceed; MOD-03 remains ACTIVE.

**Current update:** [V43–V47](Paired_Short_Transformer_Result_2026-11-02.md): failed local-event cue, independently replayed experimental short-audit calibration, 115 staged physical pairs, eight paired Transformer fits with independent checkpoint replay. Both raw gates FAIL. Delivered physical/virtual-pretrained arms produce 4/2 beneficial repairs, 32/36 known-zero harms and 2/4 changed originals. Neither qualifies; independent delivered replay completes:268 proposals,83 candidate audits,80 encoded audits,230 reference records and38 original controls. Next is [HumTrans metadata-first preparation](HumTrans_Data_Plan_2026-11-02.md), not another blind architecture sweep. Auditor implemented and three synthetic tests pass; actual metadata acquisition fails with connection resets, so no real split audit, reservation or audio acquisition is claimed. MOD-03 ACTIVE. The existing PDF remains a V41/V42 snapshot.

**Latest completed result:** [V41/V42](Short_Phrase_Model_Result_2026-11-02.md):12 fits complete; all raw gates FAIL. Revised raw24.147c versus frozen24.925c does not translate to usability:4 target gains,4 target harms,2/19 originals changed and remaining error25.236301c versus input25.040978c. Independent252-prediction and252-plan/24-encoded/176-reference/76-control replays pass. Candidate REJECTED; MOD-03 ACTIVE; defaults unchanged. Current-system PDF generated from the verified results; next version addresses short-event observability, singer transfer and known-zero preservation.

**Retained construction stage:** [V40/V41 short-phrase training](Short_Phrase_Training_2026-11-02.md):44 valid pairs/336 attempts from6 singers;19 valid originals;128 decoded audits,1,134 feature arrays and44 label files independently replayed. Revised genuine Transformer and opt-in policy implemented with9 passing contract checks; defaults unchanged. Four-fold comparison subsequently completed and was rejected; see latest result above. Actual-WAV V42 protocol locked before final raw results; neither qualification nor activation complete.

**Latest verification:** [V39 timing feasibility](Short_Support_Timing_Study_2026-11-02.md): real-voice forward perturbations pass8/96 baseline,17/96 guard-only,23/96 shorter ramp+guard;240 decoded audits and240 opposite-request controls independently replayed,144 exclusions reproduced. No model accuracy or production selection. V40 training-source reservation retains21 recordings/14 singers and all existing holdouts; preparation/training have not begun. Short-scale model features are required because existing step evidence starts at0.86 s.

**Current override:** MOD-03 remains ACTIVE. The owner chose 5–15-second melodic hums with short notes. V35/V36 centered-auditor engineering results are 29/200 and 11/48 verified outputs; original controls unchanged. V37 fresh construction has zero scoreable cases among 48 attempts, so fresh model accuracy remains unassessed. The next implementation is the [short-phrase plan](Short_Phrase_Transformer_Plan_2026-11-02.md), starting with experimental short-support rendering/measurement before retraining. Older held-voice next actions below are historical. The Transformer and shared demo selection remain unchanged.

| ID | Existing capability / observation | State | Evidence and limitation |
|---|---|---|---|
| BASE-01 | Technical validation, source preservation and accepted-only package | IMPLEMENTED | `audio/pipeline.py`, `audio/contracts.py`; musical correctness remains separate |
| BASE-02 | Acoustic features, PESTO/YIN and runtime boundary views | IMPLEMENTED | `audio/analysis.py`, `repair/model_runtime.py`; agreement is not intended-note truth |
| BASE-03 | ExtraTrees reviewed experimental demo | IMPLEMENTED / LIMITED | Whole-recording review/export exists; prior calibration/pilot research gates failed |
| BASE-04 | New context Transformer fit and saved weights | EXECUTED / LIMITED | [Run evidence](../results/transformer_recovery/760a6f1745c94263a5b9a51b7dafffd5/evidence.json); 43 audited training interventions / 64 attempts, 7 represented training singers |
| BASE-05 | New calibration raw correction | VERIFIED DEVELOPMENT RESULT | 5.839 cents; 10 scoreable / 16 attempts; 2 singers / 3 scoreable original recordings; no BPDR-loss superiority established |
| BASE-06 | New-Transformer permitted correction | FAILED TO SHOW BENEFIT | [Permission audit](../results/transformer_recovery/760a6f1745c94263a5b9a51b7dafffd5/permission_audit.json); scored error remains 25.833 cents; no rendering in this replay |
| BASE-07 | Scope/selector bottleneck | READ-ONLY REPLAY VERIFIED | [Baseline replay](../results/planning_review/2026-10-27_1754e056/baseline_replay.json); 7/10 cases have no scope-eligible scored frames |
| BASE-08 | Encoded-WAV audit and rollback helpers | IMPLEMENTED | Existing learned runtime; not yet a usable integration of the new context Transformer |
| BASE-09 | Individual repair review / user protections | NOT IMPLEMENTED | Whole-recording confirmation is existing functionality; per-repair control is the new feature |
| BASE-10 | Independent listener / novice study / extraction benefit | NOT ESTABLISHED | Reviewer status deferred/unconfirmed; no fabricated approvals; MIDI/CSV execution does not establish F1 gain |
| BASE-11 | Model-first improvement matrix | EXECUTED / MIXED RESULT | [60-fit report](Transformer_Improvement_2026-10-27.md); original requests fall 30.6%, correction MAE rises 7.2%; all objectives fail all-singer prediction gate |
| BASE-12 | Standalone rendered accuracy command | VERIFIED DEVELOPMENT RESULT / REPAIR FAILED | Reproduces seed 339: raw MAE 6.670 cents, 79.87% within 10 cents; 0/10 scoreable and 0/16 attempted delivered improvements; 24 targeted tests passed |

Historical full-suite and notebook-suite test counts refer to different receipts. The 27 October development work reported four new model tests plus three existing recovery tests passing. Do not report these as a newly executed full regression suite.

## 2. Selective-repair work register

Task states: **DONE, ACTIVE, NEXT, PLANNED, BLOCKED, DEFERRED**. Do not replace these with percent-complete guesses.

| Task | Deliverable | State | Depends on | Completion evidence required |
|---|---|---|---|---|
| SVR-00 | Consolidate goals, plans, decisions and evidence baseline | DONE | Owner's planning decisions | Master, this tracker, planning review, preserved brief, precedence notices and document-link/hash checks |
| MOD-01 | Model-first five-objective, four-fold, three-seed comparison | DONE | Owner's latest priority; cached audited training data | [Completed report](Transformer_Improvement_2026-10-27.md); 60 held-out fits, locked diagnostic selection, 3 full refits and separate calibration; all-singer prediction gate failed, no activation |
| MOD-02 | Repeatable checkpoint accuracy command and report | DONE | MOD-01 and lineage/metric checks | [Accuracy workflow](Transformer_Accuracy_Workflow.md); overlap refusal and standalone rendered repeat checked; [verification receipt](../results/transformer_improvement/d6e8b933b23546c78a53aa2292c03c1a/technical_execution/final_verification.json) |
| MOD-03 | Usable scoped Transformer and guarded BPDR adapter | ACTIVE | Owner's 28 October priority | [Execution record](Transformer_Usability_2026-10-28.md); broader sources, grouped fitting, locked policy, development/final actual WAVs, unchanged originals, loader and live app receipt; no research-completion claim |
| SVR-01 | Reusable complete repair-path trace, legacy policy | DONE | SVR-00; implemented within MOD-02 | Every attempt retains exclusion or raw/scope/selection/permission/render disposition; seed 339 includes actual output audits and rollback. This does not implement the new regionizer |
| SVR-02 | Reserve fresh sources and freeze development protocol | DONE | SVR-01 | 42 bounded new source files; training IDs7, four final IDs8/9 reserved before acoustic analysis; prospective run/partition declarations in MOD-03. [Final result and protocol receipts](Transformer_Usability_Result_2026-10-28.md); final cohort now inspected |
| SVR-03 | Multiple-candidate regionizer and versioned eligibility | PLANNED | SVR-01, SVR-02 | Frozen-checkpoint comparison; hysteresis/gap grid; hard preservation controls; no target/support input at runtime; no duration-padding trick |
| SVR-04 | Focused training-side preservation/continuity/limit experiments | PLANNED | SVR-02, SVR-03 | Five declared conditions, seeds 339/340/341, four grouped folds, adjacent-target validity, retained failures and deterministic selection record |
| SVR-05 | Actual-WAV repair and conditional renderer benchmark | ACTIVE | SVR-03; candidate from SVR-04 when available | [Frozen curve/renderer comparison](Transformer_Curve_Study_2026-10-28.md) executed; alternate renderer rejected on training-side WAVs. Full model usability and independent confirmation remain unmet |
| SVR-06 | Guarded context-Transformer selective runtime | PLANNED | SVR-03, SVR-05 | Saved-config loader, source/model/policy hashes, verified receipts, controlled failures and no legacy-loader/selection bypass |
| SVR-07 | Protected-region and individual repair review UI | PLANNED | SVR-06 and at least one verified repair | Local waveform drag selection + time fields; whole-candidate protection; A/B playback; Accept/Keep; reasons and stale-state invalidation |
| SVR-08 | Compose selected edits from original and export | PLANNED | SVR-07 | Multiple accepted repairs, combined 25% budget, exact protected/outside samples, decoded final-WAV audit, renewed confirmation and compatible accepted-only package |
| SVR-09 | Locked fresh-singer evaluation and listener evidence | PLANNED | SVR-02, SVR-05, SVR-08 | Actual before/after WAV scores, per-singer/direction/attempt coverage, all failures, original controls, latency and actual review records or explicit missing-review status |
| SVR-10 | Scoped eligibility, presentation package and rehearsal | PLANNED | SVR-09 | Gate receipt bound to actual bytes; three demonstration routes; reproducible result table; no simulated or overstated evidence |

The per-region UI may be prototyped against explicitly labelled test fixtures after backend contracts are fixed. Such fixtures cannot complete SVR-06, SVR-09 or a model-success claim. Do not begin a broad frontend redesign while useful repairs are still unavailable.

## 3. Research obligations retained from the proposal

| Task | Remaining research work | State | Completion evidence |
|---|---|---|---|
| RES-01 | Proposal-aligned matched BPDR / augmentation-only / frame-only comparison | PLANNED | Common frozen inputs/budgets/selection, MAE response terms, original + conservative-rule references, explicit feature-version deviation and held-out paired result |
| RES-02 | Split/merge/jitter, +/-10/20/30-cent and no-change evaluation | PLANNED | Boundary/shift/direction/control breakdowns; natural analyser errors and matched coverage; no constant-zero shortcut |
| RES-03 | Independent suitability anchors and listening | BLOCKED | Two actual qualified independent decisions and records; current reviewer availability is unconfirmed/deferred, not approval |
| RES-04 | Consented novice guide-melody/free-humming study | BLOCKED | Ethics/consent prerequisites and prospective sample-size/precision rationale; original eight-person pilot is an estimate only |
| RES-05 | Fixed-extractor before/after note benefit | PLANNED | Independent references, frozen extractor/decodes and onset+pitch / offset-aware precision, recall, F1; no teammate-feature implementation |
| RES-06 | Paired uncertainty, seed variation, matched acceptance and cost | PLANNED | Singer/source independent units, paired intervals with small-study caveats, seed variation separately, median/p95 latency and memory/render failures |
| RES-07 | Decide whether the main hypothesis is supported | PLANNED | Full BPDR vs augmentation-only is the primary comparison; report null/negative outcomes and the boundary-free alternative honestly |

Research work can be prepared alongside engineering, but neither completed UI nor the scoped public-data evaluation closes these obligations automatically. Review unavailability does not stop source-preserving engineering.

## 4. Required gates and how to count outcomes

| Gate | Pass condition | Current state |
|---|---|---|
| G1: permitted repair | At least one supported new-Transformer candidate passes unchanged numerical permission limits | Numerical permission observed in MOD-02; one target-overlap attempt rolls back and the technically verified detuned edit does not improve scored frames |
| G2: delivered repair | Encoded/reloaded output improves the reference-controlled shifted region and passes preservation checks | Observed in MOD-03: 5/40 scoreable, 5/48 attempted final interventions; insufficient usable coverage |
| G3: original preservation | Existing per-singer raw/permitted original and edit-rate criteria satisfied; exact zero-change/protected samples where promised | Latest MOD-03 preserves all 12 final originals; prior MOD-02 failures retained separately |
| G4: useful scoped coverage | Positive delivered gain, both directions represented by actual verified improvements, >=25% all-attempt verified coverage per represented evaluation singer | FAILED MOD-03: every final singer falls below coverage/gain criteria; originals remain preserved |
| G5: selective controls | Protected overlap veto, per-edit decisions, fresh confirmation and final combination verified | NOT IMPLEMENTED |
| G6: reviewed presentation evidence | Honest demonstration routes, actual listener status, byte-bound artifacts and repeatable rehearsal | PLANNED; no missing listener result may be recorded as a pass |
| G7: methodological advantage | Matched held-out BPDR objective benefit with preservation at matched coverage | UNESTABLISHED; historical failed gates remain failed |

For every comparison, keep attempted sources/interventions, scoreable interventions, eligible candidates, permitted repairs, rendered outputs, verified delivered improvements and withheld/failed outputs separate. Report overlap between rejection reasons. A scoreable but scope-excluded case stays a scored unchanged/withheld result; it does not become an intervention-audit exclusion.

Useful duration coverage is the **union** of scored erroneous duration actually improved by verified output divided by the declared eligible erroneous duration. Also report verified-improved cases divided by all attempted interventions. Do not double-count overlapping edits or hide low construction yield behind eligible-only percentages.

## 5. How to update this tracker

Before starting a task, read the master, check the current code/artifact hashes, and record its declaration/configuration. Mark a task DONE only after its completion evidence exists. When blocked, name the missing prerequisite rather than claiming completion or removing the row. Never change old outcomes to make a new experiment appear successful.

Each work entry must include:

| Field | Required content |
|---|---|
| Date / task | Local date and stable task ID |
| Change | Specific behavior or experiment completed |
| Evidence | Relative run/receipt/report or source change |
| Verification | Commands/checks actually run and their outcomes |
| Counts / limits | Independent source/singer counts, exclusions, failures and unresolved issues |
| Decision / next | Adopted or not adopted, reason, and next task ID |

### Work log

| Date / task | Work completed | Evidence / verification | Remaining / next |
|---|---|---|---|
| 2026-10-27 / BASE-04–07 | New context-model experiment and read-only failure diagnosis already completed before this plan consolidation | Existing checkpoint/evidence/permission files; planning baseline replay reproduces 5.83948 cents and the 7/10 zero-scope finding | No usable scored repair from this candidate yet |
| 2026-10-27 / SVR-00 | Original research objective checked; current master, tracker and review created; supplied brief preserved; historical entry points redirected | [Planning review](Planning_Review_2026-10-27.md), baseline/backup manifest and documentation verification receipt | **SVR-01 is next**; no new feature, fit, renderer or model activation in this documentation task |
| 2026-10-27 / MOD-01–02, SVR-01 | Completed five-objective matrix, 3 final refits, separate calibration and standalone rendered repeat; added lineage checks, metrics/reports and legacy-policy traces | [Results](Transformer_Improvement_2026-10-27.md), [receipt](../results/transformer_improvement/d6e8b933b23546c78a53aa2292c03c1a/technical_execution/final_verification.json); 24 targeted tests passed; original data/models/notebook hashes preserved; technical resume correction retained | No adoption: stronger preservation trades off correction accuracy; 0/16 delivered gains. SVR-02 next; SVR-03 frozen-model controller comparison separate |

| 2026-10-28 / MOD-03, SVR-02 | Expanded sources; 47 saved neural fits; input-only step-feature Transformer; versioned policy; guarded adapter; locked independent final and standalone repeat | [Result](Transformer_Usability_Result_2026-10-28.md); 57 targeted tests passed; 5/48 final gains, 12/12 originals unchanged; activation refused | MOD-03 ACTIVE: all four final singers fail coverage/gain. No activation or later feature work; new independent cohort needed for subsequent changes |

**Latest completed experiment:** [Frozen curve and renderer studies](Transformer_Curve_Study_2026-10-28.md). The training-side rule selects median11/Praat; known-development repairs improve from6/48 to8/48 and all12 originals stay unchanged, but female5 gain remains23.394%, below25%. The phase-vocoder alternative loses (16/200 versus22/200 constant-Praat); no development replay for it. Four curve/controller tests and two renderer tests pass. No new neural fits or final-data access. MOD-03 remains ACTIVE; next is a matched-budget adjacent-frame-supervision model experiment, not further renderer tuning.

**Latest outcome, 29 October:** [Response weighting](Transformer_Response_Study_2026-10-29.md) completed8 fits and lost actual-WAV comparison (17/200 versus23/200). [Localized context](../results/transformer_usability/2026-10-29_v13_localized/REPORT.md) completed4 new fits using the verified baseline: raw MAE improves11.901 ->10.665 cents, but original requests rise and the all-singer prediction gate fails. Both verify23/200 repairs and preserve50/50 originals; localized delivered error is worse. No full refits, activation or new evaluation-audio access. Next test broad+local context together with targeted difficult-original sampling; retain all gates and the fresh source reservations.

**Completed broad/local comparison:** [Study](Transformer_Multiscale_Study_2026-10-29.md), eight new CV fits. Uniform/difficult-original conditions verify21/200 and23/200 gains versus23/200 baseline, preserve50/50 originals each, fail female1 raw-original limits, and lose delivered error. Reference retained; actual refit command refuses promotion. Integrity check verifies546 frozen hashes,8 checkpoint lineages,44 unique encoded verified WAVs and6 protected files. No activation or new confirmation-audio access.

**Latest completed execution:** [Warm-started broad/local candidate](Transformer_Warm_Start_2026-10-29.md), four new CV fits starting from corresponding partition parents. Initial predictions match exactly; lineage checked. All-singer prediction gate passes; raw MAE11.464 versus11.901 cents baseline. Actual-WAV improvements fall23/200 ->21/200; delivered error rises20.342 ->21.648 cents. All50 originals remain unchanged; candidate rejected. Thirteen focused tests pass. Independent replay verifies552 frozen hashes,4 checkpoint lineages,21 encoded WAVs and6 protected files. Additional training budget and combined interventions are explicit. MOD-03 remains ACTIVE; next diagnose model-supported candidate withholding and output direction failures before another training change.

**Completed support study:** [Result](Transformer_Support_Study_2026-10-29.md):8 fits, matched traces, both prediction gates pass, both21/200 repairs and50/50 originals preserved. Reference23/200 retained; continuity68/108 unchanged. Fifteen focused tests pass;552 hashes,8 fold lineages and42 encoded WAVs independently verified. No full refits or activation.

**Latest completed execution:** [Broader actual-audio pool and model study](Transformer_Audio_Pool_2026-10-29.md). All373 unique variants re-audit, including108 base cases without label loss;265 additional examples share the same50 originals/10 singers. Eight matched new fits complete, with exact V16 control reproduction. Broader data reduces raw error11.901 ->11.322 cents and increases model-only support68 ->72/108, but fails female2 raw-original3.207>3 cents and yields18/200 verified repairs versus23/200. All50 originals unchanged; no adoption. Eighteen focused tests pass;1,419 hashes,8 lineages and39 encoded WAVs verified. Dataset and failures retained. Next fixed, dispersion-aware ensemble comparison; no confirmation access or activation.

## 6. Handoff instruction for the next implementation session

**Latest scoped pass,1 November:** [PCM24 diagnosis](PCM24_Stability_Diagnosis_2026-11-01.md) independently reconstructs11 candidates/33 measurements and identifies1e-12c coarse vetoes without changing outcomes. [V35 centered ALWA](Centered_Auditor_Study_2026-11-01.md) algebraically rearranges the same weighted estimator,with no epsilon/window/model/threshold change;actual delivery29/200 and19.585478c beats V33's27/200 and19.824705c,50 originals unchanged. Independent648-array/216-decision/30-candidate/29-WAV replay passes. [V36 known development](Centered_Development_Qualification_2026-11-01.md) verifies11/48,15.954963c versus9/48,17.322854c,12 originals unchanged. Female5/male6 pass35.82%/35.75%;female6/male5 unassessed. Independent48-view/144-array/48-decision/11-candidate/11-WAV/24-score/gate replay passes.44 tests pass,6 protected artifacts unchanged,default339 lineage reused,no new fits/activation. Sessions61560/16115(diagnosis),88442/74233(training-side),67576/20776(development),34750(tests) are terminal;do not restart. Pipeline/cohort locked for fresh four-source2F/2M confirmation in V37;JSUT-song's metadata-documented female fills the prior slot prospectively,first3 lexicographic WAVs to select before decode. Exact file selection/private acquisition and all fresh outcomes remain pending;no fresh acoustics accessed. Human reviews/consented intended-reference pilot remain unexecuted. MOD-03 remains ACTIVE.

**Latest verified improvement,1 November:** [V33 precision-auditor pipeline](Precision_Auditor_WAV_Study_2026-11-01.md) completes actual-WAV evaluation and independent verification. Verified repairs23/200->27/200 and singer-macro remaining error20.342142->19.824705c;all50 originals unchanged. Same frozen reference Transformer, no fits, exact648 prediction/curve arrays and216 permission decisions. Only coarse-auditor ALWA pitch/frequency arithmetic changes to float64; every sign/coverage/error limit, source, mask, support and renderer stays fixed. Independent Torch measurement rechecks30 candidates and27 encoded WAVs;50 original raw predictions/plans match,6 protected artifacts unchanged,41 tests pass. Training-side candidate selected; no activation, development or confirmation access. Full practical singer gate still fails and female7's4/20 valid-reference coverage cannot establish25%; preserve the limitation. All sessions96590(failed format comparison),86945/69231(reconstruction),9968/20087(precision),30368/53676(WAV evaluation/verification),67531(tests) are terminal; do not restart. Bound JSON-representation amendment preserves the original failure. Next verify full-fit reference lineage/default339 and prospectively lock known-development replay before untouched confirmation. Actual reviewers, consent, pilot audio and independent intention/preservation ratings remain absent. MOD-03 remains ACTIVE.

**Latest completed execution,1 November:** [V32 nearby unchanged-region sampling](Transformer_Nearby_Zero_2026-11-01.md) completes four CV fits, actual median11/Praat delivery and independent replay. All four legacy digests match V12; full172800-row actual-stream reconstruction verifies every changed query and unchanged positive/source/feature/original-anchor schedule. Raw gate passes, but20/200 repairs and21.108c delivered remaining error lose to reference23/200 and20.342c. All50 originals unchanged;216 prediction views,20 encoded WAVs and6 protected files verify. All35 focused tests pass. No selection, activation, full refits or development/confirmation access. CV session93563, delivery8591 and verifier6935 are terminal; do not restart. Four reference-only successes fail a single coarse-direction frame with fine direction passing; one new success resolves the same veto. Next reconstruct dual-tracker disagreements on the declared training-side candidate set while preserving every auditor/threshold, not another generic training sweep. Review/collection plan and templates are prepared; real approvals, people, ratings and recordings remain absent. MOD-03 remains ACTIVE; usable accuracy is unproved.

**Latest execution,1 November:** [V31 matched absolute response](Transformer_Absolute_Response_2026-11-01.md) completes all8 fits. Four MSE controls exactly reproduce V12 tensors, validation arrays and source/sample traces. Four MAE-response fits pass raw gates but lose actual delivery18/200 and22.062c versus reference23/200 and20.342c;50 originals remain unchanged. Independent replay checks216 prediction views,18 encoded WAVs and all8 fit lineages, with6 protected artifacts unchanged. All32 focused tests pass. No candidate selected; no live fit/evaluation process remains. The owner requested a [review and collection plan](Review_and_Collection_Plan_2026-11-01.md); it and3 blank JSON templates are prepared, with no actual participant, consent or rating records. A sampling audit reconstructs172800 draw rows and finds trusted nearby-zero queries occupy2.22–2.82% of draws; this motivates a prospectively declared localization experiment, not invented transition labels or a claimed remedy. FCPE, locality and V30 are terminal and not adopted. No known-development/reserved-audio access, UI, export, notebook or sibling changes. MOD-03 remains ACTIVE; usable accuracy is unproved.

**Current handoff,1 November:** owner resumed the paused goal. FCPE sidecar setup,24 synthetic controls and real200/108/92 measurement complete; [diagnostic](Acoustic_Measurement_2026-11-01.md) rejects replacement:91.064c relative MAE versus0.860c YIN, with all158 cached recordings independently replayed exactly. Current YIN/PESTO auditors stay unchanged. The privileged201-frame context diagnostic permits48 versus45,3 additions/0 losses; all108 cases and97 original-crop queries replay, not a deployable algorithm. V30 fixed binary activity decoding passes raw gates (11.873c error,0.434c original magnitude) on frozen weights; delivery is live and must be polled before any restart, followed by `verify_activity_decoding.py`. No development/fresh confirmation until qualified selection and lineage lock.28 existing focused tests plus2 new decoder tests pass. Keep all protected artifacts,200/108/92 denominators and sibling boundaries; MOD-03 is incomplete.

**Authoritative handoff,30 October completed:** read [repair feasibility diagnosis](Repair_Feasibility_Diagnosis_2026-10-30.md). V28 original CV session21580 and every recovery/evaluation/audit job are terminal; do not restart. All4 tree fits are saved, original roundoff failure retained, separate serial-reload continuation completed, actual delivery20/200 and21.217c loses to reference23/200 and20.342c, and raw gate fails male7 original4.428c. Independent216-view/20-WAV/4-fold verification passes. Combined V27/V28 alternatives verify648 views/57 WAVs/12 lineages; all50 originals unchanged per family,6 protected files match,25 tests pass. No model selected or activated. Privileged known-label diagnostics45 permitted/35 verified/10 rollbacks and supplementary outside-injection preservation are independently verified, explicitly not deployable accuracy. Next prospectively check input-measurement precision/localization using approved training sources (PENN/FCPE preflight with isolated dependencies and unchanged auditors/limits), not another generic architecture sweep. Keep200/108/92 denominators, all history, reserved confirmation audio, protected notebook/export, shared demo and sibling components intact. MOD-03 remains ACTIVE; no usability or musical-review pass.

**Newest result:** V27 completes8 fits and audio delivery. GRU18/200,21.626c,raw-gate fail; TCN19/200,21.281c,raw-gate pass; reference23/200,20.342c retained. Independent432-view/37-WAV replay and8 actual sampler/source/checkpoint lineages pass,50 originals unchanged per family,6 protected files unchanged.25 tests pass. V28 original tree CV process remains live: poll it, do not restart. Its frozen reload check is bitwise despite threaded float64 reduction; a prospective recovery tool and exact requirements are documented in [comparison](Model_Family_Comparison_2026-10-30.md). Only if the original process is terminal after checkpoint saving, use its saved models and explicit verification amendment, then delivery and independent tree verification. No usability or research pass yet.

**Current handoff, superseding older next-step paragraphs below:** follow the owner's expanded30 October goal and [model-family comparison](Model_Family_Comparison_2026-10-30.md). V26 completes4 dense residual head fits; raw gate passes but delivery21/200 and21.394c loses to reference23/200 and20.342c. Independent216-view/21-WAV/frozen-parent replay passes;50 originals unchanged and6 protected files match. All25 focused tests pass. V27 eight GRU/TCN fits and V28 four fresh context-ExtraTrees fits are running; poll their existing process handles before any restart, then finish actual-WAV evaluation and `verify_model_family_comparison.py sequence/tree`. Require matching V12 sampled-source/singer traces and preserve all200/108/92 counts. No family result yet, no selection or activation change. Primary-source research, bottleneck counts and component-safe recording-coach/multiple-take/readiness suggestions are in [model choice and input usability](Model_Choice_and_Input_Usability_2026-10-30.md). Preserve all previous failures, untouched CSD/male10/male11 audio, user notebook, exports, `pp1_demo` and sibling ownership. MOD-03 remains ACTIVE.

**Latest,30 October:** [Candidate-head/readout results](Transformer_Candidate_Head_2026-10-30.md):8 new head fits,0 new backbone fits. V22/V23 fail response gates at20/200 and21/200 verified repairs. Frozen V24 centre readout passes raw gates and reaches24/200 but loses delivered error20.706 versus20.342. All50 originals unchanged. Twenty tests pass;8 parent states/fold counts/fallbacks and65 encoded WAVs verified,6 protected files unchanged. None selected; no refits or new evaluation-audio access. Next fixed within-region distribution-readout comparison with unchanged criteria.

**Latest completed work:** [Ensemble](Transformer_Ensemble_2026-10-29.md) rejected at2/200 repairs despite prediction-gate pass; [R3 renderer](Transformer_R3_Renderer_2026-10-29.md) rejected at22/200 versus23/200 Praat. Both preserve50/50 originals. Seventeen combined tests pass;12 ensemble lineages,24 total encoded verified outputs and6 protected files checked. No new neural fits or evaluation audio access. [Candidate preflight](Transformer_Candidate_Preflight_2026-10-29.md):177 positive/8 negative/14 unassessed regions;39 baseline cases lack candidate overlap. Dense fallback required.96 unchanged feature controls add zero candidates; retain result and all denominators. Next small region head plus explicit original anchors; no activation.

Read the [V25 result](Transformer_Distribution_Readout_2026-10-30.md) and [accuracy workflow](Transformer_Accuracy_Workflow.md). Continue **MOD-03**; usability is incomplete. V25 median/upper-quartile readouts pass raw gates but verify only18/200 and19/200 repairs, with remaining error21.444c/21.992c versus reference20.342c. Neither selected. All50 originals unchanged per arm;20 tests and independent37-WAV/prediction/fallback checks pass,6 protected files unchanged. Next prospectively declare a function-preserving dense amplitude/activity residual adapter with frozen backbone, input-only cues and exact zero-adapter control. Retain temporal shape, all existing limits, held-out source separation and200/108 denominators. Selection remains raw all-singer gates, all50 originals identity, lower singer-macro delivered error; count alone cannot substitute. Keep39/108 candidate-overlap and8-real-negative limitations explicit. Require a qualified locked model/decoder before development/fresh confirmation. Reserved CSD/male10/male11 audio remains unopened. Preserve notebook/export/model/data history, shared `pp1_demo` and sibling code; no selective UI or activation yet.

| 2026-11-03 / V55 | Rebuilt short-phrase pairs on frontend-matched V54 inputs: 528 attempts, 188 valid (176 reused, 12 new), 32/33 originals valid; no fits | [Result](Canonical_Short_Pairs_2026-11-03.md); builder ran to completion; independent replay NOT yet run | No fit justified by +12 pairs; next: V55 replay verifier, then address region availability/objective comparison. MOD-03 ACTIVE |


### V65/V66 measurement addition,4 November

V66 actual-WAV evaluation is prospectively declared before new training scores:
`2026-11-04_v66_preservation_delivery/declaration.json`. Both V65 arms use identical
1152 attempted/445 scoreable cases and53 original controls, the unchanged input-only
controller, renderer, three-tracker audit and strict original/known-zero requirements.
Its evaluator, independent replay and domain/duration/direction reporter are prepared
but not executed. Historical V52 totals are not a matched model-effect baseline on
this changed support geometry. The added raw metric reporter also compares the
zero-command baseline,10/25c fractions, signed response, original extremes and
known-zero requests; descriptive paired singer bootstrap uses2000 draws/seed339,
not overlapping frames as independent people. No confidence calibration or natural
intention accuracy is claimed. Execute these after V65 checkpoint replay succeeds.


### V64 reference ceiling and queued preservation diagnosis

V64 reference sufficiency now explicitly includes duration-valid denominators.
Only HumTrans:F01,S1,S21 remain below the25% historical coverage ceiling (all have
zero scoreable pairs;16/112/32 duration-valid attempts respectively), versus seven
groups under V58. This is construction coverage, not learned repair coverage.
The retained V58 ceiling still describes V58; do not present it as the V64 ceiling.

The prepared V67 original-preservation diagnosis compares both V65 arms on53
original windows across four models each:159 fitted-source/model combinations and
53 held controls per arm. It records local activation tails, input-only permitted
proposals and actual held original changes, and has a separate424-array/plan replay
implementation. Run `diagnose_preservation_sampling.py`, then
`verify_preservation_diagnosis.py` after V66 replay. Neither has executed yet. No
fitted proposal is labelled a delivered harm. This is inference only, not extra fits.


### V65 initial completed fits, replay pending

V65 completes all8 fits (4 folds x uniform/mined,seed339,1200 updates each).
Initial raw summaries are awaiting independent replay. Uniform/mined positive
singer-macro errors are19.701240/19.958749c overall,16.923916/18.708845c on HumTrans,
and20.858458/20.479542c on Vocadito. Original-control means are1.051955/1.243259c.
Both fail the represented-group raw gate. S29 exceeds the3c original mean limit
(3.510123/4.338468c); S20/S22 have negative signed response in both arms. The newly
constructible S10/S15/S20 groups do not meet the response requirement either.

These matched results do not support adoption of the mined sampler; no extra
seed/threshold/epoch sweep is authorized by this contrast. V51's different pool is
not a matched model-effect baseline. Pipeline44844 is now independently replaying
checkpoints/sampling before the fixed V66 actual-WAV comparison; delivered benefit,
known-zero harm and unchanged-original outcomes are still pending. No activation.


### V66 initial WAV outcome, independent replay active

V66 initial delivered evaluation completes both arms on the same1152 attempted /
445 scoreable cases and53 original controls. Independent replay is running in83651;
these figures are not yet the completed replay result. Uniform/mined:24/21 beneficial
repairs,148/128 reference harms,143/127 known-zero harms,11/9 changed originals.
Remaining singer-macro target error24.028204/23.618242c from input24.896935c.
Uniform has275 identity/183 audited candidate/40 rollback outcomes; mined293/158/47.
Humming:15/9 beneficial and33/21 harms on100 scoreable cases; Vocadito9/12 beneficial
and115/107 harms on345. These are controlled reference failures, not subjective
claims that every change is audible. No encoded rollback occurs in these totals.

Mined sampling has a delivered-output tradeoff despite worse raw mean error:
fewer harms/changed originals and lower remaining target error, but fewer beneficial
outputs. Neither arm meets preservation/coverage/raw requirements. It must not be
adopted or called usable. Raw-only averages did not determine that comparison.

Prepared `tools/trace_preservation_delivery.py` requires the V66 integrity receipt
and writes `V66/pipeline_trace.json` after replay. It checks actual output waveform
hashes and separates target direction, misplaced support, unlabelled edge requests,
already-zero-labelled false requests and actual samples changed outside injection.
Run it after V66 replay, then the prepared V67 diagnosis/replay; no new fit yet.


### V66/V67 completed verification and diagnosis

V66 is now independently VERIFIED:996 plans,428 candidate audits,341 encoded
outputs,890 positive/zero reference scores and106 original controls replay. Both
arms remain rejected:uniform/mined24/21 beneficial,148/128 reference harms,
143/127 known-zero harms and11/9 changed originals. Remaining target error
24.028204/23.618242c from24.896935c. Mined has fewer harms but fewer useful gains;
it is not a qualified improvement. Pipeline83651 is terminal0.

The V66 actual-sample trace confirms all143/127 known-zero harms change samples
outside injected support.84/77 include unlabelled edge requests (292/282 frames),
but125/116 also request already-zero-labelled frames (2060/2064 frames total).
58/46 harmful plans have no overlap with the injection. Even14/11 numerically
beneficial outputs change samples in the excluded near-edge guard; beneficial
means the registered reference gate, not perfect ground-truth support localization.

V67 independently replays8 checkpoints/424 arrays/plans and matches106 held
original events to V66. Uniform/mined training-assigned combinations:24/18 unwanted
permitted proposals out of159; held controls12/13 out of53, with11/9 actual changes.
Training-assigned proposals were not rendered and are not measured harms.156/159
training combinations and52/53 held windows have reliable pitch evidence; one
zero-evidence window is kept with null statistics. Fitted failures mean this is not
only an unfamiliar-singer problem. Pipeline96363 is terminal0; no jobs remain live.

## V68 active loss-only contrast

[Predeclared methods and research limits](Short_Phrase_Transformer_Plan_2026-11-02.md)
and [full result record](HumTrans_Transfer_Result_2026-11-02.md) retain the evidence.
V67 fitted-source failures and V66 already-labelled zero requests motivate testing
upper-third preservation loss before adding edge labels. Four frozen V65 uniform
parents versus4 new fits; only C-zero/generalA aggregation changes to largest4/12.
Exact traces/data/labels/model/seed339/1200 updates/48 forward rows/runtime gates
stay fixed. No k/seed/epoch/threshold sweep or zero-command-collapse adoption.

V68 implementation now passes4 loss/gradient tests and an exact source comparison
against V65's physical fitting function:only the two negative aggregation terms
change. V68 declaration and V69 unchanged-WAV policy are locked before new scores.
V69 inherits only the verified V66 uniform arm after exact data/policy/prediction
identity checks; it performs no new baseline render. The new variant is separately
rendered and independently replayed. No changed target/sampler or external code.

Current live pipeline80654 runs V68 four fits -> raw checkpoint replay/report ->
V69 new-arm WAV evaluation/replay/domain report. At leastfold1 fit/held predictions
are complete; fold2 is fitting at last inspection. Poll80654 and the durable status
before restarting. Four fits maximum, no k/seed/epoch/threshold sweep or promotion.


### V70 query-coverage contrast predeclared

V68/V69 raw and WAV replay complete:tail loss32 gains/109 reference harms versus
24/148 uniform, but11 originals still change. Next is a four-fit original-query
coverage contrast with data/labels/tail loss fixed;positive/C-zero draws stay fixed,
original group/record frequencies explicitly change. See the [exact predeclaration](Short_Phrase_Transformer_Plan_2026-11-02.md). No activation.
**6 November short-hum validation.** The eight-case renderer control reproduced
small pitch changes alongside waveform differences from zero-shift forced
resynthesis. V71 decoded-WAV replay passed and confirmed 22/445 beneficial,
89/445 harmful and 8/53 changed originals. The spectral-context comparison
is still running. No short-hum Transformer candidate qualifies.
[V71 result](HumTrans_Transfer_Result_2026-11-02.md#v71-independently-replayed-on-6-november)

The frozen V68 checkpoint replay matched the archived clean and shifted
predictions for the eight renderer controls. Zero-shift resynthesis produced
a 16.94-cent average request change in one case and under 1 cent in the
other seven. [V68 response diagnostic](HumTrans_Transfer_Result_2026-11-02.md#frozen-v68-response-to-the-control)

The S7 follow-up isolated the outlier to the pitch, voicing and pitch-delta
input block in the tested feature substitutions. The diagnostic reproduced
without a fit or policy change. [Feature trace](HumTrans_Transfer_Result_2026-11-02.md#s7-feature-substitution)

The archived short-phrase boundary perturbation affects every scored
positive query context, with the acoustic columns held fixed. This
passes the data preflight for the full BPDR, augmentation-only and
frame-only objective comparison. All three arms pass one frozen-trace
update and held inference. The full 12-fit run and checkpoint replay are
queued sequentially in session 68661.

The completed spectral replay matched all 1,494 arrays across the two
fitted arms and frozen baseline. The spectral candidate remains rejected.
[Verified spectral result](HumTrans_Transfer_Result_2026-11-02.md#verified-matched-spectral-context-result)

The longer-hum extension probe reproduced eight scoreable constructions
out of 14 attempts, with all 32 negative controls rejected. This confirms
physical coverage on existing sources, with model evaluation still pending.
[Longer-hum probe](HumTrans_Transfer_Result_2026-11-02.md#replayed-longer-hum-physical-probe)

Frozen singer-held-out V68 inference on those eight cases reproduced
13.261 cents of raw error versus 20 for zero. Large requests on all four
originals keep preservation unresolved. The follow-up delivery replay
returned six identities and two renderer rollbacks. All altered inputs
and originals stayed exact, so the model produced no useful longer-hum
repair in this sample.
[Longer-hum transfer](HumTrans_Transfer_Result_2026-11-02.md#frozen-v68-transfer-beyond-10-seconds)

The two renderer rollbacks come from a nonzero request at the exact
waveform endpoint. Clearing that frame passes the diagnostic audits,
and decoded scoring confirms known-zero harm in both cases. One edit
starts after the detune, and the other repairs the target while editing
past it. This isolates the endpoint mismatch and a localization error.
[Endpoint trace](HumTrans_Transfer_Result_2026-11-02.md#f05-endpoint-trace)

The experimental sample-backed planner now withholds this impossible
request before rendering and preserves the input waveform. The endpoint,
controller and short-policy suites passed all 16 tests. This guard is
separate from the frozen comparison and does not qualify any repair model.
