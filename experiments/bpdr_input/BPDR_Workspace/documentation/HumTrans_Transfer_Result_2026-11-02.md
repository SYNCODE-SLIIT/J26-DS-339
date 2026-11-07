# Real-hum transfer comparison: prediction checked, usable accuracy not established

Updated4 November2026; filename/run labels retain the2 November reservation date.
V51's four added-hum fits, V52's delivered-audio evaluation and their independent
replays are complete. MOD-03 remains
ACTIVE; no candidate is qualified or activated.

## Matched comparison and evidence

Four frozen V46 paired-physical checkpoints are compared with four new fits using
the same genuine 21,666-parameter Transformer, seed339,1,200 updates, optimizer
and paired-response/preservation loss. The new arm adds61 physical HumTrans pairs
to115 Vocadito pairs. Training/source holdouts remain disjoint; only the training
pool and its singer-balanced sample draws change. No actual development or
reserved-unseen recording is reopened. Prior historical HumTrans engineering use
is retained, so these are not fresh confirmation results.

The cohort retains528 attempted interventions,176 scoreable pairs and33 attempted
original recordings,28 of which are valid controls. Eight attempted groups have no
positive pair and remain unassessed: HumTrans F01/M01 and Vocadito S1/S10/S13/S15/
S20/S21. Four humming singers contribute the61 positive cases; the number of
augmented variants is not the number of independent people.

Independent replay reloads all eight checkpoints, reproduces408 prediction arrays
and four new physical sampler traces, checks source/singer disjointness and
recomputes every summary. Evidence is under
`results/transformer_usability/2026-11-02_v51_humtrans_transfer/`, including
`integrity_receipt.json`, saved checkpoints, declarations and held-out predictions.

## Raw prediction outcome

| Domain | Vocadito-only parents: correction error | Added real hums: correction error | Original-prediction error, parent -> added hums | Represented-group gate |
|---|---:|---:|---:|---|
| HumTrans, four scoreable singers | 19.365c | 17.845c | 1.156c ->0.648c | Both FAIL |
| Vocadito, eight scoreable singers | 21.201c | 22.652c | 0.673c ->0.809c | Both FAIL |
| Combined, twelve scoreable singers | 20.589c | 21.050c | 0.815c ->0.762c | Both FAIL |

The correction columns are singer-macro errors on held-out supervised positive
frames. Original-prediction errors describe model requests, not changed WAVs.
They cannot be called repair accuracy or evidence of preserved musical intention.

The improvement is uneven. F02 improves13.265 ->6.388c with signed response near1.
F03 improves15.896 ->15.355c. F04 changes27.621 ->24.652c but has almost zero useful
response. F05 regresses20.680 ->24.984c and also has almost zero response. A better
average over the four voices cannot override those group failures or absent F01/M01
positive evidence. The combined result worsens because Vocadito transfer regresses.

## Actual-output evaluation

V52's policy was declared before final raw scores. It uses the same expanded
cohort for both arms, the V47 input-only proposal policy,20ms ramp/70ms guard and
experimental three-tracker audit. Reference audio/masks enter only after output
for evaluation. A beneficial repair must improve the injected plateau without
worsening known-zero frames by more than1e-6c; any original-control change prevents
qualification. Numerical preservation failure is distinct from audible damage.

The evaluator and independent replay complete:408 proposals,134 candidate audits,
116 encoded audits,352 reference records and56 original-control checks. The domain
report reproduces the V47 parent result on retained Vocadito data.

| Domain | Parent beneficial / harms | Added-hum beneficial / harms | Changed originals, parent / added hums | Remaining error, parent -> added hums |
|---|---:|---:|---:|---:|
| Humming,61 scoreable/192 attempts | 0 /32 | 3 /12 | 2 /2 of9 | 23.568 ->24.107c |
| Vocadito,115 scoreable/336 attempts | 4 /32 | 3 /22 | 2 /2 of19 | 22.740 ->24.227c |
| Combined,176 scoreable/528 attempts | 4 /64 | 6 /34 | 4 /4 of28 | 23.016 ->24.187c |

Combined input error is24.841c. More beneficial outputs and fewer preservation
harms do not offset changed originals and insufficient coverage. Both arms remain
rejected. All reported harms correspond to actual sample changes outside the
known injected support in V61's waveform trace; this is not a claim that every
change is audible. Natural intended-pitch accuracy remains unmeasured.

## Newly confirmed preprocessing mismatch

V53 compares the five training crops excluded by the experimental builders with
the existing app frontend, without changing the app policy. The app already
permits a low fraction of full-scale samples and applies uniform peak-protection
gain for resampling overshoot. Rare full-scale samples alone do not establish
that the whole recording is overloaded or musically unusable.

All five are technically accepted by the existing frontend. Four also meet the
five-second task minimum: vocadito13/vocadito18 and both M01 hums. Their native
fractions at or above.999 amplitude are respectively0.003279,0.0000136,0.0000866
and0.0001587, all below the existing.01 maximum. Recorded gain factors are
0.989893,0.995738,0.989639 and0.992161. The4.936s F01 clip remains out of target;
technical acceptance does not authorize padding it or silently changing the task.

This is a training/frontend mismatch, not a newly invented denoising or clipping
repair. V50/V51 stay fixed historical comparisons. Independent V53 replay completes:
all five working PCM files are recreated exactly, source/preserved-crop hashes and
policy/gain operations agree, and source-only short-support eligibility is measured.
Vocadito18 has room for0.24s support; M01's second recording has room for0.24/0.28/
0.36s. Vocadito13 and M01's first recording still lack the required source context.
These technical accepts are not new human-reference approvals, audited physical
training pairs or pitch-correction successes. The completed receipt is under
`2026-11-02_v53_frontend_parity/integrity_receipt.json`.

## Next action and limits

### Completed coverage and mechanism checks,4 November

V54 applies the unmodified app frontend uniformly to all33 training sources;
32 are duration-valid. Independent replay recreates33 working files and576
context arrays. V55 retains188 pairs/528 attempts, including12 newly admitted
pairs after the frontend correction, with432 historical cases matched and16 new
decoded audits/32 controls/288 arrays replayed.

V56 is now independently verified:4 checkpoints,112 original prediction arrays
and112 proposal plans. There are2 permitted unwanted proposals among84 fitted-source
combinations, and4 among28 held originals. Reliable-frame extremes reach26.887c
and28.174c respectively. These repeated model/source combinations are not new
people. The four held proposal sources exactly match V52's four changed originals;
the two fitted proposals were not rendered and are not measured delivered harms.

V57 uses fixed consecutive nonoverlapping10s windows over all existing training
sources. Of72 slots,53 meet duration/technical requirements and19 remain short
exclusions. All first33 window links,21 later working files and378 later arrays
are independently checked. V58 stages349 pairs/1152 attempts:188 earlier pairs
remain,161 are new;512 earlier cases,164 new decoded audits,328 controls and3,209
arrays replay. This adds acoustic variation, not people or fresh confirmation.
S25 now has26/48 scoreable attempts; several other reference ceilings still fail.

V59 shows that steady-pitch segmentation excludes some continuously reliable
melody. The predeclared V60 pilot changes only that source-region condition on
4 training windows/3 groups. It yields50 audited cases/64 attempts, compared with
zero constructible in those selected windows under V58. S10 contributes22, S15
12 and S20 contributes16. All50 retain two-sided context. Independent replay
checks56 decoded cases,224 negative controls,56 sample tampers and800 arrays.
Only196 negative controls have enough shared coverage; the rest are refused but
do not establish a pitch-error false-accept rate. No model has used these examples.

V61 traces saved predictions, permission, actual waveforms and audits for all
V52 cases, including deterministic representatives. Added-hum example C_222 repairs
entirely inside the injected0.50s support and benefits. C_210 has a0.24s injection
at2.48–2.72s, but permits2.52–2.78s:1,323 samples change beyond the injection.
Its three reliable requested frames beyond the injection have no training label.
Across all34 added-hum harm cases,22 include such guard-zone requests;75 outside
frames are unlabelled and832 requested outside frames are already labelled zero.
Thus both missing edge supervision and failure on existing preservation targets
need attention. Counts are retrospective diagnostics, not new validation.

V62 freshly replays four known22.05kHz-mono cases through the actual technical
frontend, context extraction, model, selection, rendering/rollback and encoded
verification. All four plans/outcomes/output waveforms reproduce V52. Request
durations are12.306,7.492,6.258 and5.607s. This small known-case timing sample
excludes Python startup, upload and human review; it does not establish deployment
latency or populationp95. Offline construction timing is a different measurement.

Current loss actually executes pairedA/C regression, paired-response preservation,
known-zero/original penalties and activity BCE. It is not the full proposed
BPDR/augmentation-only/frame-only comparison. Boundary channels69/70 are absent
from short-model features; zero boundary-view disagreement is by construction.

### Evidence-supported next action

The frontend/coverage work above completes the earlier next action; old outcomes
remain retained. Next scale V60's validated construction rule uniformly in its own
declared data phase, then use the [ranked plan](Short_Phrase_Transformer_Plan_2026-11-02.md)
for a bounded preservation-sampling comparison. Keep any new edge-anchor target
definition separate so its effect can be assessed. Do not begin another capacity,
seed or threshold sweep without new evidence. Keep all models inactive until
existing output qualification and reserved-data requirements pass.

The evidence does not prove that Transformers are intrinsically unsuitable. It
shows uneven transfer, limited positive-source coverage, selective-preservation
problems and a concrete preprocessing inconsistency. Earlier matched alternative
models also failed to qualify. More data improves some voices but has not produced
the requested broadly usable short-hum repair. Independent intended-idea/listening
review and the proposal's full BPDR objective comparison remain outstanding.


## Uniform melodic source construction, V63/V64

V63 metadata verification is complete:72 slots/53 working windows,27 unchanged source regions and26 changed, with audio/context bytes and partitions unchanged. V64 construction is terminal0 (session95750):445 passing examples/1152 attempts versus V58's349,17 contributing groups,53 valid original windows.432 case records exactly reuse unchanged geometry,171 valid examples reused;274 are newly measured at the new support geometry. Exclusions:332 no source region,304 short-window,71 three-tracker audit failures. These construction results now pass independent replay; they are not learned accuracy. All audit/renderer limits remain fixed.

V64 independent replay is COMPLETE:340 new decoded audits,1360 refused wrong-request controls (1240 with adequate common coverage),340 refused outside-sample tampers,274 labels and6096 arrays recomputed. All445 pairs are verified. Coverage comparison:127 newly constructible keys,31 lost at the changed locations,137 old-and-new valid keys with changed support; net+96 is a reference-coverage gain, not matched repair accuracy. S10/S15/S20 are newly represented.

Initial pipeline15131 is terminal1:the sampler placeholder used n*441 samples, giving n+1 controller frames for n feature frames. No fit/checkpoint or model score existed at failure. The fix uses (n-1)*441+1 samples; protection exactly matches actual-waveform policy on3 representative lengths/cases. Original failed declarations and failure receipt are retained under V65/V66; the training-recipe hash was reissued before fitting, with no data/model/loss/threshold change.

Current sequential pipeline session44844 is RUNNING from V65 training only:8 fixed-budget uniform/mined fits -> independent checkpoint/sampler replay -> detailed raw report -> V66 delivered-WAV evaluation -> independent delivered replay -> domain/duration/direction report. Every nonzero phase stops later work. Check `results/transformer_usability/preservation_pipeline_status.json` and poll44844 before restarting. No new model accuracy result or activation is claimed yet. The prepared raw reporter passes singer-weighting/empty-subset checks;3 mining unit tests and the refusal of unverified data also pass. V66's fixed output protocol is locked before new scores.


### Prospective output evaluation and detailed accuracy reporting

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

V65 first uniform-fold fit receipt now records1200 updates,21666 parameters and
42.305s optimization time; the mined counterpart is active. Fold1 uses5649 hard
negative draws with8751 uniform fallbacks. No combined score/checkpoint replay or
delivered result is complete yet; decreasing fit loss is not usable accuracy.


### Reference sufficiency and new-model preservation diagnosis

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


### Controlled target direction in the output report

The V66 output reporter additionally separates request-following from target
correctness: on registered positive plateau frames, it counts permitted requests
opposite the known counter-detuning target and delivered fine-pitch movements
opposite that target. Measured movements use the existing1e-6c numerical margin;
withheld/zero movements are not opposite edits. This is post-output controlled
reference scoring, not a changed runtime gate or natural intention claim. The
existing renderer audit still independently checks agreement with the request.
No new model score was used to choose these direction-report definitions.


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


### V65 verified negative result and live output evaluation

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


### V64 supervision geometry, no new labels

A data-only supervision-geometry diagnostic now completes on the verified V64
pool (`supervision_geometry_report.json`). Across445 pairs,3534 of7313 reliable
injected frame centers have registered positive labels (48.3%);3779 reliable
inside centers and3282 reliable near-outside centers are unlabelled. All445 pairs
have some unlabelled reliable near-outside centers. Positive-label fractions by
support width are29.9% (.24s),38.7% (.28s),52.8% (.36s),65.2% (.50s). Existing
known-zero labels total125836 frames. Counts describe overlapping derived frames,
not independent observations/people and not new trustworthy pitch labels.

No target, waveform, fit or runtime policy changed. This supports inspecting edge
supervision with the V66 trace; it does not prove gaps caused output harms or that
all unknown frames should be labelled. Any future waveform-command anchors must
explicitly differ from measured instantaneous-F0 labels and retain old evaluation.
The source is `tools/diagnose_short_supervision_geometry.py`; this is not a model
accuracy gain.


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

## V68 predeclared next experiment: upper-tail preservation loss

**Reason and ranking.** V67 shows harmful requests even on training-assigned
originals; V66 has2060/2064 requested frames already labelled zero, beyond the
292/282 unlabelled edge frames. Mean preservation loss may tolerate rare large
requests. Test its aggregation before changing edge labels, representation or
architecture. This is a hypothesis, not an established cause or promised gain.
The earlier fixed-parent sampler and the new loss are distinct variables.

**Primary evidence.** [Fan et al.,Learning with Average Top-k Loss](https://arxiv.org/abs/1705.08826)
formalizes averaging the largest individual losses between mean and maximum risk,
with classification/regression experiments. [Official code](https://github.com/YanboFan/learning-with-average-top-k-nips2017)
is MATLAB/synthetic-data code. Our per-minibatch,per-negative-class PyTorch adaptation
is not their exact training experiment; their theory does not guarantee deep vocal
repair or justify the fixed choice k=4. No external code/data dependency is added.

**Baseline.** Freeze the4 verified V65 uniform checkpoints and identical held arrays,
not mined variants. Their V66 output24 gains/148 harms/11 changed originals is the
matched delivery baseline on the same445/1152 examples and53 controls.

**One changed variable.** Replace each mean of12 squared normalized corrections
for C-zero and general-original rows with the mean of the largest4 of those12
squared values. Keep coefficient.5 for each class. PairedA anchor,pairedC target,
paired response and activity BCE stay exactly as V65. No new target or sampling.

**Fixed inputs/budget.** Reuse bit-identical V65 uniform physical traces and data
roles. Same21666-parameter Transformer,initial seed339,4 folds,1200 updates,
AdamWlr.0005/wd.001,gradient clipping1,48 forward rows/update. Exactly4 new fits;
4 baseline checkpoints are reused. No guide notes or development/reserved audio.

**Selection/stopping.** Retain old all-represented-group raw gates,zero changed
originals/known-zero harms and useful per-group delivered coverage/reduction;
retain unassessed groups and every construction exclusion. Replay checkpoints,
traces and output evidence. A zero-command collapse or lost positive coverage is
not success. No k/seed/epoch/threshold sweep after failure; stop this contrast after
its four fits. No promotion/UI change. Freeze the unchanged actual-WAV protocol
before inspecting new scores, reusing V66 baseline only after exact identity checks.

**Deferred.** Edge-command anchors remain a separate later contrast, retaining
positive plateau supervision and old evaluation. The geometry report counts gaps
but does not make all those centers valid observed-F0 labels. Full BPDR versus
augmentation-only/frame-only research conditions remain distinct and unproven.


### V68 implementation checks and live fit

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


### V68 initial completed fits, replay pending

V68 completes all4 predeclared fits. Initial raw summaries (independent replay
active in80654):uniform/tail overall error19.701240/20.995372c; HumTrans
16.923916/19.327460c; Vocadito20.858458/21.690335c. Original means improve
1.051955->.766431c (HumTrans.607125->.413704c,Vocadito1.242596->.917600c),
but both raw gates fail. This is a positive-preservation tradeoff, not usable
accuracy. No k/seed/epoch/threshold sweep; no extra fits. Await raw replay and
fixed V69 actual-output comparison before any decision. No activation.


## V70 predeclared experiment: complete original-frame query coverage

**Evidence/hypothesis.** V69 replay verifies the tail-loss variant32 gains/109 harms
versus24/148, but still11 changed originals. V67 baseline-query audit finds29734
of40899 reliable training-assigned frame/model positions directly or exactly
identically queried as zeros;782/1548 activation-tail frames and440/748 planned
frames lack such queries. Similar-input gradients are not counted, and seen
queries do not guarantee successful optimization: this is not proof of causality.
Original pools per fold are10552/10430/10837/9080 frames, below the existing14400
original-query budget, making a full query pass feasible without more updates.

**Baseline/changed variable.** Freeze the4 verified V68 upper-tail checkpoints and
V69 outputs. Change only general-original query scheduling:fixed-seed shuffles of
all training-original reliable zero-target frame positions without replacement,
repeating only after every eligible position is queried. Positive C,pairedA and
C-zero draws remain bit-identical. GeneralA record/group frequencies therefore
change from group-balanced replacement sampling toward frame-proportional full
coverage; do not describe those frequencies as unchanged. The upper-third loss
and all other training/runtime details remain fixed.

**Budget/roles.** Exactly4 new fits,4 existing folds,seed339,1200 updates,48 forward
rows/update,old optimizer/clipping/data/targets/architecture. Seed339+fold sets the
new query permutation;no adaptive scores or teacher. Skip zero-evidence windows
in sampling while retaining all53 waveform controls and every1152 attempt/445 pair.
Refuse if the full eligible pool cannot fit14400 draws;never silently claim partial
coverage as complete. No actual development or reserved-unseen audio is opened.

**Validation/stopping.** Prove all eligible original frames are queried, first24
trace entries per update unchanged,all targets/classes valid,repeat order deterministic,
then independently replay checkpoint predictions and actual WAV results against
V69 under identical policy. All previous accuracy/preservation/coverage gates remain;
query coverage is not gradient coverage or usable accuracy. No seed/epoch/threshold
sweep or extra fit after failure. A zero-command collapse or lost repair coverage
is not success. V71 output protocol must be locked before new model scores;baseline
reuse requires exact data/policy/prediction identity. No model promotion/UI changes.

**Priority.** This sampler contrast precedes deferred command-edge anchors because
it targets measured exposure gaps, retaining labels and the tested tail loss. Edge
labels,positive command targets,architecture changes and full BPDR research conditions
remain separate. No benefit is promised before measurement.


### V68/V69 complete and V70 active

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


### V70 verified raw outcome and additional head diagnosis

V70's4 fits and raw/query replay are COMPLETE:8 frozen/new checkpoints,996
held arrays and4 traces reproduce all summaries. All10552/10430/10837/9080 original
frame pools are fully queried within14400 slots each;positive/C-zero queries are
bit-identical. Complete query coverage does not imply nonzero gradient or convergence.
Raw tail/coverage errors20.995372/21.528909c; original means.766431/.764629c,
essentially unchanged. Within10c.184085/.163069; original activation-tail windows
29/24 but maxima26.922/27.314c. Both raw gates fail. New V71 actual-WAV comparison
continues in22672 with the V69 baseline inherited under identity checks. No sweep.

Additional diagnostic `diagnose_positive_fit_transfer.py` measures correction,
activity and amplitude on training-assigned queried/unqueried positive frames versus
held singers,with paired originals. Initial job37995 stopped on a0.000025c sparse-
batch float32 discrepancy before any final result; failure receipt is retained.
Current job69538 uses the original128-row inference blocks and requires exact held
prediction equality. No tolerance, model,fit or qualification threshold changed.
Its diagnostic is not an evaluated ungated model or calibrated-intention claim.
## V71 independently replayed on 6 November

Complete original-query coverage reduced verified harms from 109 to 89,
known-zero harms from 101 to 89 and changed originals from 11 to 8.
Beneficial edits fell from 32 to 22. The 445 scoreable cases came from
1,152 physical attempts, with 707 construction exclusions. Across represented
singers, delivered error was 23.903 cents for the V69 tail baseline and
24.226 cents for complete coverage, from 24.897 cents before correction.
The preservation failures and lower beneficial coverage rule out promotion.

The independent verifier reproduced 498 plans, 147 candidate audits,
119 encoded WAV audits, 445 reference scores and 53 original checks. The
frozen V69 baseline inherited 498 verified records under matching hashes and
policy. Results and integrity evidence are under
`results/transformer_usability/2026-11-04_v71_coverage_delivery/`.
The completed domain report shows 12 benefits and 11 harms on HumTrans,
versus 15 and 12 for the frozen tail. On Vocadito the corresponding counts
are 10 and 78, versus 17 and 97. One HumTrans original and seven Vocadito
originals changed under complete coverage.

### Representative replayed transactions

- C_137 and original A_vocadito_6__window01 share the same working clip.
  A −20-cent intervention spans 6.53–6.89 s. From the acoustic input
  features the model requests up to 25.13 cents on 6.60–6.84 s, and both
  render and encoded audits pass. Target error falls from 19.31 to 3.04
  cents with no known-zero harm. On the unchanged original, the model
  requests up to 24.65 cents on 6.62–6.86 s. Both audits also pass and
  the waveform changes. The output audits verify what was rendered
  but cannot determine whether the requested edit was warranted.
- C_550 has a +20-cent intervention at 2.88–3.16 s. The selected
  interval starts early at 2.80 s and stops at 3.06 s. Both audits pass.
  Target error falls from 17.08 to 13.32 cents, while known-zero error
  rises from zero to 0.109 cents under the declared scoring rule.
- Of 498 replayed transactions, 351 returned identity, 28 rolled back
  at the post-render stage and 119 passed encoded verification. Coarse
  median tracking error appears in 21 of 28 rollbacks. Those rollbacks
  are separate from the preservation failures in delivered audio.

## 6 November renderer control

The fixed eight-case V64 probe used four training-source groups per domain.
Forced zero-shift Praat resynthesis changed samples within each registered
support, with waveform residual RMS of about 0.12–0.42 on the normalized
sample scale. Per-case median absolute fine-pitch changes were 0.25–1.97
cents, compared with 19.73–32.52 cents on the registered shifted clips.
Outside-support samples stayed exact for all eight zero-shift controls.
A separate invocation reproduced every saved measurement.

The first invocation stopped because one inherited row had no renderer
receipt. The corrected run kept the same cases and checked the renderer
settings against the historical source declaration. The failed declaration
and reproducible measurements are retained under
`results/transformer_usability/2026-11-06_resynthesis_control/`.
These eight cases establish a waveform difference under forced resynthesis.
They do not show whether the trained model responds to it or whether the
same effect occurs in natural singing.

### Frozen V68 response to the control

All eight held-source V68 comparisons reproduced in a second invocation.
Re-extracted clean and shifted inputs matched the saved predictions exactly.
On seven cases, the mean absolute change in correction request under
zero-shift resynthesis stayed below 1 cent on the registered scoring frames.
In S7 it reached 16.94 cents across four frames, while the local request
peak fell from 22.34 to 1.49 cents. The actual shifted clip moved the same
four requests by 5.63 cents on average. Two controls still had request peaks
above 12 cents, and both corresponding clean recordings already crossed
that cut. The diagnostic therefore shows a large sensitivity in one case,
alongside false requests already present on clean audio. It does not
identify which acoustic feature drove the S7 change or establish its
prevalence. Saved values are in `prediction_results.json` in the same folder.

### S7 feature substitution

In the selected S7 outlier, replacing the clean pitch, voicing and pitch
delta inputs with their resynthesized values reproduced 16.9348 cents
of the 16.9353-cent mean absolute request change across four scored
frames. Replacing confidence and energy alone yielded 0.0450 cents.
A second invocation reproduced the values exactly. This points to
sensitivity in the tracked pitch and voicing inputs in this case.
It does not isolate individual columns or establish a population rate.
The feature combinations are diagnostic model inputs, not real recordings.
The declaration and result are `feature_trace_declaration.json` and
`feature_trace_results.json` in the renderer-control folder.
## Verified matched spectral-context result

The four held-source folds produced the following raw request metrics.
Eight fitted checkpoints were reloaded in a separate invocation and
all 1,494 saved prediction arrays matched.

| Arm | Held singer-macro error (cents) | Humming | Vocadito | Known-zero frames with request ≥12 cents |
|---|---:|---:|---:|---:|
| Matched zero-channel control | 20.626 | 18.57 | 21.48 | 6,822 |
| Added spectral context | 20.682 | 17.41 | 22.05 | 5,401 |
| Frozen V68 | 20.995 | 19.33 | 21.69 | 6,775 |

The spectral arm meets every assessed original-group mean limit and
every scoreable group MAE limit. Ten of 17 scoreable groups miss the
minimum signed response, and aggregate error exceeds the matched
control. The candidate failed the declared advance rule. The WAV stage
was therefore skipped. Artifacts are under
`results/transformer_usability/2026-11-06_short_spectral_context_v2/`.
## Replayed longer-hum physical probe

On seven existing training-role clips, eight of 14 registered attempts
were scoreable in both shift directions across four people. The 0.24 s
supports fell between 11.46 and 14.34 s, and each scoreable case had
three common plateau frames. All 32 negative controls were rejected.
Four attempts lacked sufficient reliable context and two failed
the sample-integrity limit. A fresh invocation rebuilt all 14
outcomes and matched the saved report.

This establishes construction feasibility for the longer duration band
in a small existing-source sample. The probe retained metrics and
support locations, with decoded waveform checks performed in memory.
No model ran on these cases. Actual training and fresh confirmation
still need their own declared data and verification. Artifacts are
under `results/transformer_usability/2026-11-06_long_hum_physical_probe/`.

### Frozen V68 transfer beyond 10 seconds

The checkpoints that held out each singer averaged 13.261 cents of raw
error versus 20 cents for a zero command. Requests had the target sign
on 21 of 24 positive frames. Seven of eight scoreable cases had lower
raw error than zero, with one marginally worse. Every one of the four
originals received a request of at least 12 cents on reliable frames.
Six of eight corrupted cases also had a request of that size on a
known-zero frame. The largest original request was 26.50 cents.

The separate replay reproduced every outcome and decoded PCM hash,
including the six exclusions. The fixed examples are previously
inspected training-role sources evaluated with singer-held-out
checkpoints. They provide raw engineering transfer evidence. Output
permission, delivered WAV benefit, and fresh intended-note accuracy
remain unevaluated on this sample. See `transfer_results.json` in
the physical probe folder.

### Frozen V68 longer-hum delivery diagnostic

The existing policy returned identity for six altered clips. Two plans
were permitted but rolled back in the renderer. All eight outputs and
all four originals stayed exact. Mean target error remained 19.995 cents
before and after delivery. No candidate reached the encoded audit.
Both altered rollbacks were on F05 and reported “Correction outside
permitted support” on plans ending at 12.00 s. Candidate filtering on the
six identity cases recorded seven insufficient supports, six below
activation, and four protected-region refusals. These are candidate
counts, not separate clips. A second invocation matched every outcome.
Inspect the controller-to-renderer interval grid before attributing the
F05 failures to audio quality. See `delivery_results.json`.

### F05 endpoint trace

A synthetic 12 s fixture reproduced the failure. The feature grid
includes a frame at exactly 12.00 s, where the decoded waveform has
no sample. The controller permits the half-open interval [11.72,
12.00), but the nonzero endpoint is outside it. Both actual F05
plans reproduced exactly and had precisely one unsupported request,
frame 600 at 12.00 s.

Clearing only frame 600 and replanning allowed both diagnostic
transactions to pass render and encoded audits. The requested
support starts at 11.72 s for the −20-cent case and 11.50 s for the
+20-cent case. The injected region is 11.46–11.70 s in both.
Decoded scoring confirmed that the −20-cent case changed 6,108 samples
outside the injection and none inside, with target error unchanged at
19.906 cents. The +20-cent case reduced target error from 20.267 to
1.320 cents while changing 6,611 samples outside the injection.
Known-zero error rose from zero to 0.718 and 0.378 cents respectively.
Both fail the declared preservation rule. The first scoring attempts
stopped when they incorrectly assumed both edits were entirely outside
the injection. The completed measurement counts overlap explicitly.
These diagnostic WAVs are
experimental, and the existing policy and qualification decision
remain in effect. See `endpoint_trace.json` and
`endpoint_decoded_scores.json` in the physical probe folder.
