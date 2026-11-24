# Short-phrase Transformer: implementation and accuracy plan

Version 1.10, updated4 November2026. Owner-approved target: a 5–15-second melodic hum
with short notes. This replaces held tones as the engineering usability target;
it does not claim that short-note repair already works.

## Current decision and ranked experiments

Read [continuation state](CONTINUATION_STATE.md) first. Ranks1–3 are complete;
rank4 is active and rank5 remains deferred. No candidate is qualified or activated.
V69 replay verifies the tail-loss variant32 beneficial repairs/109 reference harms/
11 changed originals versus uniform24/148/11 on the same445 scoreable/1152 attempts.
V68 raw error worsens19.701->20.995c while original mean falls1.052->.766c; raw
means do not decide useful output. V67 also finds fitted-source proposal failures,
and its query audit distinguishes sparse direct exposure from optimization failure.

V70 now changes only general-original query scheduling under the fixed tail loss,
with4 new fits and frozen V68/V69 parents. Original group/record frequencies do
change;positive/C-zero queries,data,labels,model,loss and runtime gates do not.
Live pipeline22672 runs fitting,raw/query replay and locked V71 WAV checks. Do not
repeat old seeds/epochs/thresholds. The3 reference-insufficient groups stay unassessed.

| Rank / hypothesis | Baseline and one changed variable | Data roles / maximum budget | Selection and stopping rule |
|---|---|---|---|
| 1: steady-pitch construction unnecessarily excludes reliable melody | V58 has no scoreable example in the4 selected S10/S15/S20 windows. Use the longest continuous reliable span instead of the70c pitch-region split; retain150ms context and every existing waveform-audit threshold | Training-only selection from V59 geometry,4 windows/3 existing people,64 duration/sign attempts,0 fits; no guide notes or evaluation audio | At most64 attempts. Unchanged/opposite/wrong-magnitude/displaced-support and outside-sample-tamper controls must refuse. Halt on a false acceptance. Reproduce source geometry, encoded interventions, controls and context before admitting data. Passing construction is not model accuracy. |
| 2: uniform preservation samples miss rare high-request negatives | Uniform sampler versus fixed-parent mined known-zero frames on exactly the same verified data; same Transformer, loss, optimizer and operating policy | Existing singer/source folds, seed339,1200 updates, at most4 folds x2 arms=8 new fits. Teacher and mining data exclude the held-out group. Half of zero-C/original draws may use the top20% unprotected negative requests>=6c; positive/group/record draws stay fixed | Old raw gates, original-event diagnostics and actual-WAV preservation/benefit/coverage decide adoption, never pooled MAE alone. Three helper unit tests pass; no empirical benefit yet. No seed/threshold sweep after failure. If both arms fail, stop this contrast and investigate new evidence/reference coverage. |
| 3: rare large zero errors survive mean loss | V65 uniform baseline, same exact trace/model/data; replace C-zero and generalA mean-square aggregates with top4-of12 means, coefficients unchanged | Four frozen baseline folds plus4 new fits,seed339,1200 updates; no new labels or sampler | Existing raw and actual-WAV preservation/coverage gates; no k/seed/epoch/threshold sweep, no promotion from zero-command collapse |
| 4: original query duplication leaves known-zero positions unqueried | V68/V69 tail baseline; generalA uses fixed-seed full reliable-frame passes without replacement, other classes unchanged. Group/record frequency changes explicitly | Four frozen parent folds plus4 new fits,seed339,1200 updates,48 forward rows;eligible pools10552/10430/10837/9080 fit14400 draws | Refuse insufficient budget;prove complete query coverage and identical positive/C-zero trace;old raw/actual gates,all attempts retained,no parameter sweep or query-coverage-as-accuracy claim |
| 5: unlabelled guard zones permit overlong correction tails | Old outside+70ms zero-mask versus separately specified waveform-preservation anchors nearer the true injected edge; retain positive plateau supervision and keep the sampling/loss comparison separate | Deferred until rank2 results; at most one4-fold/two-condition contrast after target semantics are declared. Exact unchanged waveform regions are not the same as unbiased instantaneous-F0 measurements | V61 finds unlabelled edge requests in22/34 added-hum harm cases, but832 already-labelled zero-frame requests also remain: guard gaps are not the only cause. Do not label estimator smear as a natural pitch error or weaken evaluation. |

Rank1 pilot is now complete and independently replayed:50 physical passes/64
attempts versus0 constructible under the old condition in those selected windows;
all50 retain the required context. All224 wrong-request controls refuse,196 with
adequate shared coverage, plus56 outside-sample tampers. This supports scaling the
same source-region rule uniformly to the53 working windows in a separately declared
data phase, keeping the72 slots/1152 attempts and existing audit limits. It does not
show that a trained model improves. Do not repeatedly sweep region thresholds.

V63 independently rederives all72 source-window region records:53 working clips,
27 unchanged regions and26 changed. V64 completes445 passing examples/1152 attempts (versus349); independent
replay is complete for this uniform physical construction with four wrong-request controls and sample tamper checks on each new
render. Only identical PCM plus identical support geometry permits reuse. Independent
replay passes340 audits/1360 controls/340 tampers/6096 arrays before fitting. The prepared V65 runner refuses unverified
V64 inputs and will compare uniform versus preservation-mined sampling in eight
fixed-budget fits. Its syntax check and three sampler tests pass; the physical replay and coverage report have passed and V65's8 fits finish with initial raw failures and replay is active in session44844.
All folds are inherited from V51, with the matching frozen teacher excluded from
the held-out singer groups. No actual development/reserved audio is opened.

Further Transformer/GRU/TCN/ExtraTrees changes stay deferred until a new
representation-specific hypothesis survives these data/objective checks. Any
future architecture comparison must use the same short-melody inputs, folds and
sampled data, with explicit neural/classical budgets. No extra family fit is declared.

The V58 reference ceiling remains below25% coverage in seven groups even when
shorter-than5s windows are excluded from that calculation. S25 improved from2/16
to26/48 scoreable attempts, but this does not fix other groups or prove model
benefit. Missing reference coverage must not become a model-failure percentage.

## What primary research supports and what it does not

| Method/source | Required information and evidence | Cost / fit to this component |
|---|---|---|
| [Deep Autotuner](https://arxiv.org/abs/2002.05511) | Uses vocal and accompaniment spectrograms to infer continuous shifts; its karaoke experiment has musical context unavailable in an isolated free hum | High data/implementation change; not a drop-in argument for replacing our Transformer with a GRU |
| [Diff-Pitcher](https://engineering.jhu.edu/lcap/data/uploads/pdfs/waspaa2023_hai.pdf) | Controls resynthesis toward a target pitch profile; automatic examples use templates/scores. This addresses rendering quality, not unknown intention by itself | High: new neural renderer/target interface and preservation calibration. Current selection/preservation failures take priority |
| [PESTO official implementation](https://github.com/SonyCSLParis/pesto) | Estimates performed pitch. Transposition-equivariance does not supply intended-note truth | Already used upstream. Any version/weight change needs an isolated measurement/control study; no automatic dependency upgrade |
| [Hard-example mining](https://arxiv.org/abs/1604.03540) | General training-sampling evidence from object detection, not demonstrated vocal-repair accuracy | Low local training change. Our fixed-parent sampler is an adaptation, not that paper's exact online algorithm; benefit must be tested here |

The earlier [matched family comparison](Model_Family_Comparison_2026-10-30.md)
used held-voice inputs: Transformer23/200 delivered improvements, GRU18/200,
TCN19/200, fresh ExtraTrees20/200, all preserving50 originals. It does not rank
families on the new short-melody cohort. The V41 short-context comparison also
failed all-group qualification. The earlier FCPE sidecar lost to the existing
YIN measurement on audited references; do not repeat it without a new hypothesis.

## Objective and usability audit

### Next engineering contrast for spectral context

The positive-fit diagnostic reproduced all 445 held predictions. Its fitted
positive error was 8.726 cents versus 20.995 cents held out. Ungated amplitude
reduced held positive error to 17.846 cents but had a 20.232-cent magnitude
on the matched clean excerpts. These observations motivate testing the input
representation with the activity gate retained.

The 6 November input preflight read the existing V64 analysis arrays. All 445
scoreable corruptions and 53 originals have finite 71-column arrays, each
252–501 frames long. The short model uses eight derived features and leaves
out the 64 log-mel bands. Test whether spectral context reduces held-source
error while preserving originals and known-zero regions.

Use the frozen V68 four-fold assignments, sampling traces, paired examples,
upper-tail loss, seed 339, optimizer, and 1,200 updates per fit. Fit two arms
per fold with identical initialization and a 16-channel projection. Both
receive the existing eight channels. The control appends eight zeros. The
test appends the mean of each consecutive eight log-mel bands, expressed
relative to the query-center frame and divided by 20 dB, clipped to ±4.
Apply the existing voicing and padding masks. No boundary columns, guide
notes or knowledge of corruption intervals enter inference. Keep this
experimental implementation separate from the existing checkpoint loader.

Before fitting, bind the implementation, dataset and traces to a machine-
readable declaration. Check that the widened zero-input control reproduces
the frozen model at initialization when copying its shared weights. Compare
the fitted arms by group, domain and direction under existing raw correction
and preservation gates, including local extremes. Run the existing delivered
WAV policy and independent replay if the spectral arm passes those gates and
improves held positive error versus both the matched control and V68.
Stop after eight fits or a failed preflight. Keep all exclusions and reserved
sources sealed. A gain on controlled detuning can include sensitivity to the
renderer and needs separate real-hum confirmation. This contrast leaves the
proposal's augmentation-only, frame-only and full BPDR comparison open.

### Resynthesis control

V64 generates controlled shifts with Praat overlap-add, which reassembles
windowed waveform segments at pulse locations. The model could learn a
change introduced by resynthesis as well as a change in performed pitch.
PESTO estimates the pitch present in acoustic frames. It provides no
independent label of the intended note. [Praat overlap-add manual](https://praat.org/manual/overlap-add.html)
[PESTO study](https://arxiv.org/abs/2508.01488)

Run a no-fit diagnostic with up to four source groups per domain, choosing
the first scoreable V64 row per group in manifest order. Use the registered
support and original pitch tier, force overlap-add without a pitch shift,
then apply the existing local envelope. The normal renderer returns the
original for a zero command, so this path belongs only in the experiment.
Compare original, forced resynthesis and shifted audio for waveform residual
level and fine-pitch change on registered positive frames. Keep unavailable
measurements visible. Eight cases maximum, no tuning and no confirmation
access. This checks whether the control can expose resynthesis differences
for later model diagnosis.

For the next no-fit check, use exactly those eight cases and the V68
checkpoint that held out each source group. Re-extract features for the
clean, forced zero-shift and stored shifted waveforms with the same input
pipeline. Compare gated predictions on the registered positive mask, using
that mask only for scoring. Report changes from clean, absolute request
peaks and counts at the existing 12-cent diagnostic cut. Record differences
between re-extracted clean/shifted inputs and archived predictions. The
sample bounds this diagnostic to the selected cases and does not change
qualification rules.

For the S7 outlier only, compare the original and forced-resynthesis feature
arrays with the frozen V68 model. Starting from the original, substitute
the resynthesized pitch/voicing/delta columns together or substitute its
confidence/energy columns together. Also score the full original and
resynthesized arrays. Keep the same registered scoring frames. These
hybrid arrays are an exploratory model sensitivity check on a case selected
after the first diagnostic. They are not examples of naturally occurring
audio or an estimate of how often the sensitivity occurs.

The current paired training path actually optimizes original/zero preservation,
C correction, paired(C-A) response and activity BCE. It does not execute the full
proposal comparison of BPDR versus augmentation-only versus frame-only. The short
features omit boundary columns69/70, so unchanged predictions under boundary-view
perturbations are tautological input invariance, not evidence of a learned BPDR
advantage. Keep the matched scientific comparison in the research plan; a safe
engineering candidate alone cannot complete that claim.

Before designing the matched short-phrase objective run, check whether the
existing V64 main and +40 ms boundary views differ inside the actual
Transformer context. Count affected queries on positive, known-zero and
original masks separately. Verify the acoustic columns are identical between
views. This preflight uses the existing training-source arrays only and
fits no models. If the boundary input never changes at eligible queries,
the archived view perturbation cannot test learned invariance. The A/C
counter-detuning contrast can use the existing aligned paired targets,
including their known-zero frames.

The preflight verified 53 original and 445 controlled arrays. With the
existing voiced token mask, boundary differences occurred somewhere in
the context of all 3,534 positive, 125,836 known-zero and 13,633 original
queries. At the query center, the respective counts were 2,826, 113,615
and 12,241. On humming, 341 of 727 positive centers changed, while all
727 contexts changed. One original had no eligible query. All first 70
columns matched exactly between the compared views. The receipt is
`results/transformer_usability/short_boundary_view_preflight_2026-11-06.json`.

The next comparison should use A/C main and B/D +40 ms views with the
existing short context and grouped splits. Give both boundary-aware arms
the same boundary channel, and zero that channel in the parameter-matched
frame-only arm. Use shared anchors and smoothness across all arms. Only
full BPDR adds boundary agreement and counter-detuning response, scored
on aligned positive and known-zero frames. Keep original controls and
the common sampling schedule. Before fitting, verify numerically that
the frame-only arm is invariant by construction, that the boundary-aware
inputs differ, and that each added term produces gradients on a fixture.
Freeze the exact budget, loss weights and evaluation procedure in the
runner declaration. This compares objectives on short phrases after the
spectral input contrast failed, with no dependence on reserved audio.

For the runner, freeze four V68 singer/source folds and the complete 1,200
update V68 trace for each fold, with seed 339 and the saved AdamW settings.
Reuse the same 16-channel projection shape in all arms, with eight acoustic
channels, one boundary channel and seven zero channels. Use the existing
official project objective with default weights on predictions and targets
divided by 30 cents. Use the common anchor and smoothness losses for the
first 200 updates, then enable the two response terms only for full BPDR.
For each traced center also evaluate its next grid frame to measure
smoothness. Mask unavailable frames. Use A/B as C/D with zero target for
general original queries. Compare full BPDR with augmentation-only for
held boundary sensitivity, and with both controls for held counter-detuning
error, raw correction and preservation. Require the existing raw gate and
lower held counter-detuning error than both controls before WAV testing.
Report boundary sensitivity for the frame-only arm as structural. Stop
after 12 fits and checkpoint replay. A comparison of objectives does not
establish musical intent or satisfy production qualification by itself.

The 6 November duration audit read the working WAV headers for this exact
V64 pack. The 5–7 s band contains 3 original windows and 28 scoreable
pairs. The >7–10 s band contains 50 originals and 417 pairs. The >10–15 s
band contains zero. HumTrans contributes 2 originals/16 pairs and
12 originals/84 pairs respectively in the two covered bands. Even a
passing objective result would therefore leave longer hums unassessed.
The historical HumTrans declaration caps working copies at 10 s.
Seven hash-verified native training-role recordings across five people
exceed 10 s and could support a separately declared engineering pack.
Any additional windows from those people must keep the same source roles.
Fresh confirmation needs longer recordings from its own reserved people.

For the longer-clip geometry check, use all seven existing training-role
sources over 10 s. Take the first up to 15 s, select the highest-energy
native channel as before, and use the same PCM24 working format and
reliability extractor. Record V64 continuous-region eligibility for each
width, both anywhere in the longer clip and wholly after 10 s. The second
count tests whether new audio can support cases. It does not change
the V64 center rule. This check makes no render, correction or fit.

The geometry run found 18 eligible source-width combinations across five
of seven longer clips. Eleven combinations on four clips still qualify
when the entire reliable region and its required context must lie after
10 s. One clip has no qualifying region anywhere and one fails the
existing sample-integrity limit in the extended window. No thresholds
changed. Use `long_hum_geometry_preflight_2026-11-06.json` for the
per-source breakdown. Actual rendering and independent reference checks
would be needed before these clips contribute any scoreable examples.

The next physical probe is limited to the shortest registered width,
0.24 s, with −20 and +20 cent shifts on all seven longer clips. Retain all
14 attempts, including exclusions. Center each shift in the longest
continuous reliable span wholly after 10 s, with 150 ms of context on
both sides. Keep V64 rendering, PCM24, three-tracker audit, unchanged,
opposite-direction, wrong-magnitude and wrong-location controls, and
plateau/context checks. Require exact decoded samples outside the
support. Any negative-control acceptance invalidates the probe. Rebuild
from sources in a second invocation. No tuning, model fitting, held-out
confirmation access or alteration of the active objective comparison.

For transfer beyond 10 s, rebuild the eight replayed cases and use only
the V68 checkpoint that held out each singer. Compare raw requests
against the injected targets and against a zero-command baseline. Report
per-case direction, requests on original and known-zero frames, and
retain all six construction exclusions. Freeze the checkpoints and
existing masks, replay the result, and keep it separate from the running
objective comparison and from fresh confirmation. This probe measures
raw transfer and uses no delivery-stage pass criterion.

Follow with a separate delivery diagnostic on the same eight cases and
four originals. Keep the V71 selection, permission, renderer, decoded-WAV
audit and scoring rules fixed. Verify rebuilt input hashes against the
transfer receipt. Report every withheld or rolled-back case, actual error,
and changes to originals. Use the known supports only for scoring. This
is a frozen rejected-model diagnostic with no new fit or promotion.

For the two F05 renderer rollbacks, test whether the feature frame at
the exact waveform end is the only requested frame outside the permitted
half-open interval. Rebuild both cases and reproduce the original plan.
In a separate diagnostic arm, clear only the zero-sample terminal frame,
rebuild permission with the same limits, and rerun the same renderer and
audit. Stop if any other frame falls outside permission. Keep the previous
results and executable policy intact.

Implement the conservative endpoint guard as a separate fixture-planning
variant. Withhold an otherwise permitted proposal if a nonzero request
has no corresponding waveform sample. Leave the curve and all old
limits intact. A request on a final frame with a real sample still goes
through the existing gates. Verify both boundaries and confirm the
withheld proposal never calls the renderer.

The app already has technical acceptance, resampling peak protection, acoustic
evidence/intent feedback, whole-recording review and optional noise previews.
After model qualification, useful additions are time-specific explanations of
which stage blocked repair and a technical comparison of multiple takes using
clipping/reliable-voice coverage. Neither should judge musical intention or infer
key, tempo, final notes or harmony. Build on existing feedback rather than duplicate it.

Report per-domain/singer/direction outcomes, rollbacks, original and known-zero
preservation, coverage and stage latency. Historical V52 timings were not recorded.
V62 now replays four fixed known22.05kHz-mono cases through input integrity, fresh
features, model, selection, rendering/rollback and encoded verification:5.61–12.31s
per request, with first/warm initialization retained. These are small-sample
diagnostic timings, not populationp95 or live UI qualification; startup/upload/
review time is excluded. Polling durations are not latency. Natural intended-pitch accuracy,
controlled recovery and downstream extraction benefit remain separate outcomes.

## Historical stage notes (superseded by the current decision above)

**Current verified findings:** [V51/V53 result](HumTrans_Transfer_Result_2026-11-02.md)
completes four new real-hum transfer fits and8-checkpoint/408-array replay. Hum error
improves but Vocadito regresses; both represented-group gates fail. V52 actual-WAV
evaluation runs under fixed policy. Existing-app frontend behavior exactly
recreates five previously excluded working copies; four meet target duration and
two have short-support context. Next data preparation must use that frontend
uniformly across the declared cohort, preserving old results and audit thresholds.

**Local-data correction:** [V49/V50](Local_HumTrans_Recovery_2026-11-02.md) recovers
the existing24-recording HumTrans intake, preserves established source roles and
begins192 declared training-only short interventions. The V48 lack-of-data claim
and proposed person split are superseded. Complete independent replay before
retraining; current candidates remain unqualified.

**Current implementation:** [V43–V47 paired experiment](Paired_Short_Transformer_Result_2026-11-02.md)
adds explicit experimental short-audit calibration,115 staged physical pairs and
eight paired Transformer fits. All-group raw gates still fail; actual delivered
outputs contain32/36 known-zero harms and change2/4 originals across the two arms.
No candidate qualifies. Next work is [real-hum metadata/data preparation](HumTrans_Data_Plan_2026-11-02.md)
with pre-inspection person/composition separation. The metadata checker passes
synthetic tests, but real acquisition has failed; no new recordings are claimed.
This current direction supersedes older next-iteration paragraphs below.

**Latest evidence:** [V41/V42 complete result](Short_Phrase_Model_Result_2026-11-02.md)
rejects the first revised short-phrase candidate after12 fits and independent
actual-WAV replay:4 target gains,4 target harms,2 changed originals, worse remaining
error. All-group raw gates also fail. Preserve the broad owner target and all
failed duration/source groups. The next iteration must improve identifiable local
short-event evidence, training-source coverage and known-zero preservation before
another qualification attempt; no UI substitution or activation.

## Outcome and research boundary

Prepare a normal short solo vocal idea, retain its timing, rests and expression,
and offer only local waveform repairs that improve independently measured delivered
audio. The research remains boundary-robust pitch repair under segmentation errors,
with counter-detuning and preservation anchors, compared with augmentation-only and
frame-only controls at matched acceptance coverage. More accurate correction alone
does not establish the BPDR research claim.

BPDR owns audio validation, relative local repair, protected intervals, verification
and accepted-WAV provenance. It does not output a new symbolic melody, intended-note
sequence, beat, key, genre or harmony. A sustained incorrect note without an
observable local anchor can have ambiguous intention: abstain, retain the audio or
ask for review; do not infer intention by snapping it to a scale.

## Why another old-style training run is premature

| Existing constraint | Effect on the new target |
|---|---|
| Acoustic eligibility requires a 1.10 s stable region | Ordinary short notes are ineligible before prediction matters |
| Renderer/controller minimum support 0.86 s | A short-note proposal cannot reach delivery |
| 80 ms ramp + 100 ms audit guard at both ends | Removes 360 ms from a support before audit samples are counted |
| At least ten common 20 ms audit frames | Needs about 180–200 ms more usable interior; grid alignment and missing frames matter |
| Training pair minimum deviation 1.16 s + two-sided context/trim | Approximately 1.80 s stable source region needed |

The frame-count requirement is not proof of statistical independence: overlapping
estimator windows are correlated. Reducing the frame count arbitrarily is not a
validated short-note audit. Model-family comparisons already made do not show that
replacing the Transformer resolves these geometric constraints.

## Work sequence and exit evidence

| Work | Concrete implementation | Required evidence before proceeding |
|---|---|---|
| SP-01: measurement feasibility | Versioned short-support renderer probe; characterize original/decoded pitch, direction, frames, grid alignment and exact untouched samples | Retain all planned supports and failures; distinguish synthetic execution from real voice evidence |
| SP-02: short-phrase recovery data | Deterministic permitted real-voice 5–15 s crops; duration strata 0.20–0.35, >0.35–0.50, >0.50–0.80 s; local physical deviations, clean originals and transition/glide/vibrato negative controls | Count all attempted crops/variants and source groups; audited realized perturbations; no annotation/support/reference in inference |
| SP-03: short-support policy and audit | Separate opt-in protocol/version; boundary-aware acoustic eligibility; explicit abstention near unreliable boundaries; validate estimator temporal support and measured displacement before selecting timing constants | Decoded-WAV checks plus independently replayed direction/error/preservation controls on real voice; publish timing limits prospectively before comparison |
| SP-04: matched Transformer retraining | Retain genuine sequence Transformer; add dense short-deviation supervision, unchanged-expression examples and duration-balanced sampling; compare current frozen model, retrained Transformer and matched ExtraTrees | Singer/source-held-out folds, same data and budget, no test-driven seed selection; raw prediction and actual delivered comparisons |
| SP-05: independent usable-accuracy check | Freeze data recipe, checkpoint, support policy, renderer, audit and operating point; reserve unfamiliar sources before acoustic inspection | All attempts, per-duration/source/direction results, unchanged controls, artifact checks, calibration, latency, reference limitations and human review |
| SP-06: component integration | Only qualified opt-in model; repair preview/protection/accept/reject; compose selected edits from original and re-audit final WAV | Compatible accepted-only handoff; no modification to other members' musical-analysis features |

SP-01 is implemented as an experimental tool, not a runtime feature:
`tools/probe_short_phrase_renderer.py`. Its declaration fixes sixteen synthetic
requests (two registers, four support widths, two directions) before measurement.
The default 0.86 s renderer restriction is checked; a separate diagnostic renderer
uses a 0.24 s minimum to measure feasibility. Ramp and all audit thresholds remain
unchanged. No permission controller is bypassed for a user export, no model is
trained or scored, and no synthetic result is a human-vocal accuracy claim.

**Completed SP-01 probe, 2 November:** all sixteen experimental renders completed;
default renderer rejected all sixteen short supports. All eight zero-request
source fixtures were exact identities, and decoded edited outputs retained exact
outside-support samples. The unchanged audit passed four 0.70 s cases (18 common
frames each). All four 0.50 s cases passed the other audit checks but failed the
ten-frame minimum (eight common frames). All eight 0.24/0.36 s cases had zero common
interior frames. This proves neither short-hum accuracy nor production permission.
Results are retained at `results/transformer_usability/2026-11-02_v38_short_phrase_probe/`.
The initial run was interrupted after four cases; a JSON tuple/list comparison
fix and source-hashed resume amendment are retained before the full sixteen-case
repeat. No result or threshold was selected using the incomplete run.
Independent replay (`tools/verify_short_phrase_renderer.py`) checked all sixteen
decoded files and re-extracted/reproduced all sixteen audits exactly. It also
verified that V37's locked component runtime closure is unchanged. The initial
verifier comparison needed JSON tuple/list normalization; no measurement changed.
The completed receipt is `integrity_receipt.json` in the probe result folder.

**Subsequent real-voice timing probe:** [V39](Short_Support_Timing_Study_2026-11-02.md)
replays 240 decoded audits and 144 construction exclusions across three arms. On
real voice, baseline timing passes 8/96 planned requests, guard-only 17/96 and a
shorter ramp plus guard 23/96. These are forward perturbation measurement passes,
not learned repair accuracy. The 0.24 s gap, grid-sensitive 0.36 s results and zero
JSUT passes remain unresolved. Ten-frame/error/direction limits are unchanged;
there is no production selection. V40 preserves all existing Vocadito singer
partitions and reserves 21 training recordings from 14 singers before the next
short-phrase construction; no training or research-reference approval is claimed.

SP-02/SP-03 must be designed together: training labels must be observable with the
new measurement method. Do not enlarge an edit across an intentional note transition
merely to collect ten frames. No production timing thresholds are selected by this
plan. The declared duration bands are development coverage targets, not validated
supported durations.

**Current implementation:** [V40/V41 execution](Short_Phrase_Training_2026-11-02.md)
completes the first training-only melodic construction:44 audited pairs/336 attempts,
19 valid original controls,6 contributing singers; independent128-audit/1,134-array
and44-label replay passes. Short-scale Transformer inputs and an experimental
permission module are implemented;9 contract checks pass. Four-fold matched fitting
and delivered-WAV replay subsequently completed with rejection; see the latest
result above. V42 was declared before final raw results.
No completed usability, independent-review or production claim follows from this.

## Accuracy report after training

Report separately:

1. **Construction:** attempted recordings/variants, valid references, excluded and
   unassessed cases, by source and duration. A source with no valid reference is
   unassessed, not silently removed or assigned zero model accuracy.
2. **Prediction:** singer-macro correction MAE in cents, within-10/25-cent fractions,
   sign/response, uncertainty calibration and original-control predictions.
3. **Actual output:** decoded-WAV remaining error and reduction from the corrupted
   input; verified beneficial repairs / all attempts and / scoreable attempts;
   accepted harmful repairs, withholding/rollback reasons and acceptance coverage.
4. **Preservation:** every original control unchanged, exact samples outside edits
   and protected intervals, timing/rest preservation and expressive negative controls.
5. **Usefulness:** per-person/per-duration/direction coverage, processing time,
   participant intention judgments and two independent musical reviews. Controlled
   recovery error is not unknown natural intended-note error.

Retain the old 25% per-group repair coverage and error-reduction targets as historical
comparators. The new acceptance specification must additionally state supported
duration bands, worst-group results, confidence intervals and reference sufficiency
before the final test. Keep max 30 c, max 25% edited duration, zero tolerated accepted
wrong-direction frames, exact untouched/protected samples, nonclipping finite output
and explicit withholding unless an independently justified versioned experiment
establishes a replacement. Any change to timing/measurement is reported as a new
protocol, not an improvement under identical old rules.

## Collection and research evidence

Use the [review/collection plan](Review_and_Collection_Plan_2026-11-01.md): eight
planned adults × six phrases, four development and four untouched participants,
two independent reviewers. No recruitment, consent, institutional authorization,
ratings or musical intention labels are currently claimed. Record normal 5–15 s
hums; do not require long sustained tones to fit the system. Guided references need
approved stimuli and independently aligned intended-reference information; free
ideas may remain unsuitable for numerical intended-pitch accuracy.

V37's inspected twelve sources remain diagnostic if used here. A future independent
confirmation requires a new reservation and a frozen short-phrase pipeline.

Primary research context: [BERT-APC v3](https://arxiv.org/abs/2511.20006v3) separates
stationary pitch from expression and assesses rendered audio, but its symbolic
musical-context inference exceeds this component's scope. [PESTO's official
implementation](https://github.com/SonyCSLParis/pesto) supports configurable pitch
inference; a smaller hop does not by itself prove adequate independent temporal
resolution or cents-level accuracy on short hums. These are design motivations,
not evidence that this BPDR pipeline is already usable.


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


### Controlled target direction in the output report

The V66 output reporter additionally separates request-following from target
correctness: on registered positive plateau frames, it counts permitted requests
opposite the known counter-detuning target and delivered fine-pitch movements
opposite that target. Measured movements use the existing1e-6c numerical margin;
withheld/zero movements are not opposite edits. This is post-output controlled
reference scoring, not a changed runtime gate or natural intention claim. The
existing renderer audit still independently checks agreement with the request.
No new model score was used to choose these direction-report definitions.


### Verified rank2 result

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


### V66 preservation tradeoff pending replay

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
