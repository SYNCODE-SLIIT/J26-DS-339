# PP1 master plan: BPDR and team integration

**Project:** J26-DS-339  
**Individual:** Liyanage L. N. P., IT23141506  
**Planning date:** 27 September 2026  
**User-supplied PP1 deadline:** 25 October 2026  
**Implementation location:** `C:\Users\lnipu\Projects\Music Rp\J26-DS-339\experiments\bpdr_input`  
**Initial dataset:** `C:\Users\lnipu\Projects\Music Rp\Input Validation\vocadito`

## 1. What PP1 must demonstrate

Build a working input-validation and bounded-repair component, accompanied by a small reproducible experiment that makes BPDR's research question visible. The examiner should be able to identify the recording problem, see the proposed mechanism execute, inspect comparative evidence, and distinguish completed work from remaining research.

This is a progress-presentation plan, not a requirement to finish the complete research project by 25 October. The critical PP1 deliverables are functioning core paths, an implemented research mechanism, preliminary independent evaluation, credible progress records and an understandable demonstration.

Read this master plan first. Then execute [the phase plan](PP1_Phase_Execution_Plan.md) in dependency order and maintain [the evidence and readiness tracker](PP1_Evidence_Tracker.md). All dates are planning targets. The supplied rubric does not specify the presentation date, slide count or speaking duration.

### Source precedence and verified starting position

The four PDFs supplied in `Input Validation` define the component proposals. Their searchable text was rechecked on 27 September and matches the versions reviewed earlier. The PP1 mark sheet defines assessment expectations. The supplied BERT-APC paper informs the related-work distinction. Older unified input/harmony-critic drafts and exploratory novelty documents are background, not additional mandatory BPDR scope.

The source `Input Validation` folder contains the proposals, BERT-APC, the rubric and vocadito. Development now takes place only in the supplied repository at `experiments/bpdr_input`. A Phase 0 package, draft hand-off contracts and dataset-audit command have been created there; audio acceptance, acoustic analysis, repair and learned checkpoints remain unimplemented. Existing workspace design documents and the residual-event toy experiment do not establish a trained BPDR component. Code completed elsewhere can be credited only after inspecting its artifacts and results. This plan does not infer completion from the earlier September schedule.

The dataset inventory already established 40 mono 44.1 kHz WAVs, 40 F0 references, two note annotations per recording and metadata containing 29 singer IDs. Nineteen recordings exceed 20 seconds. These are file/header/metadata findings, not a complete decode, reference-quality or licensing audit.

## 2. How the marks work

The supplied one-page rubric gives PP1 a **15% contribution to the course**. The five category weights total 100% of the PP1 mark. Subcriterion weights below are calculated by multiplying their category weight by their internal share. [R, p. 1]

| Rubric area | PP1 weight | Evidence to prepare |
|---|---:|---|
| LO1: problem and proof of concept | **10%** | Show the failure/uncertainty the component addresses, then execute the proposed solution. |
| LO2: applied specialized knowledge and technologies | **25%** | Explain acoustic analysis, data design, learning objective, fair comparisons and actual implementation decisions. |
| LO3: implementation | **40%** | Demonstrate design, substantial completed work, best practices, functional requirements and risk controls. |
| LO4: communication | **15%** | Clear presentation, effective visual evidence, time management and confident evidence-based Q&A. |
| LO5: commercialization potential | **10%** | Supported user benefits, credible intended users, alternatives and feasible cost assumptions. |

| Subcriterion | Effective points out of 100 | Concrete BPDR evidence |
|---|---:|---|
| Problem definition | 3 | One real vocal example; distinguish unchanged audio with wrong boundary metadata from locally detuned audio. |
| Proof of concept | 7 | Actual correction predictions, rendered output, unchanged/failure paths and preliminary comparison. |
| Specialized knowledge pillars | 7.5 | Applied MIR/audio processing, supervised learning, controlled interventions and statistical evaluation. |
| Technology application and understanding | 17.5 | Explain feature units/alignment, frozen analyzer, Transformer, loss implementation, masking, controller and renderer using saved runs. |
| Design excellence | 8 | Implemented modular processing, immutable originals, explicit states, rollback and accepted-only hand-off. |
| Completion | 12 | Dated work-package progress with accepted deliverables and truthful corrective actions. |
| Standards/best practices | 8 | Source-separated splits, versioned contracts/configurations, reproducible runs and meaningful tests. |
| User/functional requirements | 8 | Proposal FR1-FR7 mapped to working paths and realistic acceptance evidence. |
| Risk mitigation | 4 | Data, rendering, scientific and integration risks with actual actions or an executable mitigation plan. |
| Communication and Q&A | 9 | Problem-to-mechanism-to-result narrative; explain limitations and comparisons. |
| Presentation skills | 6 | Readable plots, labelled audio, practiced transitions and timed rehearsal. |
| Commercial potential | 10 | Observed user/workflow evidence, achievable benefits and measured cost/latency; planned validation is labelled remaining work. |
| **Total** | **100** | Evidence should support each area; a screenshot alone is insufficient. |

The published bands are Excellent 75-100, Good 60-74, Average 40-59 and Below Average 0-39. A PP1 mark of 80 would contribute 12 percentage points to the course: `80 x 0.15`.

### The completion requirement

The Excellent completion descriptor refers to satisfactory work of approximately 50% **where applicable**, with no identifiable delay against the project plan. It does not provide an official percentage formula or mean that every feature must be half complete. The Good descriptor allows minor delays when corrective actions are identified and being executed.

Maintain the proposal's original work breakdown and a separately dated PP1 milestone plan. Do not silently remove overdue work or count slides, downloads, generated variants or unrelated older components as implemented research. If reporting a completion percentage, show its denominator, weighting method, accepted deliverables, exclusions and snapshot date. Get the baseline interpretation reviewed with the supervisor.

The rubric has **LO1**, not a separately defined Q1 requirement. If "Q1" means the gap/proof-of-concept area, use the LO1 evidence above. If it means Q1-ranked publications, this mark sheet does not impose that requirement. It does explicitly assess Q&A under communication.

## 3. Make the problem, gap and contribution distinguishable

| Layer | Proposal-faithful statement | What the examiner sees |
|---|---|---|
| User problem | A short solo recording may carry a useful melody despite small pitch errors or uncertain note transitions. Preparation must preserve the performed musical idea. | Original playback, measured pitch and one clear affected interval. |
| Technical problem | A false split or merge changes provisional note-boundary information without changing the audio. A repair model can react to that representation error. | The same waveform and checksum under nominal and corrupted boundary tracks. Show any measured change in the chosen baseline's correction. |
| Bounded research gap | The reviewed evidence does not establish whether minimal repair can resist plausible split/merge corruption while retaining a correct response to small local detuning and preserving acceptable input. | Paired boundary-only and local-detuning cases, evaluated together. |
| Proposed contribution | Time-aligned correction-response training combines boundary invariance, local counter-detuning and supervised preservation anchors. | Four aligned views, implemented losses and saved predictions from the trained model. |
| Evidence of advantage | The additional objective must improve held-out repair while meeting a predefined preservation margin at comparable acceptance coverage, relative to same-data augmentation-only and matched frame-only controls. | A preliminary source-separated result table including uncertainty, failures and coverage. |

BPDR's novelty is a proposed **training formulation and its measured effect**. The Transformer, continuous pitch correction, audio libraries and general consistency learning are supporting established methods. A functioning pipeline demonstrates implementation; an advantage needs fair comparative evidence. [S1, pp. 11-22]

### Related-work distinction that can be defended

The supplied BERT-APC paper already describes segmentation-dependent correction and tests displacement of existing boundaries. Its Section IV-H and Table VIII concern boundary shifts, including modest degradation at small shifts; they do not report the explicit insertion/deletion split/merge intervention used in this BPDR proposal. Present that specific distinction rather than saying prior methods ignore segmentation uncertainty. Its published scores concern different data and tasks and must not be compared directly with BPDR's pilot scores. [B, pp. 3, 12-13]

BERT-APC is related-work evidence, not an extra mandatory trained baseline: the submitted BPDR proposal retains three trainable conditions. Do not add older drafts' per-loss ablations or extra models to the PP1 critical path. [S1, pp. 17-18]

### What would fail to demonstrate the contribution

- A denoising/upload application without the BPDR training objective.
- A flat zero-correction curve described as robustness: it can be boundary-invariant while failing repair.
- Corrections taken directly from injected ground truth and presented as model predictions.
- Improvement against original audio only, without the same-data learned controls.
- Tiny-set fitting or hand-picked audio presented as independent accuracy evidence.
- A model that rejects nearly everything while reporting good error on the few remaining clips.
- A chord-quality score from the older harmony critic presented as proof of BPDR repair.

The purpose is to make the proposed contribution inspectable. If results are inconclusive or favor the simpler models, say so and show the next experiment. PP1 does not authorize inventing successful novelty results.

## 4. Define the PP1 build boundary

### Required target for a credible BPDR PP1 demonstration

This target is our planning interpretation of the rubric, not a university-prescribed feature list:

1. Original retention, decoding/integrity checks, a timing-preserving working copy and clear statuses.
2. Frozen performed-pitch analysis, aligned acoustic features and explicitly provisional boundaries.
3. Audited source manifest, source-group splits and verified four-view interventions.
4. Implemented augmentation-only, frame-only and BPDR models with saved checkpoints and a first fair controlled comparison.
5. Original and conservative-rule reference conditions, a bounded edit controller, separate rendering and rollback.
6. Original/prepared playback, correction/pitch plots and recorded changed intervals.
7. Accepted-only output package and a working fixed-extractor adapter or clearly identified contract stub.
8. Preliminary independent results, requirements/progress/risk evidence and a rehearsed presentation.

One equally budgeted run per learned model can establish initial PP1 execution if resources limit runs; predeclare this and disclose that seed stability is not established. Three seeds remain the proposal's preferred research setting where feasible. Do not give BPDR extra unreported search or training resources.

### Remaining research after PP1

Complete the main-study data collection and sample-size decision, broader natural-error evaluation, repeated runs, full blinded listening and downstream-benefit evaluation, long-term robustness and final reporting after the appropriate gates. Optional denoising is within the proposal but can remain a clearly tracked incomplete feature if core evidence needs priority.

Cloud deployment, subscriptions, accounts, extra formats, whistling, long recordings, multi-singer recovery, rhythm replacement and the older post-generation critic are outside the initial BPDR demonstration. A small local interface is sufficient. Record deferred proposal tasks as remaining work rather than declaring them completed.

## 5. Design the examiner's demonstration

Use one labelled evidence screen with an operational view and a research-comparison view. Prediction generation reads an actual checkpoint; cached results identify their run and cannot masquerade as a fresh live calculation.

| Case | Action | Evidence | Meaning |
|---|---|---|---|
| D1: acceptable reference | Process a reviewed usable vocal. | Original/output playback, status, duration and empty musical-edit log. | Useful input can be accepted without unnecessary tuning. |
| D2: boundary-only error | Keep the waveform fixed; select false split or merge metadata. | Boundary tracks and same-model correction curves; pairwise sensitivity. | Only the machine representation changed. This tests invariance. |
| D3: controlled local detuning | Process a verified detuned version of D1. | Known evaluation-only detuning, predicted correction and independently remeasured rendered pitch. | This tests whether invariance coexists with actual corrective response. |
| D4: expression/no-change | Process a reviewed vibrato/slide or preserved transposition example. | Correction magnitude, edit eligibility and explanation. | Expression is not automatically an error. |
| D5: non-accepted input | Use a corrupt file or unusable recording. | Reasoned status and no accepted-audio reference; downstream stops. | Failure handling is part of the functioning solution. |
| D6: traceable hand-off | Send an accepted package to the extractor adapter. | Timed output, common ID and recorded component versions. | Shows integration. A stub proves only the contract. |

Live showcase examples explain the mechanism; the result table covers the declared independent evaluation collection. Include successes, a failure/uncertain case and exclusions. Do not select all evaluation cases after seeing favorable outputs.

### Four-view evidence

| View | Waveform | Boundary track | Training target |
|---|---|---|---|
| A | Reviewed acceptable source | Nominal | Zero correction |
| B | Identical to A | Corrupted split/merge/jitter | Zero correction |
| C | Verified local detuning of A | Nominal under the controlled pairing rule | Negative injected detuning |
| D | Identical to C | Corresponding corrupted track | Same target as C |

For controlled factorial testing, declare how nominal/corrupted tracks are held across paired waveforms. Recompute spectral/pitch features after waveform rendering. Test naturally recomputed analyzer boundaries separately. Detuning targets and reference curves are evaluation/training supervision, not runtime information about an unknown user's intention.

Use two musically experienced reviewers for acceptable zero-change anchors, logging disagreements and reasons. Flag/guard ambiguous transitions. Define a realized-shift tolerance before generating the training corpus and reject interventions outside it: the negative injected curve is trustworthy supervision only after that audit. Supplement the frozen analyzer with appropriate reference/manual or alternative measurement checks; agreement with the same tracker alone does not independently certify rendering accuracy.

Explain the losses in ordinary language first: boundary changes alone should not change the repair; known local detuning should produce the opposite corrective change; already acceptable material should be preserved. Keep the precise proposal equations in technical backup material. [S1, pp. 21-22]

## 6. Minimum measurement specification

| Measurement | PP1 interpretation |
|---|---|
| Correction error in cents | Primary outcome on reliable controlled detuned regions; compare BPDR against augmentation-only and frame-only. Report source/singer denominators. |
| Boundary sensitivity | Difference between correction curves for identical audio with changed boundary metadata. Interpret jointly with repair error. |
| Preservation | Unnecessary correction on no-change examples and outside detuning support; expression/transition examples; report listener evidence only when actually collected. |
| Realized repair | Difference between rendered audio pitch and the reference, with identity-render controls. |
| Acceptance/edit coverage | Count all four statuses and distinguish audio acceptance from fraction actually edited. Report harms and refusal/failure counts. |
| Extraction | Same fixed extractor, same reference notes, before/after preparation. A stub cannot establish extraction benefit. |
| Latency | Actual median and 95th percentile on named hardware; the proposal's 30-second target is not a measured result. |

Use a reliable-frame mask fixed independently of the compared models; exclude empty-mask cases with logged reasons. Report detuned and preservation regions separately. Normalize/fill features using training data only; keep singer groups and their augmented descendants in one partition. Do not treat frames or variants as new participants.

Estimate paired uncertainty by resampling independent singer groups (or source recordings when singer identity is unavailable), keeping paired methods/views together. Report seed variation separately; one run cannot establish training stability. Small PP1 holdout intervals are descriptive pilot evidence, not precise population estimates.

Report raw model predictions before the controller and delivered audio afterward. Apply comparable policies and rendering to correction methods. Original audio receives necessary conversion only. Compare at attainable matched acceptance coverage; disclose if matching is not possible.

Choose thresholds, loss weights, model selection and metric definitions using development data. Keep an untouched PP1 evaluation subset distinct from any final-research test set if the PP1 observations will guide later changes. A corrected model needs fresh independent evaluation for confirmatory claims. Never relabel a previously examined subset as unseen.

The proposal's +/-10-30-cent interventions may remain within conventional note-matching tolerances. A constant note F1 does not by itself mean small-pitch repair failed. Cents error and rendering/preservation checks are needed; final-extractor benefits are a separate hypothesis.

Do not promise an arbitrary accuracy percentage now. Decide the meaningful preservation margin and practical effect target before reading held-out results. Present observed effect size and uncertainty, even if the outcome is inconclusive.

## 7. How the colleagues' components align

For team readiness, prepare individual implementation evidence and a shared contract. The rubric header identifies a student and group; it does not establish identical marks for all members. Each member should explain and demonstrate their own contribution.

| Owner | Proposal contribution | PP1 research evidence target | BPDR dependency |
|---|---|---|---|
| You: IT23141506 | Minimal repair learned to separate boundary-only changes from local detuning. | Paired audio experiment, three-model comparison, preservation and accepted-only delivery. | Independent of finished chord models. |
| Perera E. V.: IT23439078 | Shared CNN/Transformer melody understanding and unified musical representation. | Actual audio-to-timed-note output, labelled provenance for beat/key/phrase fields, annotated baseline results and first proposed-model training evidence. Joint-learning benefit remains a hypothesis. | Immediate integration partner; freeze input/timeline/adapter. |
| Perera P. D.: IT23148086 | Test explicit key-relative root/family planning beyond normalization and auxiliary supervision. | Audited mapping, actual plan inference, transposition/unsupported-label tests and preliminary fair baseline comparison. Full coherence advantage needs later shared evaluation. | Consumes extracted melody, not BPDR raw features. |
| Jayakody J. A. S. R.: IT23157132 | Contextual basic-to-rich refinement, including candidate alternatives. | Audited chord pairs, functioning learned refinement/baseline, fixed event timing and plan-adherence checks. Multiple candidates are not proof of multiple valid answers. | Consumes aligned melody and planner output. |

### Decisions to settle in Phase 0

- **One ownership diagram:** the shared architecture figures show BPDR -> understanding -> planning -> refinement. Melody-understanding prose/WBS still assigns the fourth member to accompaniment generation. Resolve that stale contradiction; assign simple playback/export explicitly. [S2 pp. 38-40, 49; S3 p. 23; S4 p. 26]
- **One timeline:** BPDR preserves duration/rests; understanding proposes silence trimming. Preserve the accepted waveform origin and map any internal analysis windows back to it. [S1 pp. 26-29; S2 pp. 30, 42]
- **One musical contract:** understanding owns seconds-to-beats/key/metre estimates. Planner root/family labels are not full harmonic-function labels. Refinement preserves root, family and timing in the agreed PP1 path; substitutions are separate future scope. [S3 pp. 19-24; S4 pp. 22-29]
- **One label provenance policy:** genre-labelled MP3s do not automatically provide verified timed melody/rich-chord targets. Mark automatic extractions as pseudo-labels and audit them. [S3 p. 16; S4 pp. 20-21]
- **One bounded demo excerpt:** BPDR supports up to 20 seconds while planner examples span 4-16 bars. Choose compatible excerpts without silent truncation.

If a teammate's model is unavailable, use a fixed existing baseline adapter and label it. Fixture-backed arrows show integration mechanics, not completed team model performance. Do not wait for the full chain to begin BPDR experiments.

## 8. Requirements, progress, risks and business evidence

Map the proposal's FR1-FR7 to capture/upload, original retention, assess/preserve/prepare, bounded edits, playback/feedback, checked export and evaluation plots. For each requirement record the implemented path, an acceptance check, evidence location and remaining limitations. An empty/corrupt-input test is necessary reliability evidence; tests that merely confirm a constant output are not accuracy evidence. [S1 pp. 28-29]

Maintain W1-W8 from proposal Table 11: literature/gap, design/protocol, data/interventions, models/experiments, prototype/preparation, integration, evaluation and reporting. Its planning weights total 640 hours. Do not mistake them for actual hours spent. A team-defined progress indicator can use agreed milestone weights within each package, but must remain distinct from rubric marks. Exact progress is currently unverified.

| Risk | Required action before PP1 |
|---|---|
| Small or unreliable reference set | Audit available vocadito material, review anchors, report independent singer/source counts and pilot limits. Participant collection follows the proposal's approval/consent conditions. |
| Renderer/measurement cannot resolve small shifts | Identity controls and measured intervention checks before generating training labels. Improve the chain or narrow the experiment transparently. |
| Baseline already ignores boundaries / BPDR adds no gain | Test baseline sensitivity; retain frame-only and augmentation-only controls; report a negative result rather than hiding it. |
| Compute/dataset delay | Benchmark one small run early; use equal predeclared reduced budgets; keep the original milestone and record corrective action. |
| Rejected input leaks downstream | Non-accepted statuses have no accepted path; adapter must enforce this and tests must exercise the stop. |
| Team integration unavailable | Freeze contract fixtures and a temporary extractor; separate integration status from research results. |
| Demo interruption | Prepare an offline replay of the actual saved run, audio and metrics with run IDs; rehearse fallback disclosure. |

Commercial evidence can remain small and realistic: identify a specific target workflow, document what user problem was actually observed, show how the prototype helps, compare appropriate existing approaches and estimate processing cost from measured time/hardware. If no user pilot has occurred, show the planned protocol and label demand/pricing as hypotheses. Do not invent interviews, adoption, willingness to pay, revenue or ownership rights. A business canvas alone is weak evidence of commercial potential. [R, LO5]

## 9. Presentation narrative and examiner questions

Use the actual allocated presentation slot once provided. Suggested time proportions: problem/gap 15%, method and applied technology 20%, live evidence/results 35%, completion/requirements/risks 20%, benefits/next work 10%. This is a planning choice, not the rubric's mark distribution or an official time rule.

Presentation sequence:

1. State the user problem with one audible/visible example.
2. Show prior capability and the narrower unanswered question.
3. Locate BPDR in the team architecture and identify your responsibility.
4. Explain A/B/C/D and the training-response objective.
5. Demonstrate unchanged input, boundary-only change, local detuning and a controlled failure.
6. Show comparative preliminary results and preservation/coverage, including limitations.
7. Show implemented architecture, FR acceptance and actual completed work against the plan.
8. Explain mitigations, achievable user benefit and next research milestones.

Be ready to answer: Why not remove boundaries? What prevents zero correction? How are targets known? Why is this different from BERT-APC? How were singers split? Which curves are measurements versus predictions? Was the rendered shift correct? What if the singer intended that note? Why are three models needed? How much is implemented? What fails? Which teammate outputs are learned, baseline or fixtures? Why might note F1 remain unchanged?

Answers must point to a source, implemented operation or saved run. "We use a Transformer" is not a sufficient novelty or technology-depth answer. Acknowledge when intended melody is ambiguous, when public-data evidence is only a pilot, and when the comparative advantage is not established.

## 10. PP1 exit gate

By 25 October, aim to hand over a working local BPDR prototype, three saved learned conditions with an initial fair comparison, reviewed paired examples, rendered audio verification, accepted-only contract evidence, a requirements/progress/risk register and a rehearsed presentation backed by actual artifacts.

The core gate is not met if only a polished interface or prewritten plots exist. If required evidence is incomplete, keep the status explicit, show executed corrective actions and narrow the claim. The examiner should be able to see **what was built, why it addresses the problem, what was tested, and what is still unknown**.

## Source register

PDF page numbers below are positions within the supplied files. Literature/market statements in the proposals are not all re-verified by this planning task.

- **R:** [PP1 Assessment Criteria (1).pdf](<C:/Users/lnipu/Projects/Music Rp/Input Validation/PP1 Assessment Criteria (1).pdf>), p. 1, all five learning-outcome sections and grade bands. The complete page was extracted and visually inspected.
- **S1:** [IT23141506_J26_DS_339.pdf](<C:/Users/lnipu/Projects/Music Rp/Input Validation/IT23141506_J26_DS_339.pdf>), gap p. 14; objectives p. 15; controlled data/comparisons pp. 16-22; architecture pp. 24-27; FR/NFR pp. 28-29; WBS p. 35; failure tests p. 40.
- **S2:** [IT23439078_J26_DS_339.pdf](<C:/Users/lnipu/Projects/Music Rp/Input Validation/IT23439078_J26_DS_339.pdf>), gap/objectives pp. 24-28; data pp. 29-30; evaluation pp. 31-36; architecture/ownership pp. 38-40; requirements pp. 41-43; schedule p. 49.
- **S3:** [IT23148086_J26_DS_339.pdf](<C:/Users/lnipu/Projects/Music Rp/Input Validation/IT23148086_J26_DS_339.pdf>), gap/question pp. 14-15; controls pp. 16-18; representation/contracts pp. 19-24; requirements p. 25; work packages p. 30.
- **S4:** [IT23157132_J26_DS_339.pdf](<C:/Users/lnipu/Projects/Music Rp/Input Validation/IT23157132_J26_DS_339.pdf>), gap/objectives pp. 17-19; data/model pp. 20-23; experiments pp. 24-25; boundaries/requirements pp. 28-30; experimental appendix pp. 38-40.
- **B:** [BERT-APC.pdf](<C:/Users/lnipu/Projects/Music Rp/Input Validation/BERT-APC.pdf>), architecture p. 3; segmentation experiment Section IV-H p. 12 and Table VIII p. 13. These complete relevant pages were extracted and visually inspected. The downloaded PDF's generic journal-template header is not treated as publication metadata.
- **Previous planning baseline:** [BPDR_Development_Plan.md](<C:/Users/lnipu/Projects/Music Rp/Validation Framework/output/BPDR_Development_Plan.md>). Its September dates remain a historical plan, not proof of completed work. This PP1 package supplies the rebased assessment milestone plan.
