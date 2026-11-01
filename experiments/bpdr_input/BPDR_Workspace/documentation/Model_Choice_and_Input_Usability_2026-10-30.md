# Model choice, repair bottlenecks and input usability

## Updated objective

The owner asks whether difficulty comes from choosing a Transformer, requests
comparison with other models and web research, and prioritizes more accurate
delivered output. Additional input-preparation features must respect other
members' ownership. This extends the model-improvement work; it does not permit
lowering verification limits or claiming musical correctness from raw MAE.

## What current evidence establishes

Transformer selection has **not been proved wrong or best**. Many variants of
one family cannot answer that question. The current reference gives11.901c raw
MAE on108 scoreable cases but23 verified repairs from200 attempts and20.342c
remaining delivered error. V25 improves raw MAE slightly while losing delivery.
These are controlled-intervention recovery results, not intended-note accuracy.

Three difficulties have evidence:

1. **Independent data are limited.** The base evaluation uses50 originals from
   10 attempted singers, only8 with scoreable interventions. The broader pool
   contains373 unique valid variants from the SAME originals/people. Frames,
   augmentations and seeds do not add independent singers. Real measured
   negative-region supervision is sparse:8 candidates in the V21 pool.
2. **Prediction and usable repair differ.** The reference attempts30 nonidentity
   renders on108 shifted cases, verifies23 and rolls back7. Its remaining78
   shifted cases have no nonidentity edit. Improving an average prediction can
   therefore fail to produce long, reliable, permitted correction supports.
   These are stage counts; their precise causes need the saved diagnostic.
3. **The target is constrained.** Original controlled recordings supply a known
   recovery reference; they do not supply the user's intended melody on natural
   novice singing. Input-only relative pitch evidence cannot generally resolve
   musical intention without additional supervision or user review.

The stage bottleneck replay has completed in
`tools/diagnose_current_repair_bottlenecks.py`, producing V26's
`reference_bottlenecks.json`. It reuses frozen reference predictions and checks
that recomputed proposal intervals agree with saved decisions. Candidate reasons
are nonexclusive; counts must not be summed as independent cases. It finds99/108
recordings with at least one long unprotected eligible input region,94 with a
prediction-support candidate and30 permitted nonidentity plans. The eligible
region can be elsewhere than the required correction;99 is not a repairability
ceiling. Case-level candidate rejection counts are24 no activation,21 too short,
29 protected and12 weak step-shape evidence;14 cases have no candidates. These
categories overlap. This diagnostic cannot establish which architecture would
do better, but demonstrates why model MAE alone is an inadequate selection rule.

## Research checked against the component scope

| Primary source | Relevant evidence | Decision for BPDR |
|---|---|---|
| [Grinsztajn et al., NeurIPS2022](https://arxiv.org/abs/2207.08815) | Trees are strong on the paper's medium-size tabular benchmark; neural capacity alone does not guarantee better results. | Include a fresh context-feature ExtraTrees baseline. BPDR is sequential, so this paper does not establish a tree winner here. |
| [Bai et al.,2018](https://arxiv.org/abs/1803.01271) | Temporal convolutions compete strongly with recurrent networks on the evaluated sequence tasks. | Test a compact temporal convolution alongside Transformer and bidirectional GRU. This is rationale for an experiment, not a BPDR result. |
| [Wager et al., Deep Autotuner, ICASSP2020](https://arxiv.org/abs/2002.05511) | A convolutional/GRU pitch-correction model uses vocal/accompaniment spectrogram relationships and4,702 selected performances, learning correction and intentional-variation preservation. | Include a recurrent alternative and strengthen preservation examples. Its accompaniment-dependent result is not transferable to our unaccompanied input or current data size. |
| [Riou et al., PESTO2025](https://arxiv.org/abs/2508.01488) | Self-supervised equivariance supports monophonic pitch estimation with a lightweight model. | Better acoustic measurements may help; pitch estimation still does not identify an intended correction. Existing PESTO already uses weighted decoding, so no unverified argmax-fix claim. |
| [Dong et al., October2026 preprint](https://arxiv.org/abs/2610.11524) | Frozen MusicBERT with singing adapters separates editing necessity from signed changes, using symbolic notes and a tonal prior. The paper was submitted to ICASSP2027. | Useful distinction between detection and correction, already represented by BPDR activity/amplitude heads. Do not import note transcription, key estimation, tonal reranking or semitone edits: these exceed this component's scope and differ from bounded local acoustic repair. Its reported note accuracy is not comparable to our metric. |

Inference from these sources: a compact Transformer remains plausible, but
model/data/task matching matters more than model reputation. No reviewed paper
proves the winner for this project. The current frozen-backbone adapter tests
whether retaining learned temporal shape helps; a negative result should lead
to a controlled family comparison rather than further readout sweeps.

## Next controlled comparison

| Condition | Question | Status |
|---|---|---|
| Frozen reference Transformer | Reproducible existing benchmark | Completed, retained |
| Dense residual Transformer | Can learned corrections improve the existing function without changing the backbone? | V26 completed4 head fits; raw gate passes but21/200 repairs and21.394c delivered error lose to reference; not selected |
| Fresh ExtraTrees on input-only context | Is a simpler small-data regressor more effective under the same holdouts? | V28 finishes4 fits; raw gate fails,20/200 repairs and21.217c delivered error lose to reference; independent216-view/20-WAV replay passes |
| Compact bidirectional GRU | Is an ordered recurrent representation a better fit? | V27 finishes4 fits; raw gate fails,18/200 repairs and21.626c delivered error lose to reference |
| Compact temporal convolution | Does local temporal structure generalize better? | V27 finishes4 fits; raw gate passes but19/200 repairs and21.281c delivered error lose to reference |

The V27/V28 declarations freeze one shared training dataset,
input-only context definition, singer/source folds, sample schedule, supervised
targets, original controls and selection rule. Exclude IDs and provisional
boundary metadata from all features. Use the same neural optimization budget
and loss for Transformer/GRU/TCN, recording parameter counts and time. Treat
ExtraTrees' different fitting objective/budget explicitly; it is a practical
classical baseline, not an identical-gradient experiment. Fit it afresh: the
active demo's historical model is not automatically an independent comparator.
V27 substitutes only encoder constructor/checkpoint handling into the existing
V12 training function; every actual fold sampler digest must match its reference.
V28 materializes those exact43200 training rows/fold and fits256 depth-limited
ExtraTrees using weighted squared error. All models receive the same acoustic
context and presence mask; the tree flattens them to459 features. The GRU and
TCN input/output, gradient, boundary-metadata and reload checks pass, as do the
tree feature-metadata, bounded-output, identity and reload check. V27 retains
the reference after actual audio evaluation; independent432-view/37-WAV replay passes.
V28 also retains the reference after actual delivery and independent replay.
The [completed feasibility diagnosis](Repair_Feasibility_Diagnosis_2026-10-30.md)
now supersedes the generic architecture-comparison next step: investigate
acoustic measurement and spatial localization before further model sweeps.

Compare raw MAE/tolerances, original prediction magnitude, false-edit rate,
per-singer/direction performance, permitted support, encoded-WAV repair benefit,
all-attempt coverage, preservation and processing time. Retain all200 attempts,
108 scoreable and92 exclusions. Rank on actual delivered error after the same
gates; unscoreable singers remain unassessed. No early family winner from raw
accuracy alone and no held-out audio used for per-case routing.

If all families fail at the same input-eligibility or audio-verification stage,
that is evidence to investigate data/support/measurement limitations before
scaling another model. It is not permission to loosen gates. A practical winner
still requires a locked pipeline, known development, fresh confirmation and
independent musical review before activation. Preserve the original planned
confirmation representation; do not quietly shrink the cohort to available data.

## Component-safe usability additions

Existing app behavior already includes technical feedback, listening review
and accepted-package export. Do not present these as new features. Selective
protection/review is already in the master plan, and follows usable main repair.

New extensions worth prioritizing after the model work:

- **Time-specific recording coach:** extend existing generic feedback with
  highlighted clipping, unstable voicing or insufficient stable support and
  an actionable rerecord instruction. Validate each reason against measured
  acoustics; do not tell users which musical note to sing.
- **Multiple-take preparation:** compare short user takes for technical
  suitability, preserve them separately and let the user choose after listening.
  Assess clipping, usable voiced duration and processing eligibility; do not
  infer whether different takes contain the same intended melody.
- **Evidence-based readiness preview:** show which time regions can be safely
  handed on unchanged, considered for a verified repair, or rerecorded, with
  transparent reasons. Do not display an uncalibrated model probability as an
  accuracy percentage. Calibrate any future risk/coverage display on independent
  examples before calling it confidence.

All additions belong to audio input preparation, sample/time-based regions and
accepted-waveform provenance. Final notes/durations, tempo/key/mode, genre,
chords and harmony remain other members' responsibility. No sibling components,
shared `pp1_demo`, active exports or model selection change in this investigation.
