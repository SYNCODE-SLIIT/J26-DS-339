# Paired short-note Transformer: measured progress, qualification failed

2 November 2026. Target: 5–15-second melodic hums with short notes. MOD-03 remains
ACTIVE. Neither new candidate is qualified or activated. This record supersedes
V41/V42 as the latest experiment; their results and the delivered PDF remain
valid historical snapshots. No shared demo, teammate feature or default model was changed.

## What changed and what the checks establish

V43 tests an input-only short-event cue. Six contract tests pass, but its real-voice
correction error remains 25 cents. It detects other natural events without
identifying the controlled target reliably. It is not used by the V46 models.
Context diagnostics show that missing anchors explain some cases, not every
failure: nonlinear natural pitch movement remains difficult even in a diagnostic
that knows the true support. That diagnostic is never available at inference.

V44 implements a separate **experimental measurement protocol**: 20 ms ramp,
70 ms guard and three common native audit observations instead of ten. It adds
a waveform Praat raw-cross-correlation pitch check alongside PESTO/YIN, retaining
direction, error, coverage and exact-outside-samples checks. The three observations
overlap; they are not independent samples. Praat CC shares the renderer's library
and uses temporal path finding, so it is corroboration, not independent pitch truth.
The historical default ten-observation protocol is unchanged. These results must
not be described as improvements under unchanged historical audit rules.

Of 336 planned training-source perturbations, 128 render and 115 pass the new
three-tracker protocol. By duration: 36 at 0.24 s, 36 at 0.28 s, 23 at 0.36 s and
20 at 0.50 s. All 512 wrong-request controls are refused; 432 have adequate common
pitch coverage, while 80 displaced-support controls lack coverage. All 128
outside-support sample tamper controls are refused before acoustic measurement.
This finite, correlated control set does not establish a general false-accept rate.
Independent replay reproduces the audits, exclusions and controls.

V45 stages those 115 physical pairs from eight singer groups, plus 19 valid
originals from 21 attempted sources. All 336 attempts remain counted. It retains
positive-plateau and known-zero masks separately; those masks are supervision and
evaluation only. Independent replay checks 1,035 staged arrays. No nontraining
Vocadito singer or annotation was opened. One group, S25, has only 2/16 scoreable
attempts: this pack cannot establish the historical 25% per-group all-attempt
coverage requirement even if both succeed.

## Training and raw prediction results

V46 retains the same 21,666-parameter, genuine sequence Transformer and four
source-held-out folds. Eight fits compare 1,200 paired physical updates with
400 virtual pretraining plus 800 physical updates, seed 339. The final 800 physical
sampler updates match between arms. The loss includes paired response and
unchanged-expression preservation; this is not yet the research comparison of full
BPDR, augmentation-only and frame-only objectives at matched acceptance coverage.

Virtual examples change pitch-feature trajectories, adding shared expressive
movement to original/corrupted pairs and a short pulse only to the corrupted half.
They are not additional recordings, people or validated waveform interventions.
Frontend behavior is not simulated. Independent replay reloads eight checkpoints,
reinfers 268 prediction arrays, recreates four physical sampler traces and exactly
recreates 19,200 virtual examples. Three virtual-example contract tests pass.

| Arm | Held-singer macro correction error | Original-prediction macro error | All-group raw gate |
|---|---:|---:|---|
| Paired physical | 21.201 cents | 0.673 cents | FAIL |
| Virtual then paired | 22.037 cents | 0.711 cents | FAIL |

The physical arm responds on some voices, but S11 responds in the wrong direction
and several other groups have near-zero response. A pooled improvement cannot
hide these failures. The new physical data and changed audit make direct numerical
ranking against V41's smaller pack inappropriate.

## Delivered-WAV results

V47's policy was declared before final V46 raw results. Each arm evaluates
115 scoreable corrupted cases and 19 originals; 221/336 attempted corruptions are
construction exclusions. Proposals use input audio/features only. Reference masks
and original performances are used after delivery to measure benefit and harm.

An output counts as beneficial only if the encoded audit passes, the injected
plateau improves and the known-zero area does not worsen by more than 1e-6 cents.
Any plateau worsening or known-zero worsening counts as harm. This stronger
preservation definition prevents target-only gains from hiding collateral edits.

| Arm | Beneficial outputs / 115 scoreable | Harmful outputs / 115 scoreable | Target-only improvements | Originals changed / 19 | Remaining singer-macro target error |
|---|---:|---:|---:|---:|---:|
| Paired physical | 4 | 32 | 25 | 2 | 22.740 cents |
| Virtual then paired | 2 | 36 | 21 | 4 | 24.098 cents |

Input target error is 24.873 cents. All reported harms include known-zero worsening;
target gains and harms can overlap. Known-zero error increases range from
0.000588 to 2.037 cents (median 1.340) for physical and 0.011102 to 2.222 cents
(median 1.244) for virtual then paired. These are strict numerical preservation
failures, not a claim that every change is audible. Neither arm meets preservation or usable
coverage. The original controls changed are vocadito37/39 for physical, and
vocadito14/29/37/39 for virtual then paired. Independent replay completes:
268 proposals,83 candidate audits,80 encoded audits,230 positive/known-zero
reference records and38 original-control checks. Both arm summaries reproduce.
The receipt is `2026-11-02_v47_paired_delivery/integrity_receipt.json`.

## Next work and evidence required

Stop treating another small optimizer or architecture sweep as the next default.
These experiments do not prove that Transformers are intrinsically unsuitable;
they show that current supervision, transfer and selective preservation are
insufficient. Earlier matched model families also did not qualify.

1. Audit real humming data before audio inspection. The
   [HumTrans preparation plan](HumTrans_Data_Plan_2026-11-02.md) pins provenance,
   separates singers and compositions, and keeps confirmation closed.
2. Build broader training-only controlled local interventions and unchanged
   transition/glide/vibrato examples from actual hums. Count failed attempts and
   every duration/person group; do not quietly lengthen notes to fit the protocol.
3. Freeze a model and operating point, then check delivered improvements,
   known-zero preservation, original identity and per-group all-attempt coverage
   on development. Open confirmation only after a qualified development result.
4. Use the [review and collection plan](Review_and_Collection_Plan_2026-11-01.md)
   to establish intended-idea and listening evidence. No reviewers, consent,
   collection, natural-intention accuracy or research validation is claimed.

## Inspectable experiment records

All paths below are under `results/transformer_usability/` in this workspace:

- `2026-11-02_v43_short_event_evidence/`: failed cue/context diagnostics.
- `2026-11-02_v44_short_audit_calibration/`: declaration, controls, replay receipt
  and `negative_control_coverage.json`.
- `2026-11-02_v45_native_short_pairs/`: staged physical pairs and replay receipt.
- `2026-11-02_v46_paired_short_models/`: checkpoints, raw results and replay receipt.
- `2026-11-02_v47_paired_delivery/`: declared policy, output WAVs, target/zero
  scores and completed independent replay receipt.

The legacy checkpoint/source lock remains authoritative. Additive experimental
modules do not activate a model or establish the full research hypothesis.
