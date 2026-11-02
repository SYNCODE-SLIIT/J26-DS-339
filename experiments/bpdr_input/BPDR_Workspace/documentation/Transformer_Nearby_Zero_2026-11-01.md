# Transformer nearby unchanged-region sampling study

Declared 1 November 2026. Status: four held-out fits and actual-WAV evaluation complete; candidate rejected.
Independent replay passes all four sampling streams, 216 prediction views,
20 encoded WAVs, 50 original controls and six protected artifacts. All 35 focused regression tests pass, including
three new trusted-zero sampler checks. Tests are not an accuracy qualification.

## Question and fixed intervention

Can the same Transformer learn more spatially precise corrections when a fixed
share of existing real-audio zero-target draws is concentrated near known
positive targets? The authenticated sampler audit finds such queries occupy
2.22–2.82% of all training draw rows within43 frames. This is a hypothesis about
localization, not evidence that the sampling change will improve delivered audio.

For each legacy real-C draw with a trusted zero target, a separate random stream
(seed339+32532) replaces its frame with probability0.5 by a uniformly sampled
trusted zero within43 frames of any known positive frame in the same recording.
An empty pool retains the original index. The paired original uses the same
replacement index. All masks and target values are retained. Never fill unknown
transition/ramp labels. Feature-only augmentation, positives, source selection
and separate original-anchor draws retain their exact original schedule.

Use four singer/source-held-out fits,1200 updates and default seed339. Reuse the
four exactly reproduced V31 MSE controls authenticated against V12. Keep the
51x8,32-wide,two-layer Transformer, losses and weights, AdamW settings, decoder,
median11/Praat renderer and every permission/output verification unchanged.
No new sources, label-derived runtime support or extra training examples.

## Evidence required

Save every legacy/actual query index, key, target and weight. The legacy stream
must match V12's actual SHA256. An independent reconstruction from the original
sampler must reproduce all changed indices and preserve every positive,
augmentation and original-anchor index. Verify all held-out prediction arrays,
encoded WAVs, source/singer separation and protected artifacts.

Selection requires all-singer raw gates, all50 originals unchanged, and strictly
lower singer-macro delivered remaining error than reference20.3421417351 cents.
Reference verifies23 repairs among200 attempted constructions,108 scoreable and
92 excluded. Repair count alone cannot select a model. Every candidate output
retains max30 cents,min0.86-second support,max25% edit coverage,uncertainty and
protection vetoes,unchanged samples outside proposed supports,and dual-track
post-render/encoded-WAV checks. No threshold relaxation or support trimming.

These folds have been reused across exploratory engineering studies. Their
results support comparison during model development, not an independent final
accuracy estimate after repeated model selection. Keep all negative studies and
retain untouched confirmation for a separately locked pipeline.

A CV win would still require full-fit lineage, known development and untouched
confirmation before activation. Musical reviewers and new-participant collection
remain separate proposed evidence, not completed validation. Preserve selected
exports, notebook04, shared demo and sibling features.

## Execution

Use `train_nearby_zero_transformer.py declare`, then `cv`, then `delivery`,
then `verify_nearby_zero_study.py` in the existing environment. Retain any failure
and its actual completed work; do not restart live or completed fits blindly.
Results root: `results/transformer_usability/2026-11-01_v32_nearby_zero`.

## Observed result

| Measurement | Reference | Nearby-zero arm |
|---|---:|---:|
| Raw correction MAE, cents | 11.901 | 12.027 |
| Original prediction magnitude, cents | 0.789 | 0.554 |
| Outside-target prediction magnitude, cents | 0.824 | 0.726 |
| Raw all-singer prediction gate | Pass | Pass |
| Verified delivered repairs / all attempts | 23/200 | 20/200 |
| Singer-macro delivered remaining error, cents | 20.342 | 21.108 |
| Original recordings delivered unchanged | 50/50 | 50/50 |

The original200/108/92 attempted/scoreable/excluded denominator is retained.
Both arms have30 nonidentity candidate outcomes; the reference has7 rollbacks
and this arm10. Nineteen verified cases are shared, one is new and four are lost.
The smaller outside-target requests do not translate into better delivered
output. This fixed sampling change is not selected; the failure does not prove
that all localization methods or Transformers are unsuitable. Do not combine
case-wise winners using reference answers or tune a radius on these outcomes.

All four legacy sampler digests match their reference receipts. Actual changed
query rows are3096,3074,3076,3078 out of43200 rows/fold; paired A/C changes count
as two rows, not independent recordings or people. Independent full-trace reconstruction verifies the positive, source, feature
and original-anchor schedules. `paired_comparison.json` binds the descriptive case comparison
to both accuracy-file hashes. Full refits and known-development/confirmation
access do not follow a failed training-side selection. Model usability remains
unproved; review and consented collection remain pending real execution.

## Failure-stage diagnosis and next action

All four reference-only successes become `COARSE_DIRECTION` rollbacks in this
arm: case indices16,78,118,162. Each has one opposing coarse frame and zero
wrong-direction fine frames. The one new success (case66) resolves the same
one-frame coarse-direction veto found in the reference. Across all rollbacks,
reference has6 coarse-direction and1 coarse-median-error reason; this arm has10
coarse-direction and2 coarse-median-error reasons (reasons may overlap).

This localizes the observed outcome change to post-render measurement checks;
it does not establish that a tracker is wrong or that rejected audio is safe.
The next mechanism investigation should reconstruct these disagreements on the
whole declared training-side candidate set, recording before/after fine/coarse
tracks, requested shifts and timestamps. Do not bypass either auditor, omit
opposing frames, retune thresholds, trim supports, select case-wise winners or
interpret within-request fine accuracy as preservation of intended melody.
Another loss/architecture sweep is not justified by this result alone.
