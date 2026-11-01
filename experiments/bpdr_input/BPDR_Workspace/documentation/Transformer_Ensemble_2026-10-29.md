# Frozen three-Transformer ensemble

Status: prediction and actual-WAV evaluations completed; ensemble rejected.
MOD-03 remains incomplete. No neural fitting, full refits or activation.

The fixed members are V12 reference,V16 support-supervised and V18
broader-audio models. All12 member checkpoints inherit their original four
singer/source folds. Hashes bind weights, saved predictions, receipts and
training declarations. Every member is checked against its held-out sources;
one actual held-out probe per member/fold reproduces cached predictions.

The primary prediction is the framewise median. Min/max member envelopes
remain uncertainty evidence. Primary support is generated with the unchanged
held-voice policy and median11 conditioning, then final permission checks
the FULL high-minus-low range against the unchanged5-cent limit. Original
controls use the same member envelopes, not copied primary-only predictions.
No reference-based routing, per-case member selection or support padding occurs.

These are member envelopes, not boundary perturbations or calibrated confidence.
The report renames the legacy disagreement metric accordingly. The original
member models still omit provisional-boundary metadata by construction;
no learned boundary-objective superiority is established.

| Metric | Frozen reference | Fixed ensemble |
|---|---:|---:|
| Raw correction MAE,cents | 11.901 | 11.536 |
| Raw original mean,cents | 0.789 | 0.920 |
| Prediction gate | Pass | Pass |
| Verified actual-WAV improvements /200 | 23 | 2 |
| Delivered after MAE,cents | 20.342 | 24.615 |
| Original WAV identity | 50/50 | 50/50 |

The extra disagreement evidence withholds too much useful repair. Do not loosen
the5-cent limit or remove dispersion after seeing this outcome. The saved
selection retains the reference, with the ensemble inactive. All200 attempts,
108 scoreable interventions and92 construction exclusions stay visible.
These are training-side engineering results, not new independent evaluation.

Three tests pass: median/envelope permutation invariance and identity; invalid
or over-limit member rejection without clipping; full-range veto and exact
primary support preservation. Full joint/curve/runtime regression checks are
separate from model usability. The active export and protected files remain.

- [Prospective declaration](../results/transformer_usability/2026-10-29_v19_ensemble/declaration.json)
- [Prediction result](../results/transformer_usability/2026-10-29_v19_ensemble/cross_validation.json)
- [Actual-WAV evidence](../results/transformer_usability/2026-10-29_v19_ensemble/training/median11/evidence.json)
- [Locked selection](../results/transformer_usability/2026-10-29_v19_ensemble/locked_selection.json)
- [Ensemble and lineage evaluator](../tools/evaluate_joint_ensemble.py)

Next isolate the renderer on the already stronger frozen single Transformer.
Retain all signal/numerical/output audits; changing the renderer does not prove
better raw learning or establish the research hypothesis.

Independent replay confirms all12 member singer/source lineages, unchanged
CV-versus-delivery prediction arrays,2 encoded verified outputs with exact
outside samples,50 original identities and6 protected files. Seventeen combined
ensemble/R3/joint/curve/runtime tests pass. The follow-up R3 renderer also loses
its fixed comparison; neither experiment activates a model.

- [Integrity receipt](../results/transformer_usability/2026-10-29_v19_ensemble/integrity_receipt.json)
