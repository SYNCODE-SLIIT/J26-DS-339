# Broader actual-audio training pool

Status: data preparation,373/373 acoustic re-audits and the eight-fit matched
model comparison completed. Broader-data candidate rejected. No activation.

Complete-support supervision did not increase model-supported case count or
beat the reference's delivered repairs. The next candidate will vary actual
audio examples rather than run another small loss-weight sweep. This stage
checks already authorized component-local historical training data first.

## Sources and deduplication

The pool prospectively names V6,V1,long-V2,V3,V4,V5 training manifests, in that
order, and permits only the50 original source hashes/10 singer identities in
V6. No development IDs5/6, inspected final IDs8/9 or reserved CSD/male10/male11
sources enter. Every derivative must inherit its original singer/source fold.

The1,200 historical manifest entries include373 unique valid variants,200
duplicate valid entries and627 construction-exclusion entries. All source
attempts remain in the provenance register. These are not1,200 independent
experiments or new people. The current108 valid V6 cases come first;265 other
unique variants add real rendered duration/position variation from the same
original recordings. Amplitudes remain the existing+/-20/30 cents.

Preflight verifies original file hashes and source/category consistency,
unchanged policy-0.3 historical audit receipts, exact outside-support waveform
samples, label/time-grid/feature alignment and all current input hashes. It
deduplicates by source, decoded clean waveform and decoded shifted waveform;
identical audio with conflicting supervision stops preparation. V3/V4 valid
entries duplicate long-V2 and cannot inflate fitting weights.

## Fresh acoustic audit before fitting

Recompute fine YIN from each actual clean/shifted WAV, retain the saved PESTO
features, recompute current runtime reliability and apply the existing physical
intervention audit. No tolerance changes. Base108 cases must all pass without
losing existing labels; otherwise no fitting-ready manifest is released.
Additional valid masks only intersect their old masks with freshly verified
frames. No new label is inferred, no failed case is replaced, and no waveform
is smoothed. All current inputs are hash-checked before and after.

Only a successful receipt may release `fit_pairs.json`. Before then,
`candidate_pairs_NOT_TRAINING_READY.json` is an inventory, not training data.
The candidate pool does not establish natural intended-pitch accuracy or
replace independent musical review. Cached PESTO evidence is explicitly
distinguished from the newly measured fine-pitch audit.

## Planned model comparison after data readiness

### Data readiness outcome

All373 variants revalidate. All108 base cases retain every existing label.
The released fitting manifest contains465 entries: the unchanged200 base
attempts (including92 construction exclusions) plus265 unique valid additional
variants. It has373 scoreable variants from the same50 original recordings and
10 singers. No new original singer or independent evaluation is claimed.
The data receipt and fitting manifest are hash-bound; base evaluation remains
exactly200/108. Saved per-case re-audits bind additional target files and masks.

- [Acoustic pool receipt](../results/transformer_usability/2026-10-29_v17_audio_pool/acoustic_pool_receipt.json)
- [Fitting-ready manifest](../results/transformer_usability/2026-10-29_v17_audio_pool/fit_pairs.json)

### Declared model experiment

Freeze a separate model declaration. Compare parent continuation on current
data with continuation on the deduplicated actual-audio pool at matched update
budgets, retaining uniform originals and existing losses. Do not simultaneously
change model architecture, protection, support limits, renderer or output audit.
V6 evaluation remains200 attempts/108 scoreable interventions; extra variants
from a held-out singer may never enter its training fold. Default seed339 and
three qualified full refits remain required before known development access.
No confirmation audio access before a locked development pass.

The separate V18 declaration now compares `continued_point` and
`audio_augmented`: eight new four-fold fits, seed339,600 additional updates,
36 windows per update, the same V12 fold parents, AdamW0.0002 and unchanged
point losses, uniform original anchors and feature-only controls. Only the
actual-audio pool differs. Between-arm case traces are intentionally different;
the control must exactly reproduce V16 continuation parameters and its sampler
in every fold. Compared with the frozen V12 reference, both children have an
additional600 updates. Full refits remain conditional on actual-WAV selection.

The qualified-development checker now validates the selected arm's exact
fitting manifest and source allow-list, not merely the evaluation manifest.
It permits declared augmented parents to have a subset of the expanded fitting
sources, while preserving strict full-data refit lineage and disjoint evaluation.
Three new tests pass: selected-manifest/hash binding, source/singer holdout
refusal even with updated file hashes, and refusal of external manifests.

- [Model declaration](../results/transformer_usability/2026-10-29_v18_audio_augmented/declaration.json)
- [Model comparison runner](../tools/train_audio_augmented_transformer.py)
- [Matched fitting code](../source/bpdr_input/experiments/learning/audio_augmented_training.py)

## Completed model result

| Condition | Raw MAE, cents | Raw original mean, cents | Complete-support cases /108 | Verified repairs /200 | Delivered after MAE, cents | Original identity |
|---|---:|---:|---:|---:|---:|---:|
| Frozen reference | 11.901 | 0.789 | 68 | 23 | 20.342 | 50/50 |
| Matched current-data continuation | 11.635 | 0.922 | 68 | 21 | 21.753 | 50/50 |
| Broader actual-audio continuation | 11.322 | 1.493 | 72 | 18 | 22.186 | 50/50 |

The new control reproduces all V16 control parameters exactly and its sample
hash in every fold. Broader audio improves raw error and potential model-only
support, but fails the raw-original gate for female2 (3.207 cents versus the
fixed3-cent limit) and loses delivered repair. Both regression and actual WAVs
therefore reject adoption. Four additional model-supported cases are not four
additional verified repairs. Unknown natural intended-pitch accuracy and
independent listening remain unestablished.

The locked selection retains `matched_baseline`. The actual refit command
refuses the unselected new condition. No full refit, development replay,
confirmation audio access or activation occurred. The validated dataset is
retained for subsequent model work; failed experiments are not overwritten.

Eighteen focused supervision/qualification/joint/curve/runtime tests pass.
Independent verification checks1,419 frozen hashes,8 fold lineages and fitting
manifest/count bindings,39 encoded verified WAVs including decoded hashes and
exact outside samples,all50 original controls per arm and6 protected files.

The next comparison should freeze a single three-member Transformer ensemble
(V12 reference,V16 support,V18 broader data), with explicit member dispersion
and a global combination rule. Every member must retain singer/source holdouts;
no per-case reference routing, target-driven support or relaxed output limits.
The ensemble is a new candidate, not adoption of a failed member, and must pass
raw/original and actual-WAV gates before any composite full refits. Member
dispersion is not calibrated confidence. Keep reserved confirmation unopened.

- [Cross-validation](../results/transformer_usability/2026-10-29_v18_audio_augmented/cross_validation.json)
- [Actual-WAV locked selection](../results/transformer_usability/2026-10-29_v18_audio_augmented/locked_selection.json)
- [Model-only continuity](../results/transformer_usability/2026-10-29_v18_audio_augmented/audio_augmented_support_metrics.json)
- [Integrity receipt](../results/transformer_usability/2026-10-29_v18_audio_augmented/integrity_receipt.json)

- [Prospective pool declaration](../results/transformer_usability/2026-10-29_v17_audio_pool/pool_declaration.json)
- [Preflight receipt](../results/transformer_usability/2026-10-29_v17_audio_pool/pool_preflight.json)
- [All historical source entries](../results/transformer_usability/2026-10-29_v17_audio_pool/all_source_attempts.json)
- [Acoustic re-audit declaration](../results/transformer_usability/2026-10-29_v17_audio_pool/acoustic_reaudit_declaration.json)
- [Preflight tool](../tools/prepare_historical_training_pool.py)
- [Acoustic re-audit tool](../tools/reaudit_historical_training_pool.py)
