# Complete-support Transformer supervision

Status: eight CV fits and both actual-WAV evaluations completed; neither
condition selected. MOD-03 remains incomplete. No activation.

The previous warm-start candidate improved raw MAE but lost delivered repairs.
Its input-only replay found26 identity cases containing insufficient-support
candidates, alongside overlapping activation, shape and protection rejections.
This study targets a specific training gap: isolated sampled frames do not
directly teach a consistent response throughout a minimum-length repair.

## Controlled training comparison

Both new conditions use the unchanged eight-channel broad/step Transformer,
the exact corresponding V12 parent checkpoint and four singer/source partitions.
They share seed339, AdamW learning rate0.0002,600 additional updates,36 point
windows per update, uniform original anchors and existing feature-only
augmentation. A separate RNG selects one real paired43-frame block per update;
both conditions forward its86 clean/corrupted windows. Point/block sampler
hashes must match between conditions.

| Condition | Extra training supervision |
|---|---|
| `continued_point` | Existing point losses only; block predictions computed but excluded from its gradient |
| `support_blocks` | Weight0.25 dense paired block objective: anchor, response, activity and soft-min support margin |

Each block is43 consecutive fully labelled frames (0.86 seconds), with a
constant nonzero corrupted target and zero clean target. Both views must be
reliable. Ramps, unknown labels, sign transitions, gaps and short labels cannot
be padded or bridged. Cases without complete blocks remain in point training
and every evaluation denominator. The auxiliary margin uses the existing
8-cent activation and4-cent release levels and conservative soft extrema;
an isolated low prediction cannot hide in a block average.

Unlike V11's three-neighbour difference loss, this supplies dense absolute and
activity labels over a complete physically supported region. It does not
establish proposal-objective superiority, additional people or new real WAVs.
The new arms share update and input-forward budgets; backpropagation work
differs because one auxiliary objective is disabled. Against the frozen
1,200-update reference, the1,800-total-update children are engineering
comparisons with extra training, not matched-budget proof.

## Evaluation and boundaries

Architecture and callable inference inputs remain unchanged. No labelled
interval or reference enters inference. Runtime support, protection, eligibility,
30-cent maximum,0.86-second minimum,25% duration budget,5-cent disagreement,
median11, Praat and both fine/coarse encoded-WAV audits stay fixed.
All200 attempts,108 scoreable interventions and50 originals remain.

Select by all-singer prediction gates, original WAV identity and lower
singer-macro delivered error; retain the reference on ties. Only a newly
selected qualified condition can receive full refits from V8 parent
seeds339/340/341 plus1,200 updates. Seed339 stays fixed for known development.
A favourable development replay is not fresh confirmation. Reserved CSD and
male10/male11 audio remains unopened until locked development passes.

Three focused tests passed before fitting: contiguous known labels required;
a single gap penalized despite equal mean; sign symmetry; false clean
corrections increase loss; finite gradients. Active exports, notebook edits,
shared demo and sibling components stay intact.

- [Prospective declaration](../results/transformer_usability/2026-10-29_v16_support/declaration.json)
- [Objective](../source/bpdr_input/experiments/learning/support_supervision.py)
- [Matched trainer](../source/bpdr_input/experiments/learning/support_transformer_training.py)
- [Runner](../tools/train_support_transformer.py)

Run the runner with `cv`, `delivery`, then conditionally `refit`.
The existing `evaluate_response_transformer.py --run-root` checks full-data
groups, sources, parent lineage, seeds, hashes and updates before development
access. No model is qualified or activated by this plan.

## Completed outcome

| Condition | Raw MAE, cents | Raw original mean, cents | Model-only complete-support cases /108 | Verified repairs /200 | Delivered after MAE, cents | Original identity |
|---|---:|---:|---:|---:|---:|---:|
| Frozen reference | 11.901 | 0.789 | 68 | 23 | 20.342 | 50/50 |
| Continued point | 11.635 | 0.922 | 68 | 21 | 21.753 | 50/50 |
| Support blocks | 11.607 | 1.001 | 68 | 21 | 21.140 | 50/50 |

All three pass the all-singer prediction gate. Matched sampling/parent hashes
agree between the two new arms in every fold. Block supervision modestly beats
its matched continuation control in raw and delivered error, but neither beats
the frozen reference in delivered output. It does not improve the targeted
continuity count. The locked selection retains `matched_baseline`; no full
refits, development replay or confirmation access is justified.

The continuity metric requires a sign-correct predicted43-frame block (all
frames>=4 cents, at least one>=8 cents) within genuinely labelled constant-target
frames, before scope/protection/permission/rendering. Only103 of108 scoreable
cases have such complete labels. All200 attempts remain in the repair
denominator; model-only support is not delivered repair coverage.

An explicit training-label-only plateau reference, without any rendering or
neural success claim, yields45 permitted nonidentity plans and63 identities
among108 scoreable cases. Candidate rejection labels include44 shape failures
and4 protected overlaps. This is a diagnostic of this specific labelled curve,
not a global achievable ceiling and never an inference input. It reinforces
that frame MAE alone cannot establish usable pipeline performance.

Fifteen focused supervision/joint/curve/runtime tests pass. Independent replay
verifies552 frozen hashes,8 checkpoint lineages,42 encoded WAVs including decoded
hashes and exact outside samples,50/50 original identities per arm and all6
protected files. A Transformer activation card remains absent.

## Next evidence-driven change

Earlier approved training-only experiments contain other real rendered
interventions at different durations and positions. Preflight across six
specified historical manifests finds373 unique previously audited variants
from the same50 original recordings/10 singers, including265 beyond the current
108. The1,200 manifest rows contain200 repeated valid rows and627 construction
exclusion rows; these are not1,200 unique experiments or additional people.
The next study must freshly re-audit labels, retain failed constructions, avoid
duplicate weighting and inherit original singer/source folds before fitting.
V6 evaluation stays200/108, with no development/final/confirmation data added.

- [CV results](../results/transformer_usability/2026-10-29_v16_support/cross_validation.json)
- [Actual-WAV selection](../results/transformer_usability/2026-10-29_v16_support/locked_selection.json)
- [Continuity report](../results/transformer_usability/2026-10-29_v16_support/support_blocks_support_metrics.json)
- [Labelled-plateau reference](../results/transformer_usability/2026-10-29_v16_support/labelled_plateau_reference.json)
- [Integrity receipt](../results/transformer_usability/2026-10-29_v16_support/integrity_receipt.json)
- [Historical pool preflight](../results/transformer_usability/2026-10-29_v17_audio_pool/pool_preflight.json)
