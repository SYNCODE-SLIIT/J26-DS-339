# Warm-started broad/local Transformer

Status: four fits and actual-WAV evaluation completed; candidate rejected.
MOD-03 remains incomplete; no activation.

The V14 multiscale models improved raw correction but failed unchanged-vocal
prediction limits and lost the actual-WAV comparison. This candidate starts
from the strongest existing broad-context Transformer instead of training the
wider input from scratch. Five new local-context projection columns start at
zero; all existing broad/step weights and Transformer layers are copied.
Two transfer tests passed: initial predictions match, local weights receive
gradients, parent parameters are independent, and localized parents are refused.

Four CV fits use their matching V12 fold parents, never a full-data checkpoint.
Training verifies exact parent singer/source sets and seed, freezes the parent,
and compares actual clean/corrupted probe outputs before learning. Each child
adds 600 updates at learning rate 0.0002. Half of extra original frames use the
same reliable difficult-original sampler as V14. A weight-0.5 teacher-output
preservation loss applies only to zero-target examples. Other losses remain.

This is an engineering candidate combining changes and extra training, not a
matched-budget proof that multiscale context or the research objective is
superior. The baseline remains frozen at 1,200 CV updates; each child receives
1,800 total updates. Seeds and sampled windows are not additional people.

All 200 attempted /108 scoreable interventions, 50 originals, singer/source
partitions, median11 conditioning, Praat renderer and output audits stay fixed.
No gate, interval support or denominator is relaxed. Selection requires the
all-singer prediction gate, original WAV preservation and lower delivered error.
Only a selected qualified child can receive three full-data refits using the
matching V8 seeds339/340/341 plus1,200 updates. Default seed339 then faces the
already inspected development set; that is not fresh confirmation. Reserved
CSD and male10/male11 audio remains unopened until development passes.

- [Prospective declaration](../results/transformer_usability/2026-10-29_v15_warm/declaration.json)
- [Runner](../tools/train_warm_multiscale_transformer.py)
- [Training](../source/bpdr_input/experiments/learning/warm_multiscale_training.py)
- [Weight transfer](../source/bpdr_input/modeling/multiscale_transfer.py)

Run the runner with `cv`, then `delivery`, then conditionally `refit`.
The generic `evaluate_response_transformer.py --run-root` evaluator checks
qualified full-refit lineage before accessing known development WAVs.

## Completed result

| Metric | Frozen reference | Warm-start child |
|---|---:|---:|
| Raw correction MAE, cents | 11.901 | 11.464 |
| Frames within10 cents | 61.318% | 63.076% |
| Raw original mean, cents | 0.789 | 0.880 |
| All-singer prediction gate | Pass | Pass |
| Verified improvements /200 attempts | 23 | 21 |
| Delivered after MAE, cents | 20.342 | 21.648 |
| Original WAVs exactly preserved | 50/50 | 50/50 |

Each new fold started with exactly matching probe predictions (maximum error
0.0 cents), retained its corresponding parent lineage, and saved a reload-checked
checkpoint. Four reference fits were reused, not counted as new fits. All108
scoreable interventions and92 exclusions remain in the200-attempt denominator.
The locked selection retains `matched_baseline`; additional full refits and
development evaluation are not justified. No reserved confirmation audio was
accessed. Accuracy here concerns controlled interventions, not intended musical
pitch on arbitrary natural singing.

## What the failed comparison reveals

Both conditions leave78 of108 scoreable corrupted inputs unchanged before
rendering and attempt30 renders. Baseline rolls back7, child9. All ten changes
in verified-repair status (four gained, six lost) involve the existing coarse
pitch-direction audit. The child has9 direction failures and2 median-error
failures, with overlapping reasons; baseline has6 and1 respectively. This
shows sensitivity at the encoded-output verification stage. It does not prove
that the tracker is wrong or authorize removing its guard.

The next model decision should use the saved identity-candidate diagnosis and
render-failure traces. Raw-MAE-only training changes have now repeatedly failed
to improve delivered usability; do not start another arbitrary weight sweep.
Keep threshold/support/output limits fixed and study which model-supported
candidates are lost and whether rendering or model support is responsible.

The input-only replay of all78 identity cases is now complete. Rejection labels
overlap:26 cases have a too-short predicted candidate,22 have a candidate
without activation,10 fail the step-shape guard,28 include protected overlap,
and20 have no predicted supported candidate. These are model/policy candidate
labels, not proven causes of the singer's pitch error. Neither a reference
target nor its interval entered these decisions. Preserve protection and
minimum duration; investigate supervised temporal support and calibrated
activity rather than extending predicted intervals or changing the denominator.

## Verification and evidence

Thirteen focused multiscale/transfer/joint/runtime tests pass. The independent
integrity replay checks552 frozen input/source/parent hashes, four fold
checkpoints,21 encoded WAVs including decoded hashes and exact outside samples,
50 original-control identities and all6 protected files. No active Transformer
card exists. The current demo and notebook/export selections remain intact.
The actual full-refit command refuses the unselected child, and the development
command refuses the retained reference before opening evaluation WAVs. These
expected refusals verify the promotion boundary; they are not successful fits.

- [Cross-validation](../results/transformer_usability/2026-10-29_v15_warm/cross_validation.json)
- [Actual-WAV locked selection](../results/transformer_usability/2026-10-29_v15_warm/locked_selection.json)
- [Repair differences](../results/transformer_usability/2026-10-29_v15_warm/repair_difference.json)
- [Failure diagnosis](../results/transformer_usability/2026-10-29_v15_warm/failure_diagnosis.json)
- [Input-only identity-candidate diagnosis](../results/transformer_usability/2026-10-29_v15_warm/warm_multiscale_identity_diagnosis.json)
- [Integrity receipt](../results/transformer_usability/2026-10-29_v15_warm/integrity_receipt.json)
