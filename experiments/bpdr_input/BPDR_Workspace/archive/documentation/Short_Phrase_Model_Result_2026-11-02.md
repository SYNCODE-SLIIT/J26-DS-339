# Short-phrase model comparison: complete, candidate rejected

Completed 2 November 2026. V41 fitting/checkpoint replay and V42 actual-WAV replay
are complete. MOD-03 remains ACTIVE. The Transformer is not qualified for normal
short-hum usability and is not activated. Current demo/model selection is unchanged.

## What was fitted and checked

V40 supplies44 audited pairs/336 attempted physical deviations,19 valid original
controls/21 attempted sources,14 attempted training singers and6 singers with valid
positive pairs. All existing nontraining singer partitions stayed closed. The dataset
construction exclusions and reference limitations are in the
[execution record](Short_Phrase_Training_2026-11-02.md).

Four grouped folds completed12 fits: retrained old joint representation, revised
short-phrase Transformer and matched ExtraTrees. The fixed current model is a
separate different-history reference. Both new neural arms have21,666 trainable
parameters,800 updates and identical sampled rows/loss; token/feature contracts
differ. New Transformer versus ExtraTrees shares the same81x8 token matrix and
padding indicators, but not loss/compute or an optimal tuning search.

Independent replay reloads all12 checkpoints plus the frozen reference, reinfers
252 prediction arrays, recreates all4 sampler traces and verifies source/singer
separation within the Vocadito folds. Every summary is reproduced.

## Prediction results

| Arm | Positive singer-macro MAE (cents) | Original singer-macro raw request (cents) | All-group raw gate |
|---|---:|---:|---|
| Current frozen joint Transformer | 24.924590 | 0.488723 | FAIL |
| Old joint representation retrained | 24.996343 | 0.121046 | FAIL |
| Revised short-phrase Transformer | 24.147397 | 0.567278 | FAIL |
| Matched short-context ExtraTrees | 24.669994 | 0.598156 | FAIL |

The raw gate requires positive MAE<25c and signed response>=0.25 for every scoreable
group, plus original mean absolute prediction<=3c for every represented original
group. The revised model responds meaningfully only on S29: positive MAE19.902670c,
response0.302, but its original request4.659c fails preservation. Other scoreable
groups have near-zero response. Averages conceal this failure.

The slight aggregate improvement does not support a usable-accuracy claim. The
ExtraTrees comparator also fails; this experiment does not demonstrate that
Transformer architecture alone is the cause, or establish architecture superiority.

## Actual delivered-WAV result

The V42 policy was declared before inspecting final raw predictions. Every arm
uses the same input-only proposals, real short supports, protection/rollback,
30c/25% bounds,20ms ramps,70ms audit guard and unchanged error/direction/coverage/
ten-frame gates. No known injected support or clean reference enters inference.

| Arm | Verified target-plateau improvements /336 attempts | Target-plateau harms | Changed original controls /19 | Remaining singer-macro MAE |
|---|---:|---:|---:|---:|
| Frozen reference | 0 | 0 | 0 | 25.040978c |
| Retrained old representation | 0 | 0 | 0 | 25.040978c |
| Revised Transformer | 4 | 4 | 2 | 25.236301c |
| Matched ExtraTrees | 0 | 0 | 0 | 25.040978c |

Input remaining error is25.040978c. The revised model makes it worse overall. Its
two changed original controls are `vocadito_37` and `vocadito_39`. Allfour target
improvements and allfour target harms occur on S29. These variant counts are not
independent singers, participant success or general intended-pitch accuracy.

The locked gain metric concerns the nonzero intervention plateau; it does not
claim that every other part of that output is musically appropriate. A separate
post-hoc reference-preservation diagnostic finds increased error on predeclared
known-zero frames in18/19 verified revised-Transformer corrupted-input outputs
and3/3 retrained-old outputs. Thus a passed technical correction audit can still
execute an inappropriate prediction. These collateral counts supplement, and do
not rewrite, the locked target-plateau gain/harm metric.

Independent V42 replay reproduces252 proposals/plans,25 in-memory candidate audits,
24 encoded audits,176 fixed-reference scores and76 original-control checks across
the four arms. Exact source/output hashes, transactions and summaries reproduce.
The test therefore rejects the candidate; no registry/card/demo promotion occurs.

## What the failure teaches us

The four fixed revised checkpoints achieve training-singer positive MAEs4.83,
4.56,5.61 and4.64c, versus corresponding held-out errors25.00,25.00,25.00 and19.90c.
This establishes a transfer gap, not an inability of this small Transformer to fit
the supplied training examples. It does not by itself prove a single causal reason.
No additional fit or hyperparameter selection was performed for this diagnostic.

A source-only geometry check also shows weak local anchors in some examples:
after leaving the70ms measurement guard, none of the S22 or S7 variants has at
least3 reliable C frames on both sides within the original acoustic region.
Only4/8 S23 variants meet that check; S5,S28 and S29 do. This hypothesis cannot
explain every failure: S5 and S28 also fail despite available side observations.
Do not present anchor shortage as a proved universal cause.

The next version should distinguish physically measurable interventions from
locally identifiable errors, add acoustic-region-relative short-event evidence,
and broaden the training-only melodic coverage and preservation examples. The
0.24s measurement gap still needs a separately declared, negative-control-tested
protocol. Do not simply increase epochs, lower an error threshold, discard failed
singers or copy musical-context/key inference into another member's feature area.

Future qualification must additionally examine known-zero C preservation, not
only the positive plateau and original A controls. Current results remain failed
regardless; this is a prospective strengthening of the next protocol.

## Evidence and document

- `results/transformer_usability/2026-11-02_v41_short_phrase_models/`: declarations,
  all checkpoints/fold traces, raw results and independent replay receipt.
- `.../generalization_diagnosis/results.json`: fixed-model train/held-out diagnosis
  with saved training predictions; no new fit.
- `results/transformer_usability/2026-11-02_v42_short_delivery/`: prospective policy,
  all decisions, private candidate WAVs, complete results and independent receipt.
- `.../reference_preservation_diagnosis.json`: explicitly post-hoc collateral check.
- [Current-system and progress PDF](../output/pdf/BPDR_Current_System_and_Transformer_Progress_2026-11-02.pdf):
  reader-facing current system, research goal, ownership, results and remaining work.

No source or derivative audio has been added to a public/shared demo. Independent
musical/reference review and actual participant collection remain incomplete.
