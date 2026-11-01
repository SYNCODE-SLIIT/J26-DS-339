# Matched absolute-response training

## Purpose and declared intervention

The proposal specifies mean-absolute counter-detuning response. Current
engineering training uses a squared response residual. V31 tests exactly this
one difference on the current Transformer and training data, following the
measurement/locality/decision diagnostics. It does not change the architecture,
decoder, input features, permission, support or rendering/auditing thresholds.

Both conditions retain weighted squared anchors (original4,outside2,positive1),
activity BCE0.5,positive-only amplitude MSE0.25,12 triples/update, the existing
25% feature-only augmentation and70% plateau sampling. The response coefficient
stays0.25. Only its residual changes:

- Control: mean square of `prediction_C - prediction_A - wanted_change`.
- Treatment: mean absolute value of that same normalized signed residual.

Data, folds, seeds, random draw sequence, optimizer and update budget are fixed:
200 attempted cases,108 scoreable,92 exclusions,50 originals; four singer/source
folds,seed339,AdamW0.001,weight decay0.001,clip1,1200 updates. Both arms train
from scratch. Feature augmentations remain identified as features, not new
audio/people. No positive-target reweighting or additional objective sweep.

This tests the absolute response term only. It is **not** completion of RES-01's
full BPDR/augmentation-only/frame-only comparison or evidence of learned boundary
invariance. Boundary invariance remains input construction.

## Execution

The declaration and source snapshots precede fitting. All four newly fitted MSE
controls reproduce the authenticated V12 model tensors, all validation prediction
arrays and actual sample digests exactly. Source/singer partitions and seed/
update lineage also match. Only after all four controls pass do MAE fits begin.

Four controls are complete; four absolute-response fits are running. Their
accuracy and delivered-audio outcomes are pending. All32 focused tests pass,
including preservation of the control arithmetic and signed gradients of the
absolute response. New loss tests are not an accuracy qualification.

## Selection and remaining requirements

Require every represented singer's raw preservation/response gate, all50
original WAV identities and strictly lower singer-macro remaining delivered
error than20.3421417351c. Use the same median11/Praat pipeline and every
existing encoded-WAV verification. Count alone cannot select a model. The
authenticated control WAV evidence is reused only after exact control
reproduction. No claims of equal total wall-clock budgets beyond the stated
per-fit data,optimizer and1200 update schedule.

Independent verification is implemented in `verify_absolute_response_study.py`
for control reproduction, MAE metadata, held-source separation, matching
sampler traces, actual predictions and saved WAVs. It runs after delivery.
Any later full fits/default339,known development and untouched confirmation
need their own qualified lock. No current adoption or usability claim.

Commands: `train_absolute_response_transformer.py cv`, then `delivery`, then
`verify_absolute_response_study.py`, using the existing workspace environment.
Evidence: `results/transformer_usability/2026-11-01_v31_absolute_response`.
Existing files and all previous failed studies are retained.
