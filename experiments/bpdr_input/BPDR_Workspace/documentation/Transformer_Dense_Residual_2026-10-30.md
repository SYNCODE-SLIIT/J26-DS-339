# Dense residual Transformer experiment

## Implemented change

V26 retains the frozen reference Transformer's original per-frame hidden state,
amplitude head and activity head. Two35-to1 linear adapters add learned deltas
to its amplitude and activity logits. Their inputs are the32 learned center
features and3 existing input-only local step cues. There are72 trainable
parameters; all backbone and original-head parameters stay frozen.

The adapter starts at exactly zero, preserving the existing dense temporal
predictions. It changes neither candidate geometry nor permission limits.
The output remains `30*tanh(amplitude_logit)*sigmoid(activity_logit)`, with
unvoiced output zero. Boundary metadata and clean references are absent from
prediction inputs. This is a trained extension of the Transformer; it is not
a constant-region readout or a new backbone fit.

## Prospective protocol

The declaration is saved before training and binds source code, input files,
four out-of-fold parents, three conditional full-refit parents, loss, sampling,
budget and selection. Failed V25 readouts remain retained.

- Same108 valid actual-audio training cases from200 attempts;92 construction
  exclusions remain in evaluation. No new audio or new people.
- Existing joint feature-only augmentations from reliable training originals;
  these are identified separately from actual recordings.
- Four original singer/source folds; only training partitions enter fitting.
- Original joint frame sampler/loss:12 paired/original/anchor triples per
  update,25% augmentation,70% plateau sampling, zero-original weight4,
  zero-outside weight2 and positive weight1; normalized anchor loss plus
  0.25 response,0.5 activity BCE and0.25 positive amplitude loss.
- AdamW0.001, weight decay0.001, gradient clipping1;1200 head-only updates
  per CV fold. Seed339 fixed. Full-head2400 updates for339/340/341 only
  after actual-WAV selection and independent verification.
- Require exact zero-adapter/reference outputs on every unique training
  input, including feature augmentations, before the first update.
- Select only if all-singer prediction gates pass, all50 original WAVs stay
  unchanged, and singer-macro delivered error beats20.3421417351 cents.
  Keep all existing renderer/output limits. No repair-count-only selection.

This isolates an engineering candidate against the frozen reference. It does
not establish superiority at matched total compute or a learned boundary
invariance objective. Boundary invariance remains input construction.

## Results

Implementation is complete. All23 focused tests pass, including three new model tests: initial-function
preservation and metadata invariance; adapter-only gradients/frozen parent;
strict reload, bounded outputs and unvoiced identity. Four head-only fits finish
1200 updates each; every cached training input passes BITWISE zero-adapter
equivalence before fitting, including feature-only augmentations. Unique input
counts by fold are retained in `initial_equivalence.json`.

| Condition | Raw MAE,cents | Original prediction magnitude,cents | Raw gate | Verified repairs /200 | Delivered after MAE,cents | Original WAVs unchanged |
|---|---:|---:|---|---:|---:|---:|
| Reference | 11.901 | 0.789 | Pass | 23 | 20.342 | 50/50 |
| Dense residual adapter | 11.796 | 0.706 | Pass | 21 | 21.394 | 50/50 |

The adapter slightly improves both raw prediction and original-prediction
magnitude, yet loses actual repaired-audio performance. **Not selected.**
No full refits or development/confirmation evaluation follow. The remaining
work is the model-family comparison described in
[model choice and input usability](Model_Choice_and_Input_Usability_2026-10-30.md).
Independent replay passes: all216 held-out input views reproduce the saved
predictions, zero adapters reproduce the original reference exactly, all4
parent states remain tensor-identical, and all21 encoded verified WAVs pass
hash/decoded/outside-support/direction checks.50 original controls remain
unchanged,6 protected artifacts match and no activation card exists. No
usability or research-validation claim is made.

Commands: `train_dense_residual_transformer.py declare`, `cv`, `delivery`,
`verify_dense_residual_study.py`, and conditionally `refit`, using the existing
workspace Python environment. Declarations and checkpoints live in
`results/transformer_usability/2026-10-30_v26_dense_residual`.

Existing ExtraTrees selection, notebooks, shared demo and other members'
components remain protected. No known-development or reserved-confirmation
audio is accessed by this experiment. Usability remains MOD-03's outstanding
requirement; a passing CV metric alone cannot satisfy it.
