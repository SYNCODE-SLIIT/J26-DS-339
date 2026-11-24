# Frozen-Transformer candidate heads and centre-readout comparison

Status: eight head fits and three actual-WAV evaluations complete. No variant
selected; MOD-03 remains active. No full refit, development access or activation.

## What changed

V22 pools the frozen Transformer representation over an input-measured step
region and fine-tunes two small readout heads. Its66 trainable parameters
predict one constant region correction and activity value; the original dense
Transformer prediction remains unchanged outside measured regions. The dense
parent, positional encoding, projection, encoder, normalization and original
heads remain frozen. This is head-only fine-tuning, not a new backbone fit.

Supervision uses177 positive and8 real negative candidates, with14 unknown
labels excluded from the loss but retained in data accounting. Every positive
candidate is paired with a zero-target original query at the same input-derived
interval, plus up to four deterministic original-anchor queries per reliable
source. These queries are training controls, not additional measured negative
candidates, independent singers or real WAVs. Every descendant inherits its
original singer/source fold. No reference interval is used at inference.

V23 adds three direct input-cue summaries to the pooled representation: signed
step estimate, maximum fit-gain cue and maximum pitch-spread cue. Six additional
connections are zero-initialized, preserving the initial V22 readout. All
sampling traces, labels, losses, optimizer settings and1,200 head updates match
V22 in every fold. It has72 trainable parameters. The comparison includes both
new features and six parameters; it does not establish the proposal objective.

V24 is a separate frozen ablation: use the ORIGINAL learned prediction at each
input-measured candidate's centre and repeat it over that exact region. Dense
predictions elsewhere remain unchanged. It trains no head and changes no
backbone weights. It is not evidence that the failed trained heads succeeded.

All variants retain30-cent maximum,0.86-second minimum,25% edit budget,
5-cent disagreement/uncertainty limits, protection, median11/Praat and all
fine/coarse post-render plus encoded-WAV checks. No interval is padded or
adjusted using a target. All200 attempts,108 scoreable cases and92 construction
exclusions remain. Metrics below are singer-macro values on the scoreable cases;
repair rates retain all200 attempts. No new independent singers were added.

## Completed results

| Variant | Raw MAE,cents | Raw original mean,cents | Prediction gate | Verified repairs /200 | Delivered after MAE,cents | Original WAV identity |
|---|---:|---:|---|---:|---:|---:|
| Frozen reference | 11.901 | 0.789 | Pass | 23 | 20.342 | 50/50 |
| V22 pooled head | 12.399 | 0.789 | Fail | 20 | 20.965 | 50/50 |
| V23 pooled head +cue summaries | 12.367 | 0.789 | Fail | 21 | 20.808 | 50/50 |
| V24 frozen learned-centre readout | 11.854 | 0.789 | Pass | 24 | 20.706 | 50/50 |

Both trained heads miss male7's fixed0.25 response requirement; V22 reaches
0.2135. V23's small raw/delivery gain over V22 does not beat the reference.
V24 gains one verified case but leaves more average pitch error. The declared
selection rule therefore retains the original reference. Do not replace it
with a case-count-only rule after observing this outcome.

The region proposal unit reduces neither the evaluation denominator nor the
scope requirement.39/108 old scoreable cases lack candidate overlap; the dense
fallback remains necessary. Sparse real-negative supervision remains a limit.
Musical intention, natural singing accuracy and listener approval remain
unestablished. These are engineering comparisons, not fresh final evaluation.

## Verification

Twenty focused tests pass across candidate geometry, both heads, readout,
joint Transformer, curve conditioning and guarded runtime. They check initial
function preservation, gradients on new cue connections, frozen-parent
gradients, metadata invariance, exact fallback, support bounds and reloads.

Independent replay verifies all eight parent states tensor-for-tensor against
their original checkpoints, fitting candidate/anchor counts, held-out singer
and source disjointness, and exact dense-fallback predictions outside measured
regions for every scored input. It checks all65 encoded verified WAVs across
the three variants, decoded hashes, zero wrong-direction audit frames, exact
outside samples,50 original-control identities per variant and6 protected
artifacts. No Transformer activation card exists. No shared demo, selected
export or user notebook change is made by these experiments.

## Next bounded diagnostic

The learned-centre ablation slightly improves raw error and case count while
losing average delivered error. Next prospectively compare two GLOBAL
within-region readouts of the existing learned predictions: a median and an
upper-magnitude quartile, with fixed tie/sign handling. The purpose is to test
under-correction versus unstable point estimates; it is not a permission-limit
change or calibrated confidence estimate. Keep the original model outside
measured regions, keep all numerical/output gates and select only by the
existing raw/original/actual-WAV rule. No per-case target routing, development
tuning or new backbone fit. A new declaration must precede the comparison.

## Evidence and commands

- [V22 declaration](../results/transformer_usability/2026-10-30_v22_candidate_head/declaration.json), [metrics](../results/transformer_usability/2026-10-30_v22_candidate_head/cross_validation.json), [WAV evidence](../results/transformer_usability/2026-10-30_v22_candidate_head/arms/candidate_head/training/median11/evidence.json), [integrity](../results/transformer_usability/2026-10-30_v22_candidate_head/integrity_receipt.json)
- [V23 declaration](../results/transformer_usability/2026-10-30_v23_candidate_evidence/declaration.json), [metrics](../results/transformer_usability/2026-10-30_v23_candidate_evidence/cross_validation.json), [WAV evidence](../results/transformer_usability/2026-10-30_v23_candidate_evidence/arms/candidate_evidence/training/median11/evidence.json), [integrity](../results/transformer_usability/2026-10-30_v23_candidate_evidence/integrity_receipt.json)
- [V24 declaration](../results/transformer_usability/2026-10-30_v24_center_readout/declaration.json), [metrics](../results/transformer_usability/2026-10-30_v24_center_readout/cross_validation.json), [WAV evidence](../results/transformer_usability/2026-10-30_v24_center_readout/arm/training/median11/evidence.json), [integrity](../results/transformer_usability/2026-10-30_v24_center_readout/integrity_receipt.json)
- [Independent verifier](../tools/verify_candidate_studies.py)

Use `train_candidate_transformer.py` or
`train_candidate_evidence_transformer.py` with `cv`, then `delivery`, then
conditionally `refit`. Failed selection must refuse refitting. Use
`evaluate_candidate_center_readout.py cv`/`delivery` for its separate frozen
diagnostic. Saved declarations/results are retained, not overwritten as passes.
