# Transformer distribution-readout comparison

## Purpose and frozen scope

V25 tests whether a single learned frame is an unstable representative of a
measured repair region. It compares two globally fixed readouts of the existing
Transformer predictions: median and upper-magnitude quartile. This is an
inference comparison, with **zero new neural fits**. It does not demonstrate a
better trained backbone or calibrated confidence.

Before scoring, the declaration fixes the four held-out checkpoints, source
and input hashes, both readouts, sign handling, rank rounding and selection
rule. Each readout chooses an actual observed learned value and its associated
activity. It retains the dense prediction outside input-measured regions and
does not extend support. Ambiguous zero-sign regions are suppressed.

All existing limits, original controls, renderer and encoded-WAV checks remain
fixed. Selection requires the all-singer prediction gate, all50 original WAVs
unchanged, and lower singer-macro delivered error than the reference. A larger
repair count alone cannot select a variant. Ties retain the reference.

## Prediction results

These are training-side singer/source-held-out results:200 attempted cases,
108 scoreable,92 construction exclusions,50 original recordings. Only8 of the
10 attempted singers have scoreable intervention cases; the other2 remain
unassessed. The same previously inspected cohort is used for all variants.

| Readout | Raw MAE, cents | Within10 cents | Original prediction magnitude, cents | All-singer prediction gate |
|---|---:|---:|---:|---|
| Frozen reference | 11.901 | 61.32% | 0.789 | Pass |
| Median | 11.838 | 62.71% | 0.789 | Pass |
| Upper quartile | 11.774 | 62.71% | 0.789 | Pass |

Original prediction magnitude is a model diagnostic; unchanged delivered audio
requires the separate original-control check. These values are not natural
singing accuracy or intended-note correctness.

## Delivered-audio evaluation

| Readout | Verified repairs /200 attempts | Remaining delivered error, cents | Original WAVs unchanged | Selected |
|---|---:|---:|---:|---|
| Frozen reference | 23 | 20.342 | 50/50 | Retained |
| Median | 18 | 21.444 | 50/50 | No |
| Upper quartile | 19 | 21.992 | 50/50 | No |

Both variants improve the raw prediction diagnostic slightly while worsening
the delivered waveform result. Neither qualifies. There is no development or
confirmation evaluation, full refit, activation or usability pass.

Each condition has30 attempted rendering transactions across216 paired
original/intervention views of108 scoreable cases;186 views stay at identity.
The reference rolls back7 transactions, median12 and upper quartile11. Thus
these readouts do not solve the missing-support problem and lose additional
outputs at verification. These counts describe stages, not extra independent
cases. Withheld or rolled-back cases remain in the delivered-error score.

## Verification and reproduction

Twenty focused unit tests pass across distribution and centre readouts, input
candidate geometry, the joint Transformer, curve conditioning and guarded
runtime. Independent replay verifies exact CV/delivery predictions, observed
within-region value selection, dense fallback, all37 saved verified encoded
WAVs,50 original identities per arm,6 protected files and absent activation.
The repository uses `unittest`; `pytest` is not installed in the local
environment. No dependency installation is required.

Run `evaluate_candidate_distribution_readout.py cv`, then `delivery`, then
`verify_distribution_readout.py` from the workspace using its existing Python
environment. Existing CV results are retained rather than overwritten.

Evidence lives in
`results/transformer_usability/2026-10-30_v25_distribution_readout`:
`declaration.json`, `cross_validation.json`, each arm's
`training/median11/evidence.json`, `locked_selection.json`, and
`integrity_receipt.json`.

## Next model experiment

Stop replacing entire measured regions with one representative frame. The next
bounded comparison should preserve the original dense Transformer function at
initialization and learn a small residual correction to its existing amplitude
and activity heads. Keep the backbone frozen, retain per-frame temporal shape,
use the existing input-only local cues, and compare an exact zero-adapter
control against one trained adapter under the same singer/source holdouts.
This tests a different failure mechanism from V22/V23's changed pooled readout.
It is a planned experiment, not an implemented improvement.

Before fitting, lock its loss, sampling, training budget and existing selection
criteria. Verify zero-adapter equivalence, frozen-parent tensors, gradients,
strict checkpoint reload and metadata invariance. Use only approved training
sources; keep all failed candidates and untouched confirmation sources intact.

Known development may be evaluated only after a qualifying model/decoder is
locked and independently verified. Reserved confirmation audio remains
unopened. Fresh confirmation and independent musical review remain required
before any real-world usability claim or activation. The existing ExtraTrees
selection remains active.
