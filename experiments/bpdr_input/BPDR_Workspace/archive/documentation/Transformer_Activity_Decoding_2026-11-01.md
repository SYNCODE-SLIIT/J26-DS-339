# Fixed activity-decision experiment

## Mechanism and scope

The reference Transformer trains an amplitude head on positive corrections and
an activity head using binary cross-entropy. Its output is the amplitude
multiplied by sigmoid(activity). Thus an amplitude of20c at activity0.8 yields
a16c correction. This algebra motivates one fixed test; it does not establish
that probability shrinkage is the main source of error.

V30 compares the original expected output with the SAME bounded amplitude
only when activity logit is>=0, otherwise zero. The decision is fixed globally
before evaluation; the repair branch wins a tie. No fitted weights, new audio,
threshold tuning, per-case target routing or calibrated-confidence claim.
Existing activation/release thresholds, real support, step evidence, protection,
uncertainty and every encoded-WAV check remain unchanged.

Four frozen singer/source-held-out reference checkpoints are used. Existing
full-refit weights339/340/341 are bound for any later qualified evaluation;
they are not fitted or accessed for known-development inference by this stage.
Every unique scored input first reproduces its original expected prediction
bit-for-bit before the binary comparison is scored.

## Prediction results

| Metric | Original expected output | Binary activity decision |
|---|---:|---:|
| Raw correction MAE,cents | 11.901 | 11.873 |
| Original prediction magnitude,cents | 0.789 | 0.434 |
| Outside-region prediction magnitude,cents | 0.824 | 0.511 |
| Within10c | 61.32% | 62.76% |
| Paired response gain | 0.640 | 0.656 |
| All-singer prediction gate | Pass | Pass |

Counts remain200 attempted cases,108 scoreable,92 exclusions and50 originals.
Eight singers have scoreable interventions; two remain unassessed. These
prediction diagnostics are not natural intended-note accuracy or actual-audio
repair results.

## Delivery and verification

Actual median11/Praat rendering and50 original controls complete. Selection
requires all-singer prediction gates, all50 original WAV identities and strictly
lower singer-macro delivered after-error than the reference20.3421417351c.
Repair count alone cannot select the decoder. Binary decoding yields21/200
verified repairs and21.801c remaining error, versus23/200 and20.342c reference.
All50 originals stay unchanged. **Not selected.** No new candidate is active.

Two decoder tests pass: original expected function equivalence, fixed conditional
amplitude decision, bounds, metadata invariance, unvoiced identity and invalid
mode refusal. Independent `verify_activity_decoding.py` replay passes all216
case views,135 unique expected-reference tracks,21 encoded verified WAVs,
50 original identities,4 frozen checkpoint lineages and6 protected artifacts.
The amplitude/activity multiplication hypothesis does not produce a usable gain
under this fixed decision. The [absolute-response study](Transformer_Absolute_Response_2026-11-01.md)
is the next targeted training comparison.

Evidence root: `results/transformer_usability/2026-11-01_v30_activity_decoding`.
Commands: `evaluate_activity_decoding.py cv`, then `delivery`, then
`verify_activity_decoding.py`. Existing results are retained. Known development
and untouched confirmation require a separately locked, qualified decoder and
verified full-refit lineage first. This study does not complete MOD-03.
