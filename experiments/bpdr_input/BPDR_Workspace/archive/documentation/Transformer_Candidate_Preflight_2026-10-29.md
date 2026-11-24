# Candidate-level Transformer data preflight

Status: feature/label preflight complete. No candidate-level model fitted.
MOD-03 remains active and usability remains unproved.

The fixed ensemble and R3 renderer both lost their actual-WAV comparisons.
The next model direction is one learned decision per input-measured region,
while retaining the existing dense Transformer outside measured regions.
This targets fragmented frame predictions without padding runtime support.

## Input-only geometry

Generate regions only from nonzero existing acoustic-step evidence. Keep the
measured interval exactly and require43 frames (0.86 seconds); never enlarge
it to pass the minimum. Candidate context uses the existing51x8 broad/step
tokens plus an input-derived pooling mask. Source identity, labels, reference
recording and intervention interval do not enter this feature transform.
Boundary-metadata perturbations leave regions and contexts unchanged.

Training labels are separate files. Require at least50% known frames. An
unchanged candidate has only known-zero labels; a positive candidate requires
at least50% of its region to have known nonzero labels,95% purity among known
labels and one constant correction within30 cents. Ambiguous/missing cases
remain explicitly unassessed. These are controlled recovery labels, not human
judgments about intended musical pitch.

## Observed data

| Quantity | Count |
|---|---:|
| Input cases:50 originals +373 corrupted variants | 423 |
| Cases with no qualifying measured candidate | 224 |
| Positive labelled candidates | 177 |
| Negative labelled candidates | 8 |
| Unassessed candidates | 14 |

The data still comes from the same50 originals/10 singers. Every descendant
must inherit its original singer/source fold. Original controls are included
once, not repeated for each derivative. These counts do not establish accuracy.

Training-side overlap analysis finds39 of108 baseline scoreable cases have no
candidate overlap with the known shifted plateau. Male7 overlap averages25%
and female7 overlap50%. Therefore a candidate-only predictor would discard
coverage already available to the dense model. Retain a frozen dense fallback;
do not drop uncovered cases or shrink the200-attempt evaluation denominator.
Candidate overlap itself is neither permission nor verified repair.

A prospectively fixed negative-control check reuses only the existing CLEAN
feature-augmentation branch at seeds339/340/341 (linear drift and vibrato),
yielding96 synthetic feature controls. It produces ZERO additional qualifying
candidates. Retain this negative result; do not lower fitter thresholds or
invent negative candidates. No real WAV or independent singer is added.

## Next implementation constraints

Prototype a small candidate head on frozen Transformer features with a dense
fallback, using the8 real negative candidates plus separately identified
unchanged-original training anchors. The sparse negative evidence must remain
visible. Separate head fitting, per-singer raw/original gates, actual-WAV
selection, conditional full refits, known development and untouched confirmation.
No candidate geometry may be derived from labels at inference. Preserve all
30-cent,0.86-second,25%,5-cent,protection and encoded-waveform checks.

Three tests pass for input-metadata invariance, label purity/missingness,
unvoiced identity and short-candidate refusal. No checkpoint, active model
card, shared demo or export changes in this preflight.

- [Prospective declaration](../results/transformer_usability/2026-10-29_v21_candidate_data/declaration.json)
- [Data preflight](../results/transformer_usability/2026-10-29_v21_candidate_data/preflight.json)
- [Baseline overlap diagnostic](../results/transformer_usability/2026-10-29_v21_candidate_data/baseline_candidate_overlap.json)
- [Negative-control result](../results/transformer_usability/2026-10-29_v21_candidate_data/negative_controls/preflight.json)
- [Input-only geometry](../source/bpdr_input/modeling/step_candidates.py)
- [Training-only labels](../source/bpdr_input/experiments/learning/candidate_labels.py)
- [Preflight tool](../tools/prepare_candidate_transformer_data.py)
