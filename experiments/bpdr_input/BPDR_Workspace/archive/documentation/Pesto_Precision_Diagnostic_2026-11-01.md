# PESTO decoder precision diagnostic

Declared 1 November 2026 before testing higher-precision decoding. Status:
all90 native tracks reproduce the old float32 F0 and confidence exactly.
Independent replay verifies90 native tracks and60 candidate audits. The separate
NumPy/Torch decoders agree on all strict masks, with maximum F0 difference
6.82e-13 Hz. Two decoder tests pass.

All21 observed opposing coarse frames across60 reconstructed candidates have
absolute shifts0.000619–0.000929 cents. Requested shifts are much larger and
fine direction passes. These magnitudes motivate an arithmetic diagnosis,
not permission to round them to zero or relax the strict sign requirement.

For all60 candidates and their corresponding source C recordings, rerun the
same frozen float32 PESTO model and require exact native F0/confidence control
reproduction. Save the complete384-bin distributions. Preserve argmax, the same
five-bin local window and the same posterior weights. Compute only the weighted
pitch reduction and frequency conversion in float64. Preserve native grid,
nearest alignment, voicedness masks, YIN, requested curves, sample supports and
every existing numerical audit threshold, including strict per-frame direction.
No learned weights, labels, input permission, waveform or original audit changes.

Independently rerun model inference and decode the same distributions with a
separate Torch float64 implementation. Check the NumPy implementation's saved
tracks, all60 audits and strict opposing-frame masks against that implementation.
Report sign disagreements explicitly; a small arithmetic agreement bound is not
an acceptance tolerance. A changed number of technical audit passes is not a
verified delivered-WAV improvement, model selection or evidence of intended-pitch
accuracy. No rejected candidate is approved as delivered output by this study.

Run `diagnose_pesto_precision.py run`, then `verify` in the existing environment.
Results: `results/transformer_usability/2026-11-01_pesto_precision`.
No development/fresh-confirmation audio, main notebook, selected model/export,
shared demo or sibling component changes. A useful result requires a separately
declared complete pipeline comparison before any adoption.


## Observed pre-encoding result

For the exact frozen candidate waveforms, the float64 coarse decoder changes
post-render audit passes from23/30 to27/30 on the reference arm and20/30 to27/30
on V32. The original full200/108/92 denominator is retained. These are technical
pre-encoding audit counts, NOT27 verified delivered repairs, a trained-model
accuracy gain or a usable model. Some tiny direction vetoes are removed and
others introduced. Wait for the separate implementation's frame-level sign
comparison before deciding whether this arithmetic change is stable enough for
a further complete delivered-WAV experiment. No source or candidate waveform,
requested curve, reliable/voiced mask, threshold or learned parameter changes.


Independent verification completes with zero strict-sign implementation
disagreements and all six protected artifacts unchanged. The floating arithmetic
check is sufficiently reproducible for a separately declared full delivery
comparison; it does not authorize a model or decoded WAV. That comparison is
[V33 precision-auditor WAV evaluation](Precision_Auditor_WAV_Study_2026-11-01.md),
using only the frozen reference Transformer and no new training.


V33 subsequently completes full encoded-WAV evaluation and independent replay:
27/200 verified delivered repairs and19.824705c remaining error versus23/200
and20.342142c reference,all50 originals unchanged. This is a pipeline engineering
improvement with the same model predictions. It remains inactive and does not
pass the complete practical usability requirements. See the V33 report for
selection, complete checks and remaining development/confirmation/review work.
