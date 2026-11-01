# Acoustic measurement diagnostic

## Result

The tested FCPE checkpoint does **not** improve the existing fine-pitch
measurement on this cohort. Retain YIN and the current PESTO checks. No
Transformer, auditor, permission limit, active export or shared demo changes.

| Tracker | Singer-macro relative-shift MAE,cents | Within5c of injected shift | Within10c | Observed reference frames |
|---|---:|---:|---:|---:|
| Current fine YIN | 0.860 | 100.00% | 100.00% | 100% |
| Cached PESTO | 10.711 | 44.92% | 46.17% | 100% |
| FCPE0.0.4 bundled checkpoint | 91.064 | 86.37% | 93.31% | 100% |

These are **relative C-minus-A measurements of controlled interventions**,
not absolute pitch accuracy, natural intended-note accuracy, learned correction
accuracy or delivered repair. The same200 attempts,108 scoreable references
and92 construction exclusions remain. Eight singers have scoreable interventions;
male2/male4 remain unassessed. All50 original recordings are measured separately.

Large FCPE errors remain included:15 cases have case-mean error above100c.
They concentrate in female-tagged source groups in this particular cohort;
this does not establish a demographic model claim. For example, female3's
macro error is302.257c and female7's366.655c, despite high within-tolerance
percentages elsewhere. Do not select a median or omit these cases to make the
checkpoint appear better. Original/outside-region relative drift averages
0.0344c for FCPE and0c for the two cached/current comparators.

The corpus and scoreable masks were already audited using YIN/PESTO. This
selection limits the comparison: it does not prove YIN's absolute accuracy or
universal superiority. Paired A/C differencing also uses information unavailable
to a runtime predictor that receives C alone. Accurate pairwise measurement
does not establish that intended correction can be inferred from one recording.

## Frozen setup and execution

The source, dependencies, checkpoint, decoder, grid mapping and missing-frame
rules were bound before measurement. The owner paused the first real run on
30 October; that process was interrupted and confirmed terminal. On1 November,
the saved declaration/synthetic evidence was checked and the real stage resumed.
No existing result was overwritten or completed fitting repeated.

FCPE is an official pretrained model used for diagnostic inference, not a model
trained or fine-tuned here. Its bundled configuration identifies `yx2/train`
without adequate singer identity provenance to establish pretraining disjointness
from this corpus. No research-superiority or fresh-final-evaluation claim follows.

Five public hash-verified wheels are extracted into the ignored experiment
sidecar: torchfcpe0.0.4,einops0.8.2,local-attention1.11.2,
hyper-connections0.4.11 andtorch-einops-utils0.2.0. Existing Torch,
torchaudio and NumPy packages are inherited; the active environment and its
requirements are not modified. No user audio is uploaded externally.

The adapter uses the official native mel/model path with fixed weighted
`local_argmax` decoding and0.006 threshold. Analysis is resampled to16kHz;
native10ms frame centers start at5ms, derived from the STFT padding. Nearest
actual-frame mapping uses earlier ties, no extrapolation, no unvoiced-gap
interpolation and no artificial terminal-frame repetition. Frequencies outside
50–1500Hz remain missing, rather than clamped. The working waveform is unchanged.

All24 analytic harmonic-shift cases (110/220/440Hz,global and local
-30/-20/+20/+30c changes) have100% within5c on their scored regions. All200
native silence frames are unvoiced. Those generated controls fail to expose
the severe recording-specific errors, demonstrating why they cannot replace
real-recording evaluation.

## Verification

Three alignment tests pass; all28 focused tests across the current experimental
models and guarded runtime pass. Independent replay checks all158 cached source
tracks (50 originals +108 altered recordings), with maximum replay difference
0Hz. It verifies frame mapping, missingness, all108 case statistics, singer
aggregates,200/108/92 denominators,6 protected artifacts and absent activation.
This is verification of this diagnostic, not validation of a usable model.

Commands from the canonical workspace environment:

- `prepare_measurement_packages.py`: sidecar acquisition and hash receipt.
- `evaluate_fcpe_measurement.py synthetic`, then `real`: frozen measurement.
- `verify_fcpe_measurements.py`: independent replay.

Evidence is in `results/transformer_usability/2026-10-30_v29_fcpe_measurement`;
dependencies are in `results/transformer_usability/dependencies/fcpe_0_0_4`.
The dated root is retained from the original declaration rather than renamed.

## Next mechanism test

A privileged fixed201-frame context diagnostic with the unchanged step fitter
permits48/108 known-label cases versus45/108 whole-recording cases:3 additions,
0 losses. It creates no qualifying cue in97 paired original/crop queries, but
the global and cropped search extents differ; this is not a measured false-edit
rate. The crop center uses the injected interval, so this is NOT a runtime
feature or model improvement. Keep it separate from deployable results.

Neither a pitch-tracker swap nor context cropping explains most of the learned
correction gap. Next test one code-grounded mechanism: the existing amplitude
head has positive-only correction supervision, while activity has binary BCE;
current inference multiplies the amplitude by activity, potentially shrinking
the conditional correction. The prospectively declared V30 compares that
expected correction with a fixed binary decision at logit0, retaining the same
bounded amplitude. All scope, support, activation, uncertainty, preservation and
encoded-WAV checks stay fixed. This is a single engineering decision ablation,
not a calibrated-confidence claim or an open-ended decoder search.

V30 completes and is rejected:21/200 verified repairs and21.801c remaining
delivered error, worse than23/200 and20.342c reference, with all50 originals
unchanged. Independent216-view/21-WAV replay passes. The local-context counts
also reproduce independently for all108 cases and97 original/crop queries.
Next is the [matched absolute-response loss study](Transformer_Absolute_Response_2026-11-01.md),
not another tracker swap or decoder sweep. No adoption, development/confirmation
access or new feature release occurs.
