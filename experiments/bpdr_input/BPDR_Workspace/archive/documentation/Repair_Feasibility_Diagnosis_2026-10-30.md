# Why usable repair remains difficult

## Completed comparison

Four model families/configurations now have the same singer/source-held-out
evaluation and actual median11/Praat output path.12 new GRU/TCN/ExtraTrees fits
are compared with4 authenticated reference fits. All200 attempts remain counted:
108 scoreable,92 intervention-construction exclusions,50 originals and8 singers
with scoreable intervention cases. Two attempted singers remain unassessed.

| Model | Raw MAE,cents | Raw all-singer gate | Verified repairs /200 | Remaining delivered MAE,cents | Original WAVs unchanged |
|---|---:|---|---:|---:|---:|
| Reference Transformer | 11.901 | Pass | 23 | 20.342 | 50/50 |
| Fresh context ExtraTrees | 12.033 | Fail | 20 | 21.217 | 50/50 |
| Temporal convolution | 11.893 | Pass | 19 | 21.281 | 50/50 |
| Bidirectional GRU | 12.751 | Fail | 18 | 21.626 | 50/50 |

The dense residual Transformer also fails adoption:21/200 and21.394c, despite
slightly better raw predictions. The reference remains retained. No new model
is activated and no usable-accuracy claim follows.

This does not prove Transformers are universally superior. It shows that these
three alternatives, under their declared configurations, do not solve this
project's problem. Neural fits have matching data, losses and sampled examples;
ExtraTrees has the same sampled examples and original/positive weights but a
different regression objective/budget. Historical active-demo scores are not
substituted into this comparison.

## Isolate prediction from realization

A separately declared diagnostic supplies privileged known correction labels
only where existing supervision is known. Unknown ramp frames remain zero.
It is **not a deployable predictor, model accuracy result, general optimum,
selection candidate or correction-coverage ceiling**. It uses no new audio and
does not change numerical/support/protection/renderer checks.

| Stage | Reference Transformer | Known-label diagnostic |
|---|---:|---:|
| Scoreable shifted cases | 108 | 108 |
| Nonidentity edits permitted | 30 | 45 |
| Identity/no permitted edit | 78 | 63 |
| Encoded verified improvements | 23 | 35 |
| Rendered attempts rolled back | 7 | 10 |
| Remaining delivered MAE,cents | 20.342 | 18.251 |

All30 reference-permitted cases are among the45 known-label-permitted cases;
15 additional cases become eligible with privileged exact corrections. For
the known-label diagnostic, candidate-reason case counts are44 weak step-shape
evidence,4 protected,15 no candidate and45 eligible. Categories are generally
nonexclusive. The diagnostic demonstrates that prediction is one limitation,
while input evidence/support and actual rendering/auditing are additional
limitations. It does not prove which tracker is right on an opposing frame.

The reference-only replay finds99/108 inputs containing at least one long
unprotected eligible region,94 with a prediction-support candidate and30 with
a nonidentity plan. An eligible region can be elsewhere than the needed edit:
99 is not a count of recoverable corruptions. Raw MAE improvements can therefore
leave the final output unchanged or worsen it.

## Preservation within a corrupted recording

Existing fixture verification checks exact samples outside the **proposed edit**.
The supplementary diagnostic asks a different controlled question: did an output
change samples outside the **known injected-error support**, including ramps?
It first verifies C equals its unshifted reference A outside that support.

| Condition | Verified outputs | Outputs with some changed samples outside injection | Singer-macro outside pitch MAE,cents |
|---|---:|---:|---:|
| Reference | 23 | 7 | 0.01668 |
| Dense residual | 21 | 7 | 0.00000 |
| GRU | 18 | 18 | 0.01981 |
| TCN | 19 | 14 | 0.01241 |

These counts are not a new tuned qualification threshold or proof of audible
harm. Some changes are very short crossfade tails. Others deserve inspection:
reference case162 changes approximately0.680 seconds outside the injected
support and has2.121c outside pitch MAE on that case's reliable reference
frames. Low aggregate drift is not independent listening approval.
The fresh tree is not included in this supplementary four-condition audit;
its separate50-original and outside-proposal waveform checks do pass.

## Verification and reproducibility

All25 focused tests pass. The tree's first CV command stopped AFTER saving all4
trained checkpoints because a threaded float64 prediction sum differed by at
most6.94e-18 in its first observed128-row reload probe. The original failure,
source and checkpoints remain retained. A separate continuation verifies exact
serial reload, unchanged sample/source/singer digests and parallel agreement to
normalized1e-14, records amendments, and evaluates the existing models. No
training is repeated and no acoustic threshold is changed.

Independent tree replay verifies216 held-out input views, all20 encoded WAVs,
all50 original identities,4 checkpoint/sampler/source lineages and6 protected
artifacts. Sequence replay verifies432 input views and37 WAVs. The diagnostic
verifier independently checks35 privileged-label WAVs, decoded hashes, every
direction/outside-support check, recovery scores and all supplementary
outside-injection sample-change counts. No activation card exists.

Evidence roots under `results/transformer_usability`:

- `2026-10-30_v28_context_tree`: declaration, original CV failure, per-fold
  verification amendments/fit receipts, CV, selection, audio evidence, integrity.
- `2026-10-30_known_label_diagnostic`: prospective prediction-only feasibility.
- `2026-10-30_known_label_wav_diagnostic`: privileged actual-WAV diagnostic and
  independent integrity receipt, explicitly NOT model performance.
- `2026-10-30_controlled_preservation`: supplementary preservation and integrity.

## Next targeted work

Stop generic architecture/readout sweeps. Test the acoustic measurements and
spatial localization implicated by these diagnostics before another full model
comparison. The current input already uses fine YIN, guarded by PESTO agreement;
PESTO's existing weighted decoding is not an argmax bug to fix.

The primary research reviewed identifies two plausible measurement comparators:
[PENN](https://arxiv.org/abs/2301.12258) estimates pitch/periodicity across music
and speech; its [official implementation](https://github.com/interactiveaudiolab/penn)
supports local CPU inference. [FCPE](https://arxiv.org/abs/2509.15140) provides
context-based monophonic pitch estimation, with an
[official CPU inference path](https://github.com/CNChTu/FCPE). These are candidates
for measurement diagnostics, not established improvements here. RMVPE's
[polyphonic-vocal task](https://arxiv.org/abs/2306.15412) is less directly matched
to BPDR's declared unaccompanied input.

Published RPA percentages must not be treated as20–30c repair accuracy:
[mir_eval's standard RPA](https://mir-eval.readthedocs.io/latest/api/melody.html)
uses half-semitone/50c tolerance. A tracker can perform well on that metric yet
miss the smaller deviations this component must measure. Benchmark actual
relative shift MAE,5c/10c tolerances, direction, voicing coverage, unchanged
regions and CPU cost on approved training-side sources. Do not interpolate
unvoiced gaps into reliable evidence or silently relabel the92 excluded cases.

Before accessing audio, fix versions, model/weight hashes, decoder, time-grid
alignment, frequency range and handling of missing frames. Keep downloads and
dependencies isolated from the active environment, with no external audio
uploads. Keep the current auditor and every acceptance limit unchanged while
diagnosing disagreements. A tracker change needs its own validated feature
contract and complete qualification before it can affect a selected pipeline.

In parallel, preflight whether the current whole-region acoustic step fit loses
local evidence under natural drift/vibrato. A fixed sliding local-context
measurement with conservative agreement could be examined on training sources;
it is not implemented or qualified for selection by this suggestion.
Preserve thresholds, real support, original controls and all old failed results.

The recording-coach, multiple-take selection and readiness extensions in
[model choice and input usability](Model_Choice_and_Input_Usability_2026-10-30.md)
remain scoped to audio preparation. Notes, key, tempo, genre and harmony remain
other members' responsibility. No UI feature displaces the outstanding main
repair requirement, and no fresh confirmation or musical review has occurred.
