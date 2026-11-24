# Short-phrase training: construction and comparison complete

Updated 2 November 2026. Engineering controlled recovery only. No usable-model,
natural intended-pitch, research-validation or activation claim.

**Completed outcome:** all12 fits, checkpoint replay and actual-WAV comparison
finish. Candidate rejected:4 target improvements,4 target harms,2 changed originals
and worse overall delivered error. See the [complete model result](Short_Phrase_Model_Result_2026-11-02.md).
The execution description below retains the declared protocol; earlier running/next
language is superseded by that completed result.

## Completed data construction

V40 retained the existing Vocadito singer/source partitions. Only the 21 training
recordings from 14 singers were acoustically accessed. The first at most ten seconds
of each source, highest-energy native channel and fixed 22,050 Hz PCM24 conversion
were used, without padding, silence removal, normalization or outcome replacements.
No human annotation arrays were loaded. Source/crop mappings and hashes are retained.

| Disposition | Attempts |
|---|---:|
| Valid audited supervised pairs | 44 |
| Insufficient single reliable region with 70 ms context | 176 |
| Physical-intervention audit failed | 84 |
| Invalid source preprocessing | 32 |
| All planned attempts | 336 |

Valid pairs include eight 0.36 s, twenty-four 0.50 s and twelve 0.70 s deviations,
from six singers: S5, S7, S22, S23, S28 and S29. No 0.24 s attempt passed. S29 has
twenty valid variants from two recordings, not twenty independent people.

Nineteen original working recordings were created. Vocadito13 and18 were refused:
resampled peaks are1.0092002153 and1.0032759905 respectively. These are resampled
peak observations; the report does not claim both native recordings are clipped.
Their 32 proposed variants remain source-preprocessing exclusions. Raw source
bytes remain unchanged and no automatic normalization supplied replacement data.

The labels are the negative nominal physical shift on the common audited interior,
with zero targets on reliable unchanged outside frames. Ramps/guards are excluded.
Clean originals supply reliable-frame zero examples. References, targets, source
identities and injected supports are not inference inputs. Observed original pitch
is a controlled-recovery target, not proof of the performer's natural intended note.

Independent replay recreated all nineteen valid working originals, reinferred
**1,134 original/corrupted feature arrays**, rechecked **128 decoded physical-shift
audits**, verified **44 supervised label files**, reproduced176 geometry exclusions
and recorded the two preprocessing refusals. All336 attempts remain retained.

Evidence: `results/transformer_usability/2026-11-02_v40_short_phrase_data/`:
`source_reservation.json`, `preparation_declaration.json`, `train_pairs.json`,
`train_originals.json`, `input_hashes.json` and `integrity_receipt.json`.
Data construction does not make these examples research-training eligible;
project-specific independent reference-suitability review remains pending.

## New opt-in model and permission modules

`modeling/short_phrase_transformer.py` uses the same small genuine Transformer
family (two layers, four heads, hidden32, feed-forward96), but replaces the long-step
cue with eight continuous acoustic channels. Its81 tokens span four seconds with
dense20 ms sampling near the center and80 ms sampling farther away. Positional
encoding follows the actual physical offsets rather than a fictitious uniform grid.
Short-scale pitch contrasts use13/31/61-frame windows. No pitch-class snapping,
intended notes, key, final melody or harmony representation is introduced.

Five feature/interface checks pass: boundary/nonacoustic-channel independence,
transposition/gain consistency, center/padding alignment, finite unvoiced identity
and malformed-input refusal. These are contract checks, not accuracy evidence.

`repair/short_phrase_policy.py` is a separate experimental input-only policy:
12/6-cent signed hysteresis, three-frame median, real support at least0.24 s and
four-frame protection around acoustic/reliability transitions. The whole candidate
is vetoed on user protection. No padding or gap filling occurs. Existing fixture
limits of30 cents,25% duration and5-cent disagreement remain. Four policy checks
pass, including original default retention, protection veto, over-limit/disagreement
refusal and no short/unreliable support padding. Production loaders, the old policy,
the shared demo selection and other members' components are unchanged.

Boundary-view disagreement is not calibrated confidence. Identical token matrices
across boundary metadata views establish input invariance only. These modules do
not implement or validate the complete proposed BPDR objective comparison.

## V41 declared comparison (now completed)

Four singer-separated folds compare:

1. The existing frozen joint Transformer (different training history; reference only).
2. The old joint representation retrained on the new data.
3. The revised short-phrase Transformer.
4. ExtraTrees on exactly the revised Transformer's81x8 tokens plus padding flags.

Each new Transformer fit has800 updates,36 sampled rows per batch and seed339.
Every batch balances positive C interiors, unchanged C outside and original
preservation frames. Both neural arms use the same loss and sample trace; ExtraTrees
uses the same rows/targets but its own regression loss. This is not equal compute,
an optimal hyperparameter search or a test of Transformer architecture alone.
Twelve new fits were completed across the four folds. Completed fits have hash-bound
checkpoints, exact group/source lists and sampler traces. Partial training losses
are not reported as held-out accuracy, and no seed is selected using these outcomes.

The first completed revised-Transformer fit took about20 seconds; matched ExtraTrees
took about453 seconds on this machine. That is one fold's fit timing, not a general
speed benchmark or evidence of superior accuracy.

Root: `results/transformer_usability/2026-11-02_v41_short_phrase_models/`.
The full comparison and checkpoint replay are now complete; no arm qualifies. Development, PP1-evaluation and
final-holdout audio remain outside this work.

## Actual-WAV evaluation: declared and now completed

V42's declaration was written before inspecting final raw prediction results.
Every model will use the same input-only short policy,20 ms renderer ramps,70 ms
audit guard and unchanged ten-frame/error/direction/coverage limits. No target or
reference support enters proposal generation. Original controls, all336 attempts,
construction exclusions, decoded PCM24 audits and separate A-reference scoring
are retained. Only a post-encoding verified output with actual error reduction
can count as a beneficial repair. Failed candidates roll back to the input.

This prospective protocol is at
`results/transformer_usability/2026-11-02_v42_short_delivery/declaration.json`.
The seven-page current-system PDF is generated from the completed verified results. The [short-phrase master execution plan](Short_Phrase_Transformer_Plan_2026-11-02.md)
and independent musical-review/collection requirements remain active.
