# Broad/local Transformer and difficult-original supervision

**Status: eight fits and actual-WAV comparison completed; both new variants rejected. MOD-03 remains incomplete. No activation.**

The previous localized-context model reduced raw correction error but increased
correction requests on unchanged vocals. This experiment supplies both context
scales and tests whether targeted original-frame sampling addresses that
preservation failure. Neither new condition has been qualified for the demo.

## Model and matched sampling comparison

The input has thirteen channels per token: five broad acoustic-context channels,
five channels normalized to the current audio-derived pitch/voicing region and
three acoustic-step cues. Broad attention remains accessible. The local voiced
channel indicates membership in the current region. No reference recording,
target interval, source identity, chromatic grid or provisional-boundary
metadata enters inference. Boundary invariance is by input construction.

The two new arms share this model, initialization, all loss weights, optimizer,
36-window batch, 1,200-update budget, seed339 and four singer/source folds:

| Arm | Extra original-anchor frame sampling |
|---|---|
| `multiscale_uniform` | Existing source-balanced uniform reliable-frame sampling |
| `multiscale_hard` | Half the draws select reliable frames at or above the same original's 80th percentile of absolute local pitch residual |

The difficult-frame score uses the known original's input features only.
Every sampled original target remains zero. Source selection and paired A/C
draws retain their original RNG sequence; a separate RNG changes only the
extra original frame. Saved reference traces must reproduce the existing
baseline sampler, and paired traces must match between the new arms.
Unavailable reliable anchors remain recorded rather than fabricated.

The old eight-channel model is a frozen reference. The new projection has 160
additional parameters; hidden width32, two Transformer layers, four heads,
feed-forward width96 and 51 tokens remain fixed. The two new arms have the
same parameter count. Comparisons against the old model are engineering
comparisons, not an isolated proof of representation or BPDR-loss superiority.

All200 attempted /108 scoreable interventions, all50 working originals and
the original singer/source partitions remain unchanged. Reusing four reference
checkpoints does not add four newly fitted models or additional people.

## Qualification

### Completed outcome

| Condition | Raw MAE (cents) | Raw original mean (cents) | Verified gains / all attempts | Delivered after MAE (cents) | Original WAV identity |
|---|---:|---:|---:|---:|---:|
| Frozen broad-context reference | 11.901 | 0.789 | 23/200 | 20.342 | 50/50 |
| Broad/local, uniform originals | 11.793 | 1.276 | 21/200 | 21.803 | 50/50 |
| Broad/local, difficult originals | 10.988 | 1.357 | 23/200 | 21.039 | 50/50 |

All conditions retain 108 scoreable interventions out of 200 attempts. The new
variants fail the all-singer prediction gate: female1 raw-original mean is
4.504/5.225 cents, above the fixed 3-cent limit. Difficult-original sampling
improves correction MAE but does not fix preservation and does not improve
delivered error. The saved selection retains `matched_baseline`; no new full
refits or development evaluation are justified. These are training-side
held-out outcomes, not fresh confirmation or natural intended-pitch accuracy.

The next declared candidate starts from each corresponding broad-context
checkpoint, initially zeros the added local-input weights and fine-tunes with
negative-example parent preservation. It is a separate experiment with additional
training updates; it does not overwrite these failed conditions.

- [CV metrics](../results/transformer_usability/2026-10-29_v14_multiscale/cross_validation.json)
- [Actual-WAV selection](../results/transformer_usability/2026-10-29_v14_multiscale/locked_selection.json)

Selection first prefers the all-singer prediction gate, then original WAV
identity, then minimum singer-averaged delivered error under the existing
median11/Praat condition. All numerical, voicing, protected-region, affected
duration and encoded-WAV audits remain fixed. A raw metric alone cannot
activate a model. Three full refits at seeds339/340/341 are allowed only for a
qualified newly selected arm; default339 stays fixed for development.

The development checker validates actual full-data training groups/sources,
all three seeds, update counts, checkpoint hashes and the model/policy lock.
It dispatches to the explicit multiscale loader, which rejects incompatible
model/feature versions. Replays receive a separate folder and remain known
development evidence. CSD and male10/male11 confirmation audio stays unopened
until a locked candidate passes development usability.

Three new tests passed before fitting: broad context remains accessible while
local channels stay invariant to a neighboring-note change, difficult-original
selection excludes unreliable frames, and a nonzero checkpoint survives reload
and different inference batch sizes. These are technical checks, not accuracy
or listening evidence.

- [Prospective declaration](../results/transformer_usability/2026-10-29_v14_multiscale/declaration.json)
- [Model and input contract](../source/bpdr_input/modeling/multiscale_transformer.py)
- [Training and sampling](../source/bpdr_input/experiments/learning/multiscale_transformer_training.py)
- [Execution command](../tools/train_multiscale_transformer.py)

```powershell
& ..\.venv\Scripts\python.exe -X utf8 -B tools\train_multiscale_transformer.py cv
& ..\.venv\Scripts\python.exe -X utf8 -B tools\train_multiscale_transformer.py delivery
& ..\.venv\Scripts\python.exe -X utf8 -B tools\train_multiscale_transformer.py refit
& ..\.venv\Scripts\python.exe -X utf8 -B tools\evaluate_response_transformer.py --run-root results\transformer_usability\2026-10-29_v14_multiscale
```

Each later stage requires the preceding receipts and gates. The active
ExtraTrees export, historical negative results, user notebook edits, shared
demo and sibling components remain unchanged.
