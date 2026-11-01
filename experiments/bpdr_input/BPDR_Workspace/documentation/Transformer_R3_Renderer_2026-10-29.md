# Frozen Transformer with Rubber Band R3 rendering

Status: training-side actual-WAV comparison complete; R3 rejected. No activation.

The ensemble lost usable coverage. Earlier single-model repairs frequently
failed the post-render coarse direction check despite small fine-pitch errors.
This study isolates waveform generation while keeping the Transformer,
eligibility and numerical/output gates frozen.

The [official Rubber Band project](https://github.com/breakfastquay/rubberband)
describes R3 as suitable for vocals and smooth pitch changes. This motivates
a test, not an assumed quality or accuracy improvement. The official Windows
4.0.0 CLI is a local research dependency; archive, executable and DLL hashes
are pinned, with supplied license files retained. No audio is uploaded.

Use offline R3 at time ratio1 on each already permitted constant local
correction. Disable automatic clipping gain changes; reject nonfinite/clipped
output. Copy samples outside the exact existing support and use the existing
80ms envelope. Require exact sample count/rate, with no padding or trimming.
Dynamic pitch-map mode is deliberately excluded because the official CLI
documents inexact duration in that mode. No formant option or perceptual claim.

Compare constant R3 against constant Praat and the existing median11/Praat
reference on the same200 attempted/108 scoreable interventions and50 originals.
All30-cent,0.86-second,25%,5-cent,protection and fine/coarse encoded-WAV guards
remain. Selection uses original identity then singer-macro delivered error;
ties retain current Praat. New renderer development access is conditional on
better training-side delivery. Any development evaluation remains known data;
reserved CSD/male10/male11 audio stays unopened.

Two actual renderer tests pass: +/-20-cent tone shifts with exact outside
samples, unchanged identity, refusal of variable/short/over-limit curves and
refusal of altered dependency binaries. Version command confirms4.0.0. No
native code, live UI, model export or shared-demo change occurs.

- [Dependency provenance](../results/transformer_usability/dependencies/rubberband_4.0.0/dependency_receipt.json)
- [Prospective declaration](../results/transformer_usability/2026-10-29_v20_r3_renderer/declaration.json)
- [Pinned renderer](../source/bpdr_input/repair/rubberband_renderer.py)
- [Study runner](../tools/run_rubberband_renderer_study.py)

## Completed result

| Renderer/curve | Verified gains /200 | Delivered after MAE,cents | Original identity |
|---|---:|---:|---:|
| Existing Praat/median11 | 23 | 20.342 | 50/50 |
| Praat/constant | 22 | 20.939 | 50/50 |
| R3/constant | 22 | 20.998 | 50/50 |

All200 attempts and108 scoreable interventions remain. R3 attempts30 renders;
22 pass encoded verification and8 roll back. Six rollbacks include coarse
direction failures and two coarse median errors; their fine median errors are
0.319–0.818 cents. These observations do not prove the coarse tracker is wrong
and do not authorize relaxing its guard. The constant-curve comparison loses
slightly to constant Praat and does not beat the current median11/Praat path.

The training-side lock retains Praat/median11. The actual development command
prints `R3_NOT_SELECTED` and does not open development audio. No model refit,
new independent evaluation, listener approval or live activation occurred.
Seventeen combined ensemble/R3/joint/curve/runtime tests pass. Independent
verification checks all22 encoded R3 outputs, hashes, decoded waveforms,
unchanged outside samples,all50 original controls and6 protected artifacts.
The native dependency stays a local research artifact with its license.

- [Selection](../results/transformer_usability/2026-10-29_v20_r3_renderer/locked_selection.json)
- [Actual-WAV report](../results/transformer_usability/2026-10-29_v20_r3_renderer/arms/rubberband_r3/training/span_median/REPORT.md)
- [Integrity receipt](../results/transformer_usability/2026-10-29_v20_r3_renderer/integrity_receipt.json)
