# Gate E reference-cue ceiling - declaration, 17 October 2026

**Declared, not executed.** Machine-readable protocol: [declaration.json](../runs/reference_cue_ceiling/declaration.json) (SHA-256 `3b4396a7d406ea9313605cb6cbc696314bf82bb6a13f7f4821466f269e3f0fcb`). It follows the [failed runtime-cue gate](Gate_E_Runtime_Cue_Diagnosis_2026-10-17.md): a single detuned view gave 8.6% correction gain (Jamendo 9.8%) and a net loss on unshifted audio.

## Question

Does a tuning reference that the injected detuning cannot move, together with non-quantized pitch, raise the achievable repair above that single-view ceiling?

## An honest caveat before spending effort

A track-level mixture reference removes the singer-versus-instrument tuning offset. It does **not** remove the singer's note-by-note scatter around the grid (about 15 cents Jamendo, 21 Vocadito in the earlier scratch check), which is as large as the injected 10-30 cent shifts, and part of that scatter is the 33.3-cent quantization of the stored pitch column. So this step is a **ceiling measurement first**, with no neural fit, and it stops early if the ceiling is low.

## Data

- **Jamendo:** 70 training crops, 23 artist proxies (918 variants). The full-track mixture mp3 exists locally for all 70.
- **Vocadito:** 14 sources, 11 singers (77 variants). A cappella, so no mixture reference. Reported separately and never gated.
- Variants are not independent; every split is grouped by artist or singer.

## References compared

| Id | Reference | Purpose |
| --- | --- | --- |
| ref0 | own-view offset from stored quantized PESTO | existing baseline |
| ref1 | own-view offset from CPU YIN on the existing A/C waveforms | isolates quantization |
| ref2 | one tuning scalar per track from the full mixture, computed once | the unshifted reference |
| ref3 | same estimator on the crop window only | sensitivity check |

The pipeline `key.json` is excluded: it is a tonic pitch class with no sub-semitone content, derived from the vocal itself.

## Stages and gates

- **R0 (ceiling, no neural fit):** off-reference scatter of unshifted originals for each reference; the same artist-grouped frame probe and 0.5/0.25/0.25 region weights as before, with fine-pitch cues relative to each reference; artist-bootstrap intervals.
- **R1 (only if R0 passes):** a BPDR-local adapter giving the reference-relative cue to all three arms; a training-only sanity fit (at most 150 updates, correction training loss at most 0.90 of zero output); then the matched 3x300 comparison on the unchanged 304/88 development artifacts, requiring gains above 3x the checkpoint noise (about 0.02 cents) and paired response gain above 0.1.
- **R2 (if R0 fails):** stop and report blind and reference-assisted detune repair as information-limited; update plan, tracker and notebooks.

**Proposed R0 pass thresholds (Jamendo):** correction gain at least 25%, objective gain at least 10%, and an artist-bootstrap lower bound above zero for the improvement over ref0. These are my proposals; you may change them before R0 runs, not after.

## Leakage rules

No injected shift, target, private mask or paired original enters any reference or cue. The mixture contains the unshifted original vocal, as a real input would, and the reference is one track-level scalar, so it cannot encode where or by how much a crop was shifted.

## Interface condition

A reference-assisted model needs a track-level tuning scalar from the full mixture. The component receives only the separated vocal crop today. Any delivered-repair claim needs agreement with the melody-component owner (IT23439078) on the hand-off contract.

## Constraints

Inside `experiments/bpdr_input` only; no team or shared-pipeline file edited or executed; no development, PP1 or holdout data in R0; CPU analysis of existing waveforms only (no new separation, GAME, crops or shifted audio); no model adoption. E/F open, D/J4 deferred, research-ready zero.
