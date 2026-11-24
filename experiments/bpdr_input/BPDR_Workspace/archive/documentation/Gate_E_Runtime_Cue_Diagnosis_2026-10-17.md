# Gate E runtime-cue diagnosis - 17 October 2026

> **Correction, 17 October (later):** this conclusion is withdrawn. The probe tested an absolute tuning-grid cue on pitch that cannot resolve the shifts (PESTO mir-1k_g7 activations are one-hot at 33.3-cent bins). The proposal's signal is the within-note step, which correlates 0.57 with the shift on Jamendo when measured with fine pitch. See the correction entry in `PP1_Phase_Execution_Plan.md`.


**Result: the declared gate failed, so no input-adapter fit or 900-update comparison was run.** The diagnosis asked whether the correction target is predictable from cues available at runtime. Answer on this data: only weakly, and not well enough to pay for the cost it adds on unshifted audio.

## Why the revised-objective comparison (14.38) changed nothing

- In all three arms the training correction loss equals the loss of outputting zero (ratio 0.9999-1.0000 against the per-update zero baseline from the saved schedule) for all 300 updates. The model collapsed to zero output from the start; this is underfitting, not a generalisation failure.
- Checkpoint-to-checkpoint development error varies by about 0.02 cents within an arm (range 0.020-0.027). The 14.38 differences (+0.0014 BPDR, +0.0030 augmentation-only, -0.0119 frame-only) are inside that noise, and the best checkpoint of any arm beats zero correction by only 0.006-0.015 cents. 14.38 is a null result.
- Loss weighting cannot fix this: if inputs do not reveal the sign of the shift, an MAE objective on symmetric targets is minimised by output 0 for any region weights.

## Declared diagnosis (training records only)

Run `runs/runtime_cue_diagnosis/e33b5a24843749eb82459bd0d6f4cfa3` (declaration written before analysis; source hashes recorded). A frame-level gradient-boosted probe (absolute-error loss, artist-grouped 5-fold CV, the 14.38 region weights 0.5/0.25/0.25) used cues from the detuned view's own pitch track only: distance from the view's semitone grid after removing its tuning offset, at three window widths, plus short-window pitch steps. A control used the base scalar features only.

| | Correction-region error (cents) | Zero baseline | Correction gain | Original/outside error added | Objective gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base features only (control) | 20.805 | 20.955 | 0.7% | 0.66 / 0.64 | -8.7% |
| With tuning cue, all | 19.153 | 20.955 | 8.6% | 1.06 / 1.00 | -6.2% |
| With tuning cue, Jamendo | 18.217 | 20.196 | 9.8% | 1.08 / 0.99 | -5.7% |
| With tuning cue, Vocadito | 30.318 | 30.000 | -1.1% | 0.85 / 1.09 | -10.4% |

Declared pass rule: correction gain >= 10% overall and for Jamendo, and objective gain >= 5%. **Fail** (8.6%, 9.8%, -6.2%).

Supporting facts:
- The pitch column is absolute cents (SD about 600), so 30 cents is 0.05 SD and 10 cents 0.017 SD. It is quantized to 33.3-cent steps: for a 10-cent shift 52-55% of correction frames have an identical pitch column, for 20 cents 24%, for 30 cents 4%.
- 58% of the 20 nearest other-artist neighbours of a shifted frame in cue space are unshifted originals (target 0); the median neighbour target spread is 0. A shifted frame is mostly indistinguishable from ordinary singing of another artist.
- Earlier one-feature check (scratch, not of record): grid deviation of the original plateau has SD about 15 cents (Jamendo) and 21 (Vocadito), comparable to the injected 10-30 cent shifts. That natural off-grid scatter limits any blind absolute-tuning detector, however it is trained.

## What this means

No further fit of this model family on these inputs is justified: the three ingredients (loss weighting, longer budget, more variants) cannot create information the single detuned view does not carry. Per the plan, the diagnosis-supported next step is a change of framing, decided before any fit:

1. **Add a tuning reference that survives detuning**, e.g. the unshifted accompaniment/mixture tuning or key for Jamendo (the vocal is detuned, the backing is not). This needs audio processing and an owner decision on the component interface; Vocadito (a cappella) has no such reference.
2. **Redefine the target against a grid or scale** instead of the original performance. This is trivially predictable but also removes intended expressive intonation.
3. **Report the negative result**: blind absolute detune repair is information-limited on single-voice input. The plan allows an inconclusive or negative outcome.

Not done in this step: the per-term gradient-flow measurement (moot once inputs fail to separate the classes), any development data, extraction, shared-pipeline execution, or model adoption. D/J4 deferred, E/F open, research-ready zero.

## Evidence

- [declaration.json](../runs/runtime_cue_diagnosis/e33b5a24843749eb82459bd0d6f4cfa3/declaration.json)
- [evidence.json](../runs/runtime_cue_diagnosis/e33b5a24843749eb82459bd0d6f4cfa3/evidence.json)
- Code: `src/bpdr_input/cue_features.py`, `src/bpdr_input/runtime_cue_diagnosis.py` (not yet covered by unit tests; plan, tracker and notebooks are not updated for this step).
