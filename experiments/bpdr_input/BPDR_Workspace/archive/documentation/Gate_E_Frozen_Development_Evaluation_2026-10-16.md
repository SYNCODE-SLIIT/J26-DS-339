# Gate E frozen-model development evaluation - 16 October 2026

The declared **frozen-model evaluation is complete on 304 development variants and all 88 original no-change controls**. It uses the historically selected Windows CPU exports: augmentation-only update 25, frame-only update 50 and BPDR update 300. All weights, original training-only normalization, input contracts and checkpoint choices stay fixed. This is a post-selection development diagnostic, not independent held-out validation.

**No repair benefit is demonstrated.** Dataset-balanced error on injected-error plateaus is **24.603240741 cents for zero correction**, **24.605118277 for augmentation-only**, **24.605335499 for frame-only**, and **24.603258671 for BPDR**. All three observed combined means are slightly worse than zero correction. These tiny differences are reported at their actual scale; no statistical-significance or accuracy-percentage claim is made.

On human-reference Vocadito, zero correction has 30.000000000-cent plateau error and BPDR 30.000051313 cents. On model-estimated Jamendo, zero has 19.206481481 cents and BPDR 19.206466028 cents, a very small difference without demonstrated useful response. BPDR dataset-balanced plateau response gain is **0.00000012223**, where zero response is 0 and ideal injected-error reversal is 1. Its original-to-shift change therefore remains effectively zero. Small alteration and boundary sensitivity do not establish repair ability when predictions barely respond.

All original controls remain included: 14 Vocadito and 74 Jamendo recordings/crops. vocadito_9_p01 has no original runtime voiced frames, so its preservation metrics are unavailable, not zero; 13 Vocadito and all 74 Jamendo controls contribute voiced-frame statistics. BPDR mean original voiced-frame alteration is 0.000366263 cents for Vocadito and 0.000393429 cents for Jamendo. No voiced frame exceeds the existing five-cent descriptive edit threshold for any evaluated model. This is raw prediction analysis, not rendered-audio or musical-quality verification.

**334 regression tests pass**. Independent verification exactly reproduces **3,912 model prediction vectors after fresh CPU reloads** and **1,304 zero-baseline vectors**. It replays 1,216 paired metric rows and 352 original-control rows, checks dataset/group-aware means separately, and confirms identical A/B and C/D predictions for all 304 frame-only pairs. All 3,420 current Jamendo model vectors also match the historical selected-model GPU results within the pre-existing 1e-5 absolute/relative CPU-export tolerance; maximum difference is below 0.000000388 cents. All model states, normalizer values and 28 common-pipeline files remain unchanged.

The 19 Vocadito variants still represent only seven shift cases, six phrases and three singer/recording groups; Jamendo contributes 285 variants from 24 crops / 12 artist proxies. The earlier seventeen acoustic failures, four unattempted geometry keys and two unobservable merges remain excluded and visible. The 88 original controls are evaluated independently of paired audit masks, so refused sources are not silently dropped from preservation checks. This accepted subset and its uneven direction/corruption coverage do not establish full-corpus or population accuracy.

No model is selected anew or adopted. This experiment adds **zero optimizer updates, audio files or acoustic extractions**; no learned WAV is rendered. The original matched comparison remains 900 updates and previous diagnostics 800. Research-ready examples remain zero; musical review D/J4 is owner-deferred, Gate F delivered repair remains open, and Gate E remains open because useful preservation-safe learned correction is unproven. The current Vocadito paired-training pool contains only eighteen variants from one recording/singer. That is a verified coverage limitation, not a proven cause of the near-zero response. Before another fit, prepare final reference masks on the already frozen training partition under a separately declared cache-first protocol.

## Combined dataset-balanced metrics

| Condition | Plateau error | Original plateau alteration | Outside alteration | Boundary sensitivity | Plateau counter-detuning error | Response gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero_correction | 24.6032407407 | 0 | 0 | 0 | 24.6032407407 | 0 |
| augmentation_only | 24.6051182767 | 0.00623172728739 | 0.00682295116112 | 1.86259499401e-05 | 24.6034746447 | -1.27786816613e-05 |
| frame_only | 24.6053354994 | 0.0113917263473 | 0.0115796017598 | 0 | 24.6033477841 | -5.5969726331e-06 |
| bpdr | 24.6032586706 | 0.000371731058449 | 0.000405512062134 | 1.3891963488e-06 | 24.6032424019 | 1.22229579079e-07 |

All errors/alterations are cents; response gain is dimensionless. The existing whole-common-mask correction MAE is also retained in raw metrics, but is not a substitute for plateau error: zero correction has whole-common-mask MAE 1.378455769 cents while leaving 24.603240741-cent plateau error.

## Dataset-specific results

| Dataset | Condition | Plateau error | Response gain | Original all-voiced alteration | Voiced / retained originals |
| --- | --- | ---: | ---: | ---: | --- |
| vocadito | zero_correction | 30 | 0 | 0 | 13 / 14 |
| vocadito | augmentation_only | 30.0014091598 | 1.03785308359e-05 | 0.00632568801908 | 13 / 14 |
| vocadito | frame_only | 30.0057311058 | 4.65397564161e-06 | 0.0119317877882 | 13 / 14 |
| vocadito | bpdr | 30.000051313 | -2.06680752118e-07 | 0.000366263153006 | 13 / 14 |
| jamendo | zero_correction | 19.2064814815 | 0 | 0 | 74 / 74 |
| jamendo | augmentation_only | 19.2088273936 | -3.59358941584e-05 | 0.00720380323966 | 74 / 74 |
| jamendo | frame_only | 19.2049398931 | -1.58479209078e-05 | 0.0110163159632 | 74 / 74 |
| jamendo | bpdr | 19.2064660281 | 4.51139910275e-07 | 0.000393428971375 | 74 / 74 |

## Method and validation assessment: share with caveats

The declaration predates all current model forwards and locks the historical checkpoints, input catalog, original controls, normalization and metrics. Every input is full context with one unique feature array forwarded at a time; duplicate vectors are cached by input bytes. Paired targets, human pitches, original comparison waveforms and intervention amounts never enter prediction features. Inference uses frozen parameters, eval mode, inference mode and the recorded Windows CPU environment. No optimizer exists in the evaluation workflow.

Means use equal children at each dataset -> singer/artist proxy -> recording -> phrase -> shift -> corruption level. Per-dataset, per-group, per-direction and per-corruption summaries are saved, including the absent Vocadito +/-10 and +/-20 cases as absent coverage, not imputed errors. Human-reference and model-estimate roles remain distinct. Refused boundary/audit cases are not counted as model errors or silently restored to the catalog.

Plateau response gain compares shifted/original predictions on identical audited positions. Its ideal value is one; constant bias gives zero paired response even when one injected direction happens to have slightly lower absolute error. Whole-common-mask counter-detuning/error statistics are retained for continuity, alongside plateau-only metrics that expose the strong zero-anchor majority. No new acceptance threshold, significance test or checkpoint ranking is invented after outcomes. No useful repair benefit is established by tiny changes in these observed means.

Exact reload verification covers 5,216 saved prediction vectors (3,912 model, 1,304 zero) and all 1,216 paired / 352 original metric rows. An independent sorted hierarchical mean calculation checks all overall metrics. The historical Jamendo compatibility comparison checks all 3,420 vectors under the original CPU-export tolerance; it does not relax exact same-environment reload checks. One zero-voiced original stays explicitly unavailable for voiced-frame metrics. The eight new regression tests cover zero/ideal response, wrong-direction handling, constant-bias failure, hierarchy, missing datasets, frozen selection and frozen inference requirements.

Run: `runs/frozen_development_evaluation/61dbfe5ad4f4498ba406bedadc777b14`. Each condition retains `predictions.npz` and `metrics.json`; the immutable declaration, independent verification and panel receipt pin model/data provenance. Current inputs remain 936 training variants / 304 development variants / 88 original controls. CPU export metadata remains exploratory and production-ineligible; no component integration or macOS execution is claimed.

## Next — Gate E remains open

Gate E training-data prerequisite (Gate B/C reference preparation): separately declare cache-first preparation of all 30 frozen Vocadito training phrases. Authenticate existing accepted-audio and acoustic caches before any computation, reuse unchanged saved arrays, and compute only missing original PESTO/YIN measurements in the component CPU workflow after accepted-audio checks. Replay the existing human-reference guards, voicing and tracker-agreement rules; measure final reliable masks and the existing multi-interior support geometry. Retain every technical/support refusal; do not relax masks or change singer partitions. This addresses an unverified training-data prerequisite: current paired Vocadito training covers one recording/singer. It is not proof that data coverage caused the failed learning result or that more data will fix it. No shifted audio, new pairs, fitting, development/holdout processing or shared-pipeline execution in that preparation step. Keep the current 936-example training pool and all 304/88 development artifacts frozen until separately declared audits justify additions. Gate E open, D/J4 review deferred, F open.
