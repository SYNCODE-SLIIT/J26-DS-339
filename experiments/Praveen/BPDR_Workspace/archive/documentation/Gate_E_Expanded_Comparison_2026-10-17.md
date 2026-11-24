# Gate E expanded-data comparison - 17 October 2026

**The locked Gate E expanded-data comparison is complete: 900 new optimizer updates, 300 in each condition.** The same saved 300-by-8 schedule, training-only candidate normalizer, initial common-shape state, and full-phrase context were used for augmentation-only (71 inputs), frame-only (70) and BPDR (71). Execution used the declared Intel Arc XPU, torch 2.14.1+xpu, float32, physical microbatch one and eight-step accumulation. The architecture, objective, 100-update supervised warm-up and update budget were unchanged. All 36 checkpoints, three initial states, 900 history rows and checkpoint assessments are retained.

**Training used 995 exploratory examples: 77 human-reference Vocadito variants and 918 Jamendo model-estimate variants.** Vocadito spans 18 phrases, 14 recordings and 11 annotated singers; Jamendo uses 23 artist proxy groups. Each condition made 2,400 replacement draws covering 601 distinct variants, rather than a full pass over every available example; dataset draws were {'vocadito': 1179, 'jamendo': 1221}. The eighteen legacy phrase variants remain quarantined only for this new experiment under the exact-feature rule. Historical active 936-example preparation, the 1,013-example proposal, original normalization, earlier model weights, rejected audits and reviews are preserved. The new sample membership, normalization and selection rule differ from the historical comparison; a difference cannot be attributed to dataset expansion alone.

**All checkpoint decisions use the frozen 304 development variants and all 88 original controls.** The zero-correction balanced plateau error is **24.603240741 cents**. Nineteen development variants use Vocadito human references (seven shifts, six phrases, three recordings/singers) and 285 use Jamendo model estimates (24 crops, 12 artist proxies). Candidate ranking uses plateau-only repair error and strict improvement over zero, subject to each dataset separately passing one-cent engineering bounds for original plateau, shifted outside and original all-voiced alteration. Whole-mask MAE is reported only. The one-cent guard is not a human-calibrated musical threshold. All 88 controls remain; vocadito_9_p01 has no voiced frames and its all-voiced statistic stays unavailable.

**Outcome:** BPDR's exploratory development selector chooses update 75; its balanced plateau improvement is 0.007849921 cents, with response gain 3.44202192e-06 (ideal 1). This numerical selection is not model adoption or musical validation. The selected checkpoint is within supervised warm-up, before BPDR response terms were enabled; it provides no evidence of benefit from those extra terms. The table below distinguishes the selected policy from each raw 300-update checkpoint. Every non-selected checkpoint remains available, including failed or unhelpful outcomes. Useful preservation-safe musical repair is not established; no model is adopted.

**Read-only training-target support check:** under the frozen hierarchical training mass, nonzero correction occupies 7.026% of four-view anchor slots and 14.052% of counter-detuning slots; the rest have zero targets. This quantifies the existing objective, without removing outside/original protection or proving why optimization behaves this way. A targeted objective study should balance repair support and preservation explicitly, rather than assuming more examples alone will establish useful correction. No objective change or further fit occurs here.

**Independent verification passed.** All 36 saved checkpoint assessments and 52,160 prediction vectors were checked; plateau, paired response and original-preservation metrics and hierarchical dataset means were reconstructed. Reloaded model predictions satisfy the previously declared numerical bounds; exact equality is recorded separately. Common initialization and the frame-only first-seventy-column projection are exact. The shared 100-update warm-up parameters are exactly equal. All 7,544 frozen inputs and 28 common-pipeline files are unchanged; 377 regression tests pass.

**Gate E remains open.** This experiment adds 900 updates to the earlier 900-update comparison; the 800 diagnostic updates are a separate historical category (2,600 recorded updates across these categories). There was no acoustic extraction, new data generation, shared-pipeline execution, learned audio rendering, PP1/final-holdout access or production integration in this step. Research-ready examples remain zero, D/J4 review remains owner-deferred and F remains open.

## Selected policy and final raw checkpoint

| Condition | Selected update | Selected plateau error (cents) | Improvement over zero (cents) | Selected response gain | Raw update-300 plateau error |
| --- | ---: | ---: | ---: | ---: | ---: |
| augmentation_only | 175 | 24.591043750 | 0.012196990 | 1.06419106e-06 | 24.606492938 |
| frame_only | 125 | 24.599611331 | 0.003629410 | 4.29476066e-07 | 24.602848822 |
| bpdr | 75 | 24.595390820 | 0.007849921 | 3.44202192e-06 | 24.600515919 |

## Selected policy by dataset

| Condition | Dataset | Plateau error (cents) | Paired response error (cents) | Paired response gain | Original plateau alteration | Shifted outside alteration | All-original voiced alteration |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| augmentation_only | vocadito | 29.963036815 | 29.999961376 | 1.28458547e-06 | 0.074092821 | 0.073805998 | 0.073863386 |
| augmentation_only | jamendo | 19.219050685 | 19.206462697 | 8.4379665e-07 | 0.073933801 | 0.073696511 | 0.073760158 |
| frame_only | vocadito | 29.989090284 | 29.999981244 | 6.26621798e-07 | 0.021816079 | 0.021353693 | 0.021416627 |
| frame_only | jamendo | 19.210132377 | 19.206470713 | 2.32330334e-07 | 0.021312769 | 0.021123206 | 0.021256014 |
| bpdr | vocadito | 29.976130671 | 29.999837769 | 5.40621369e-06 | 0.047949365 | 0.047402124 | 0.047648907 |
| bpdr | jamendo | 19.214650969 | 19.206448042 | 1.47783014e-06 | 0.047821066 | 0.047423513 | 0.047447200 |

## Direction breakdown of the selected policy

A common small bias can help one shift direction and hurt the opposite direction. Paired response gain measures the change between shifted and original predictions; gain 1 is ideal and 0 means no response. These development aggregates are not confidence intervals or independent musical validation.

| Condition | Dataset | Injected cents | Variants | Zero plateau error | Selected plateau error | Selected paired response gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| augmentation_only | jamendo | -30.0 | 54 | 30.000000000 | 30.074165462 | -1.7172696e-06 |
| augmentation_only | jamendo | -20.0 | 51 | 20.000000000 | 20.074023560 | -3.27028032e-06 |
| augmentation_only | jamendo | -10.0 | 45 | 10.000000000 | 10.073867687 | -6.3173856e-06 |
| augmentation_only | jamendo | 10.0 | 54 | 10.000000000 | 9.925925823 | 9.49142425e-06 |
| augmentation_only | jamendo | 20.0 | 45 | 20.000000000 | 19.925890636 | 8.27629694e-06 |
| augmentation_only | jamendo | 30.0 | 36 | 30.000000000 | 29.926056472 | 5.95893281e-06 |
| augmentation_only | vocadito | -30.0 | 5 | 30.000000000 | 30.074442705 | -4.61372542e-06 |
| augmentation_only | vocadito | 30.0 | 14 | 30.000000000 | 29.925810205 | 2.79527642e-06 |
| frame_only | jamendo | -30.0 | 54 | 30.000000000 | 30.021544139 | -1.90850494e-07 |
| frame_only | jamendo | -20.0 | 51 | 20.000000000 | 20.021391805 | 1.69797539e-07 |
| frame_only | jamendo | -10.0 | 45 | 10.000000000 | 10.021183586 | -5.69750485e-07 |
| frame_only | jamendo | 10.0 | 54 | 10.000000000 | 9.978692383 | 1.38372158e-07 |
| frame_only | jamendo | 20.0 | 45 | 20.000000000 | 19.978615443 | 3.86487411e-06 |
| frame_only | jamendo | 30.0 | 36 | 30.000000000 | 29.978715658 | 2.68011236e-06 |
| frame_only | vocadito | -30.0 | 5 | 30.000000000 | 30.022056580 | -3.98335351e-06 |
| frame_only | vocadito | 30.0 | 14 | 30.000000000 | 29.978094101 | 1.2616834e-06 |
| bpdr | jamendo | -30.0 | 54 | 30.000000000 | 30.048389199 | -5.79905909e-06 |
| bpdr | jamendo | -20.0 | 51 | 20.000000000 | 20.048017328 | -1.09963129e-05 |
| bpdr | jamendo | -10.0 | 45 | 10.000000000 | 10.047794748 | -2.0818476e-05 |
| bpdr | jamendo | 10.0 | 54 | 10.000000000 | 9.951874108 | 2.74111105e-05 |
| bpdr | jamendo | 20.0 | 45 | 20.000000000 | 19.951751296 | 2.21112536e-05 |
| bpdr | jamendo | 30.0 | 36 | 30.000000000 | 29.952178351 | 1.61950622e-05 |
| bpdr | vocadito | -30.0 | 5 | 30.000000000 | 30.048757792 | -1.16849726e-05 |
| bpdr | vocadito | 30.0 | 14 | 30.000000000 | 29.951684448 | 1.0255029e-05 |

## Next - Gate E

Gate E: prepare one targeted plateau-balanced learning-objective comparison, using the saved response and dataset/direction/group evidence. Study separate reductions for injected correction support, original anchors and shifted outside preservation, retaining all protection targets and the unchanged evaluation guards. Use the existing frozen train/development artifacts; do not add more crops or repeat acoustic extraction. Any proposed objective/sampling change needs its own explicit comparison protocol before fitting. Keep zero correction as the runtime policy; no learned WAV rendering, model adoption or PP1/final-holdout assessment. D/J4 musical review remains deferred and F delivered-repair validation remains open.

## Evidence

- [declaration.json](../runs/expanded_comparison/1ab41042179b41959e52b7c9320f44d3/declaration.json)
- [evidence.json](../runs/expanded_comparison/1ab41042179b41959e52b7c9320f44d3/evidence.json)
- [independent_verification.json](../runs/expanded_comparison/1ab41042179b41959e52b7c9320f44d3/independent_verification.json)
- [panel_evidence.json](../runs/expanded_comparison/1ab41042179b41959e52b7c9320f44d3/panel_evidence.json)
- [locked_protocol.json](../runs/expanded_comparison/1ab41042179b41959e52b7c9320f44d3/locked_protocol.json)
- [expanded_comparison.png](../runs/expanded_comparison/1ab41042179b41959e52b7c9320f44d3/expanded_comparison.png)
- [augmentation_only checkpoint decisions](../runs/expanded_comparison/1ab41042179b41959e52b7c9320f44d3/augmentation_only/selection.json)
- [frame_only checkpoint decisions](../runs/expanded_comparison/1ab41042179b41959e52b7c9320f44d3/frame_only/selection.json)
- [bpdr checkpoint decisions](../runs/expanded_comparison/1ab41042179b41959e52b7c9320f44d3/bpdr/selection.json)
