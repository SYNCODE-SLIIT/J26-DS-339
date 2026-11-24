# Gate E revised-objective comparison - 17 October 2026

**The sealed Gate E objective comparison is complete: 900 new optimizer updates, 300 each for augmentation-only, frame-only and BPDR.** The revised anchor uses 0.50 original, 0.25 correction-support and 0.25 shifted-outside preservation; the paired counter-detuning term weights correction and outside regions 0.50/0.50. Boundary invariance, adjacent-voiced smoothness and their 1/1/0.1 coefficients are unchanged. No common-pipeline code was changed or executed.

**The contrast reuses the saved 900-update expanded-data control arm.** Both arms use the same 995 training records (77 human-reference Vocadito / 918 Jamendo model-estimate variants), the same training-only normalizer, 300-by-8 replacement schedule, singer/artist partitions, architecture, seed, initialization, optimizer and update budget. The schedule makes 2,400 draws per condition and covers 601 distinct variants; it is not a full pass through every available record. Execution uses the declared Intel Arc XPU, torch 2.14.1+xpu, float32, physical microbatch one and effective batch eight. All new conditions share the revised anchor during 100 warm-up updates, after which BPDR response terms activate. The new warm-up is not expected to match the old-objective arm.

**Evaluation and selection remain unchanged.** Every 25 updates, assess all frozen 304 development variants and all 88 original controls. Choose the lowest dataset/group/source/phrase/shift/corruption balanced direct plateau error among checkpoints passing each dataset’s one-cent original-plateau, shifted-outside and all-original voiced preservation guards; require strict improvement over zero or select zero correction. The one-cent bound is engineering-only, not musical calibration. All 88 originals are retained; vocadito_9_p01 remains unavailable for voiced preservation statistics. Paired response error/gain and signed directions distinguish actual shifted-minus-original response from shared output bias.

**Outcome:** BPDR selects update 200. Its selected direct plateau error is 24.596802781 cents versus 24.595390820 for the prior objective, a revised-minus-reference change of +0.001411960 cents (negative is better). Its paired response gain is 3.13914145e-06, compared with 3.44202192e-06 before; ideal gain is 1 and zero response is 0. Paired response error is 24.603171025 cents. Numerical selection is not model adoption or musical validation. The tables retain the selected-policy results and the raw final-checkpoint results, including unselected or failed outcomes. Useful musical repair is not established.

**Independent verification passed.** All 36 new checkpoint assessments and 52,160 prediction vectors were replayed, including plateau, whole-mask, paired response and original-preservation metrics and hierarchical means. All 900 recorded loss rows match the sealed regional objective, including warm-up activation. The 36 reference checkpoints are compared through their pinned saved assessments; no reference fitting or new reference forwards occur. Initial states are exactly equal across objective arms, and common new initialization/projection checks pass. The revised augmentation/BPDR 100-update warm-up is exactly equal. All three checkpoint reload checks pass their predeclared numerical bounds, with exact equality recorded separately. All 7,777 frozen inputs and 28 shared-pipeline files remain unchanged. The unchanged core implementation retains its 388-test regression pass; this step adds artifact-level execution verification.

**Gate E remains open.** This is one prespecified objective-family contrast using a development set already consulted during design and one seed; it is not independent held-out accuracy, causal proof about the earlier failure, or production readiness. Regional balancing changes gradient magnitudes and tradeoffs as well as correction-target mass. The original 900-update comparison, expanded 900-update arm and 800 named diagnostic updates remain preserved; this step adds a separate 900 updates (2,700 main comparison updates and 3,500 main-plus-named-diagnostic updates; other historical engineering/test smoke runs are excluded). No new audio, acoustic extraction, learned WAV rendering, reviews, PP1/final-holdout assessment or model adoption occurs. D/J4 review stays owner-deferred, F stays open and research-ready examples remain zero.

## Selected policy versus saved reference

| Condition | Old update | Revised update | Old direct error | Revised direct error | Revised minus old error | Old response gain | Revised response gain | Raw final error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| augmentation_only | 175 | 275 | 24.591043750 | 24.594014523 | +0.002970773 | 1.06419106e-06 | 2.00549534e-06 | 24.596028547 |
| frame_only | 125 | 200 | 24.599611331 | 24.587755224 | -0.011856107 | 4.29476066e-07 | 6.64285343e-07 | 24.604851252 |
| bpdr | 75 | 200 | 24.595390820 | 24.596802781 | +0.001411960 | 3.44202192e-06 | 3.13914145e-06 | 24.602539371 |

## Raw update-300 checkpoints, independent of selection

The three alteration columns show the larger dataset mean. Qualification still checks each dataset separately against the unchanged one-cent engineering bound.

| Condition | Direct error | Response gain | Preservation qualifies | Original plateau alteration | Shifted outside alteration | All-original voiced alteration |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| augmentation_only | 24.596028547 | 3.19847755e-06 | True | 0.042814901 | 0.042131873 | 0.042859132 |
| frame_only | 24.604851252 | 4.37365943e-07 | True | 0.011707933 | 0.011768121 | 0.011400906 |
| bpdr | 24.602539371 | 2.55625653e-06 | True | 0.003942134 | 0.003718801 | 0.004147857 |

## Revised selected policy by dataset

| Condition | Dataset | Direct plateau error | Paired response error | Response gain | Original plateau alteration | Shifted outside alteration | All-original voiced alteration |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| augmentation_only | vocadito | 29.972310848 | 30.000107580 | -3.58040444e-06 | 0.054942483 | 0.054280820 | 0.054825954 |
| augmentation_only | jamendo | 19.215718199 | 19.206310108 | 7.59139513e-06 | 0.053347756 | 0.053167080 | 0.053391084 |
| frame_only | vocadito | 29.953205903 | 30.000154018 | -5.08217321e-06 | 0.093481630 | 0.093289071 | 0.093263452 |
| frame_only | jamendo | 19.222304544 | 19.206345046 | 6.4107439e-06 | 0.092087709 | 0.091961173 | 0.092388282 |
| bpdr | vocadito | 29.980523931 | 29.999973536 | 8.93158967e-07 | 0.038673815 | 0.038359962 | 0.038726728 |
| bpdr | jamendo | 19.213081631 | 19.206368514 | 5.38512394e-06 | 0.037771063 | 0.037360640 | 0.037553912 |

## Revised selected policy by signed shift

| Condition | Dataset | Injected cents | Pairs | Direct error | Response error | Response gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| augmentation_only | jamendo | -30.0 | 54 | 30.053116769 | 29.999539081 | 1.53579947e-05 |
| augmentation_only | jamendo | -20.0 | 51 | 20.052963482 | 19.999550770 | 2.24491214e-05 |
| augmentation_only | jamendo | -10.0 | 45 | 10.052329683 | 9.999580288 | 4.1980432e-05 |
| augmentation_only | jamendo | 10.0 | 54 | 9.947201875 | 10.000292220 | -2.92293323e-05 |
| augmentation_only | jamendo | 20.0 | 45 | 19.947398804 | 20.000163177 | -8.14720936e-06 |
| augmentation_only | jamendo | 30.0 | 36 | 29.947133040 | 29.999991147 | 3.08550087e-07 |
| augmentation_only | vocadito | -30.0 | 5 | 30.054394801 | 29.999700228 | 9.99056404e-06 |
| augmentation_only | vocadito | 30.0 | 14 | 29.945358011 | 30.000221835 | -7.39394879e-06 |
| frame_only | jamendo | -30.0 | 54 | 30.091745412 | 29.999641454 | 1.19359657e-05 |
| frame_only | jamendo | -20.0 | 51 | 20.091793060 | 19.999650772 | 1.74509577e-05 |
| frame_only | jamendo | -10.0 | 45 | 10.091495397 | 9.999704340 | 2.95529767e-05 |
| frame_only | jamendo | 10.0 | 54 | 9.908479496 | 10.000226653 | -2.26928836e-05 |
| frame_only | jamendo | 20.0 | 45 | 19.908557849 | 20.000135867 | -6.78907527e-06 |
| frame_only | jamendo | 30.0 | 36 | 29.908001995 | 30.000047207 | -1.58138418e-06 |
| frame_only | vocadito | -30.0 | 5 | 30.093062401 | 29.999732971 | 8.90477259e-06 |
| frame_only | vocadito | 30.0 | 14 | 29.906879425 | 30.000289281 | -9.59595438e-06 |
| bpdr | jamendo | -30.0 | 54 | 30.037630140 | 29.999779183 | 7.34084978e-06 |
| bpdr | jamendo | -20.0 | 51 | 20.037583603 | 19.999811578 | 9.39842168e-06 |
| bpdr | jamendo | -10.0 | 45 | 10.037141182 | 9.999833149 | 1.66507588e-05 |
| bpdr | jamendo | 10.0 | 54 | 9.962633476 | 10.000068633 | -6.89059029e-06 |
| bpdr | jamendo | 20.0 | 45 | 19.962642670 | 19.999968698 | 1.58257432e-06 |
| bpdr | jamendo | 30.0 | 36 | 29.962542860 | 29.999863863 | 4.51717937e-06 |
| bpdr | vocadito | -30.0 | 5 | 30.038757881 | 29.999949455 | 1.69233205e-06 |
| bpdr | vocadito | 30.0 | 14 | 29.961326573 | 29.999989033 | 3.72175439e-07 |

## Next - Gate E

Gate E diagnosis: use the saved revised-objective checkpoints and frozen training cases for a bounded input-response and gradient-flow study. Check whether the injected pitch signal reaches the model, inspect nearby or identical inputs for conflicting correction targets, and measure whether original, correction, outside, boundary and counter-detuning gradients conflict or collapse to common bias. Use training data for that diagnosis, retain the current development evidence, and declare any later model/objective experiment before fitting. Do not repeat a broad fit, increase the update budget or acquire more crops without a diagnosis-supported change. No extraction, shared pipeline execution, PP1/final-holdout assessment, learned audio delivery or model adoption; D/J4 review deferred, E/F open, research-ready zero.

## Evidence

- [declaration.json](../runs/plateau_comparison/0483fe3d53a54397be37c254b8344324/declaration.json)
- [locked_protocol.json](../runs/plateau_comparison/0483fe3d53a54397be37c254b8344324/locked_protocol.json)
- [evidence.json](../runs/plateau_comparison/0483fe3d53a54397be37c254b8344324/evidence.json)
- [independent_verification.json](../runs/plateau_comparison/0483fe3d53a54397be37c254b8344324/independent_verification.json)
- [reference_comparison.json](../runs/plateau_comparison/0483fe3d53a54397be37c254b8344324/reference_comparison.json)
- [panel_evidence.json](../runs/plateau_comparison/0483fe3d53a54397be37c254b8344324/panel_evidence.json)
- [plateau_comparison.png](../runs/plateau_comparison/0483fe3d53a54397be37c254b8344324/plateau_comparison.png)
