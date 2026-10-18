# Gate E exploratory comparison - 16 October 2026


**Completed engineering comparison; research validation remains open.** All 88 declared candidates (14 Vocadito, 74 Jamendo) are retained. Of 528 declared source/shift entries, 366 are not attempted because the unchanged source/support checks refuse them; 162 are measured, with **95 passes and 67 failures**. The 95 passing shifts yield **285 four-view development examples from 24 Jamendo crops / 12 artist proxies**. All 14 Vocadito phrases lack sufficient continuous reliable support; their refusals replay from saved masks. Human-reference paired development scores and a full two-dataset paired macro are **unavailable**.

**Useful repair is not demonstrated.** Selected BPDR development MAE is **1.449429 cents**, versus **1.449022** for zero correction. Its detuned-plateau error is **19.206466 cents**, essentially the **19.206481** zero-correction error. BPDR is the best of the three trained conditions on the declared criterion, while its selected response remains near zero. No model is adopted.

The three conditions each execute the locked 300 updates (900 in the completed comparison), with seed 339, common saved samples, training-only normalization, common initialization and full phrase context. Effective batches of eight use physical batches of **8** on **Intel(R) Arc(TM) Graphics**, chosen by a forward/backward memory/throughput probe with zero optimizer updates. An earlier partial CPU attempt was stopped for slow throughput and preserved separately at `runs/matched_comparison/70af6211ac2f46a099686a14867b0f36`; its results are excluded from this comparison. All three conditions restart together on the same verified existing GPU environment. No new installation or shared-pipeline execution occurs. Architecture, existing per-example losses, warmup, offsets, masks and audit thresholds remain fixed. Every checkpoint at 25-update intervals is retained with raw predictions and metrics; selection minimizes the declared hierarchical criterion over the **available Jamendo development data**. Missing Vocadito results remain unavailable. PP1 evaluation and final holdout remain reserved.

The eligible training pool remains **18 Vocadito variants from one recording / one singer ID** and **918 Jamendo variants from 70 crops / 23 artist proxies**. The previously declared equal-dataset, group/recording/phrase/shift/corruption weighting is unchanged. Each condition's saved 2,400 draws visit **535 distinct variants**, covering all 71 original training recordings/crops: 18 Vocadito variants and 517 Jamendo variants across all 23 artist proxies. The actual draw counts are 1,221 Vocadito and 1,179 Jamendo; equal probability does not require exactly equal realized counts. Repeated variants do not add independent singer coverage. Normalization uses the entire eligible training pool, with no development frames.

Read-only reference diagnosis replays all 14 Vocadito final masks exactly. Twelve phrases have enough contiguous human-F0 voiced coverage, and seven have enough joint single-note coverage before the note-interior guards. **Zero have enough after those guards**, before automatic tracker filtering. The existing renderer/audit geometry needs a 0.86-second reliable endpoint span. This is a reference-support feasibility finding; no mask, tolerance, split or rule is changed, and the earlier-stage masks are never substituted as approved examples.

**Post-hoc learning diagnostics, excluded from selection:** about **92.9%** of the anchor-loss target weight in the actual training draws is zero correction. Selected BPDR's average absolute C-minus-A / D-minus-B response on audited plateaus is only **0.000063 cents**. The direct input pitch-delta cue has **16.913 cents** error against the requested shifts and is numerically zero on **26.9%** of those plateau frames under the declared macro weighting. These observations identify target balance and fine-pitch sensitivity as candidates for controlled diagnosis; they do not prove a cause, and spectral features are also available. No training, threshold, selection or data is changed after seeing these diagnostics.

An additional preservation diagnostic uses the 88 technically processed original candidates: 14 Vocadito and 74 Jamendo. It measures predicted alteration over original voiced frames with no detuning-target mask. These original-input diagnostics do not supply the missing paired human-reference recovery result or establish that original singing is musically correct. Both the paired joint-voicing diagnostic and this unpaired path retain explicit frame counts.

**Selected updates:** augmentation-only 25, frame-only 50, BPDR 300. The selected BPDR checkpoint follows warmup; the single-seed predicted-reference scope still limits interpretation. The zero-correction baseline is retained so invariant outputs cannot be mistaken for successful repair. Frame-only predictions are exactly identical within A/B and C/D. Independent verification replays all 10,260 checkpoint metric rows, checks 36 checkpoints and their selection, verifies shared initialization/warmup, and reproduces selected prediction vectors within the declared float32 tolerances. Exact-equality counts are reported separately.

The plots show lower-is-better errors in cents, not an accuracy percentage. Dependent variants, artist proxies, one seed, model-estimate Jamendo references, synthetic renderer artifacts and missing human-reference paired coverage limit the conclusions. No learned correction is rendered or handed downstream. Audio widgets elsewhere illustrate sources and controlled interventions; learned outputs here are correction curves. Reviews and the common pipeline remain unchanged.

The three selected models also have inference-only exports that load through the existing Windows CPU loader, with unchanged tensors and predictions checked against the saved GPU example within the declared float32 tolerance. Optimizer/device state remains in the original training checkpoints. These exports retain exploratory/pending-review status; component adoption, delivered audio repair and macOS execution are pending. This does not implement the future portable extraction pipeline.

**Next - Gate E reference validation:** Gate E repair-benefit validation remains open: diagnose the observed near-zero response with a small controlled overfit/gradient test, audit zero-target weighting and input pitch sensitivity, and design a human-reference-feasible follow-up. Declare each new ablation before changing masks, intervention rules, loss weighting, features or model settings. Preserve the current comparison, source partitions and reserved PP1/final holdouts. Gate D/J4 musical review is deferred; Gate F delivered repair remains open. This single-seed synthetic comparison does not authorize model adoption or shared-pipeline changes.

## Recorded selected-checkpoint measurements

| Condition | Selected update | Jamendo development MAE (cents) | Detuned-plateau recovery error (cents) | Original audited alteration (cents) |
| --- | ---: | ---: | ---: | ---: |
| zero_correction | n/a | 1.449 | 19.206 | 0.000 |
| augmentation_only | 25 | 1.457 | 19.209 | 0.008 |
| frame_only | 50 | 1.459 | 19.205 | 0.010 |
| bpdr | 300 | 1.449 | 19.206 | 0.000 |

The criterion includes protected outside frames as declared; plateau recovery and preservation must be assessed alongside it. A low global MAE alone does not establish useful correction. Source and per-group metrics, raw vectors, all 12 candidates per condition and reload verification remain in the comparison run.

Development run: `runs/development_preparation/21867884731a4a88aea044d37a2b1f33`. Comparison run: `runs/matched_comparison/64dc9a78175c4113929e8a890894dcbd`. Presentation: `runs/gate_e_presentation/55fc3a263c7e449e909a4ff8b5c3f09f`.

Reference diagnosis: `runs/reference_feasibility_diagnosis/78a7a71730f24cd682f9695efc5fc5a6`. CPU inference exports: `runs/matched_model_exports/5607b773e7004bcfbd8dfd38daa2d123`.

Learning diagnosis: `runs/matched_learning_diagnosis/798f9039e82f4c0d92abe468e0f5102d`. Post-hoc diagnostics do not change the selected checkpoints.

## Next declared work, planned only

1. Run a small controlled overfit/gradient positive control before another large fit. Compare with the analytic zero-prediction loss; check that the model can learn nonzero injected compensation and that gradients reach the pitch-sensitive features.
2. Investigate anchor target balance and input pitch resolution in separate declared ablations. Preserve this baseline; use the same sources and fixed controls within each ablation. Any additional acoustic measurement belongs in a versioned BPDR adapter, with independent audit considerations, rather than an edit to the shared extractor.
3. Design a reference-feasible support construction on development data while retaining trusted annotation interiors and explicit audits. Candidate gap-aware supports require their own declared tests; removing guards to fill the dataset is not an approved change. Reserve PP1 evaluation and final holdout for the eventual frozen design.
4. Repeat a matched comparison only after feasibility and learning diagnostics justify the new declaration. Gate F delivered repair and future musical review remain separate requirements; no current checkpoint is approved for integration.

## Completed verification and panel evidence

All 261 component regression tests pass in the final source state. Eight matched-adapter checks also passed in the existing GPU environment. The three-example GPU forward/backward probes performed zero optimizer updates, preserved model parameters and informed only hardware batch feasibility. The completed GPU comparison retains the declared 300 updates per condition. The stopped partial CPU attempt, logs, code snapshot and any checkpoints remain separate evidence and are excluded from comparison selection.

Development verification authenticates 2,933 generated artifacts, retains all 528 declared source/shift keys, replays 162 measured audits/target masks and verifies 162 exact outside-support waveforms and 27 zero-shift source identities. It validates all 285 development pairs and confirms that both training-loader modes refuse them. Refusal support masks replay without threshold changes.

Comparison verification authenticates all 36 checkpoints, replays 10,260 raw prediction metric rows, checks the earliest-minimum development selection and verifies matched initialization and shared warmup under declared float32 tolerances. It reloads all 3,420 selected paired prediction vectors, with exact-equality counts recorded separately. The frame-only control is exactly invariant to the removed boundary feature. All three selected inference exports load through the existing Windows CPU path with unchanged tensors and the declared CPU/GPU example tolerance; macOS execution and the future portable pipeline remain unimplemented.

Both maintained notebooks freshly execute with zero errors, 33 plots and 44 audio widgets. Source, executed-copy, HTML and evidence-index hashes verify. Both new figures were visually inspected. Frozen inputs, partitions, reviews and all 28 shared-pipeline files remain unchanged. No learned correction is rendered or adopted. Research-ready examples remain zero; Gate E reference validation and Gate F delivered repair remain open, with Gate D/J4 review deferred.

Latest panel: `runs/notebooks/c1f81173332b4336a50fadf32bd2d70f/02_data_review_and_paired_experiments.html`. Final proof: `runs/gate_e_presentation/55fc3a263c7e449e909a4ff8b5c3f09f/final_verification.json`.
