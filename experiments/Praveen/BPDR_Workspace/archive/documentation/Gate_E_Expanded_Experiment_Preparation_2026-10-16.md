# Gate E compatibility decision and expanded experiment preparation - 16 October 2026

**The legacy cache issue is resolved for the next experiment by a versioned quarantine decision.** The exact-feature identity rule remains unchanged. The eighteen saved vocadito_37_p00 variants are excluded from this new experiment only; all their files and historical results remain intact. Its two blocked new shift keys remain blocked. The passed vocadito_37_p01 cases from the same recording/singer are retained because the decision is phrase-specific. This decision uses the recorded identity failure before any new fitted-model outcomes. It is not a musical rejection or a claim that the retained audio, pitch or earlier targets are invalid.

**Saved-array analysis quantifies the issue without re-extraction.** The largest raw seventy-feature difference is **7.62939453125e-6**; the largest actual float32 difference after the existing normalization is **9.5367431640625e-7**. The float64 raw-difference/existing-scale calculation is approximately 6.362701841803349e-7, separately recorded from the rounded float32 transform. These are feature units and dimensionless normalized units, not cents of pitch error. Pitch, confidence, voicing, pitch-change values, YIN and waveform samples remain exact. All thirty-six A/B views in the eighteen legacy records exactly match their recorded original baseline. ULP distances are retained as descriptive numerical evidence across different dtypes; no ULP or absolute-error tolerance is adopted. The numerical cause is not established.

**The next experiment is prepared on 995 training examples: 77 human-reference Vocadito variants plus 918 existing model-estimate Jamendo variants.** The new Vocadito data span eighteen phrases, fourteen recordings and eleven annotated singers; Jamendo retains twenty-three artist proxy groups. The earlier 1,013-record candidate proposal remains saved; removing the eighteen quarantined variants gives the separate 995-record experimental list. The active historical preparation stays at 936 examples and its current pointer is unchanged. All examples remain exploratory, pending review and training_eligible=False.

**A new candidate-only normalizer and sampling schedule are prepared.** Weighted moments use exactly the selected training records, four real views per record, and **3,772,824 view-frame positions**. Padding is excluded and natural unvoiced context is retained. Hierarchical weights allocate equal mass to each dataset, then groups, recordings, phrases, shifts and corruptions. This is a declared engineering mixture, not a population-frequency estimate. The existing normalizer is preserved; only the new experiment uses the new 71-column state, with its first seventy columns used by frame-only. A seed-339, 300-by-8 schedule samples with replacement and is shared across all conditions. It does not imply that every candidate is drawn. No development or holdout frames enter these statistics.

**The locked fit budget is 300 updates per condition, 900 total, with the existing architecture and loss unchanged.** The three conditions are augmentation-only (71 inputs), frame-only (70) and BPDR (71): hidden 128, four layers, four heads, feed-forward 512, dropout 0.1, AdamW learning rate 0.001, weight decay 0.01, gradient clipping 1.0 and one hundred shared supervised warm-up updates. The existing source defines boundary/detuning/smoothness weights 1/1/0.1 and per-example loss reduction. The declared device is the verified Intel Arc XPU environment with torch 2.14.1+xpu, float32, two CPU threads, physical microbatch one and eight-step accumulation for effective batch eight. An unavailable or changed declared runtime stops the fit; no automatic CPU fallback or budget expansion is allowed. Availability verification is not model-fit execution.

**Checkpoint selection is explicitly changed to match the repair question.** Every twenty-five updates, use the frozen 304 development variants (19 human-reference Vocadito / 285 model-estimate Jamendo) and all 88 original controls. Rank preservation-qualified checkpoints by equal-dataset/group/recording/phrase/shift/corruption **plateau-only repair error**, with earliest-update tie breaking and strict improvement over zero correction. Qualification requires each dataset separately to have original audited-plateau alteration, shifted outside alteration and original all-voiced alteration each at most one cent. The one-cent bound comes from existing engineering preservation diagnostics; it is not musically calibrated acceptance. If no checkpoint qualifies and improves on zero, select zero correction and retain all model checkpoints as failed/unproven learning results. Whole-common-mask MAE remains a reported metric but no longer substitutes for plateau repair. A numerical improvement alone is not proof of useful musical correction. All 88 controls remain recorded: voiced-frame statistics are available for 13/14 Vocadito and 74/74 Jamendo; vocadito_9_p01 remains unavailable rather than zero.

**373 regression tests pass.** Independent verification exactly reconstructs all 995 selected and eighteen quarantined records, all training-only moments, hierarchical probabilities and the saved sampling schedule. All 7,431 frozen inputs and twenty-eight common-pipeline files remain unchanged. The 304 development variants and 88 original controls are reused as cached artifacts; no new acoustic/reference processing occurs. This step performs **zero audio writes, acoustic extractions, model forwards or optimizer updates**, and creates no model checkpoint. The only new normalization state is candidate-only.

**Gate E preparation is complete; model fitting has not started.** Historical comparison/diagnostic counts remain 900/800, separately from the next planned 900 updates. Data membership, normalization and checkpoint selection differ from the historical run, so a future difference cannot be attributed to source diversity alone. The earlier frozen-model negative result remains valid. This is a single exploratory development experiment, with no PP1-evaluation/final-holdout access, musical review, learned WAV rendering or model adoption. Research-ready examples remain zero; D/J4 review stays owner-deferred, F remains open and useful preservation-safe learned correction remains unproven.

## Saved-array differences

| Array | Dtype | Different elements | Maximum absolute difference |
| --- | --- | ---: | ---: |
| times_seconds | float64 | 0 | 0 |
| f0_hz | float32 | 0 | 0 |
| confidence | float32 | 0 | 0 |
| voiced_mask | bool | 0 | 0 |
| pitch_change_cents | float64 | 0 | 0 |
| pitch_change_valid | bool | 0 | 0 |
| log_energy_db | float32 | 0 | 0 |
| onset_score | float32 | 457 | 5.96046447754e-08 |
| provisional_boundary_score | float64 | 360 | 3.72529029846e-08 |
| log_mel_db | float32 | 3167 | 7.62939453125e-06 |
| frame_features_70 | float32 | 3624 | 7.62939453125e-06 |
| boundary_features_71 | float32 | 3984 | 7.62939453125e-06 |
| native_pitch_times_seconds | float64 | 0 | 0 |
| native_pitch_hz | float32 | 0 | 0 |
| native_pitch_confidence | float32 | 0 | 0 |

## Active next step

Gate E execution: run the single locked expanded-data exploratory comparison from this prepared 995-record catalog, its candidate-only training normalizer and exact saved 300-by-8 sampling schedule. Use the verified Intel Arc XPU interpreter, float32, effective batch eight with one-phrase microbatches and eight-step accumulation. Fit augmentation-only (71 inputs), frame-only (70) and BPDR (71), each for 300 updates with 100 shared warm-up updates; planned total 900. Use the frozen 304 development variants and all 88 original controls only for declared checkpoint assessment, with plateau error and per-dataset one-cent diagnostic preservation guards; require improvement over zero correction or select zero. Retain every checkpoint/outcome and dataset-specific metric, including missing voiced statistics. Stop on a changed input/runtime or unavailable XPU, without silent CPU fallback, extra updates, data regeneration or policy search. Keep historical 936/1013 artifacts, the quarantined eighteen records and prior model checkpoints unchanged. No PP1/final-holdout access, shared pipeline execution, learned WAV rendering or production/model adoption. This preparation executed no model updates; the next fit remains exploratory, with D/J4 review deferred, F open and research-ready examples zero.

## Evidence

- [declaration.json](../runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616/declaration.json)
- [evidence.json](../runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616/evidence.json)
- [compatibility.json](../runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616/compatibility.json)
- [quarantined_legacy_records.json](../runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616/quarantined_legacy_records.json)
- [training_catalog.json](../runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616/training_catalog.json)
- [candidate_normalizer.npz](../runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616/candidate_normalizer.npz)
- [sampling_schedule.npz](../runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616/sampling_schedule.npz)
- [locked_protocol.json](../runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616/locked_protocol.json)
- [independent_verification.json](../runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616/independent_verification.json)
- [panel_evidence.json](../runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616/panel_evidence.json)
