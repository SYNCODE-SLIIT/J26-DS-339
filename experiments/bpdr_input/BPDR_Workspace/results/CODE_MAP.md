# Implementation map

The source files are physically grouped by responsibility. Established `bpdr_input.model`, `bpdr_input.pipeline` and similar imports remain supported. Historical `presentation_*` imports resolve to renamed modules; there are no duplicate implementation copies.

| Start here | Purpose |
| --- | --- |
| [app.py](<../source/bpdr_input/application/app.py>) | User input and side-by-side outputs |
| [model_runtime.py](<../source/bpdr_input/repair/model_runtime.py>) | Audio-only inference, bounded candidate and rollback |
| [model.py](<../source/bpdr_input/modeling/model.py>) | Transformer definition |
| [training.py](<../source/bpdr_input/modeling/training.py>) | Original training command |

## Names clarified

| Previous name | Current file | Actual purpose |
| --- | --- | --- |
| `presentation_app.py` | [legacy_interface.py](<../source/bpdr_input/application/legacy_interface.py>) | Compatibility interface entry; active UI is app.py |
| `presentation_study.py` | [transfer_study.py](<../source/bpdr_input/experiments/evaluation/transfer_study.py>) | Real-data transfer/adaptation and model scoring |
| `presentation_baselines.py` | [baseline_evaluation.py](<../source/bpdr_input/experiments/evaluation/baseline_evaluation.py>) | Original/rule correction baseline evaluation |
| `presentation_statistics.py` | [paired_statistics.py](<../source/bpdr_input/reporting/paired_statistics.py>) | Group-paired uncertainty/statistical comparisons |
| `presentation_release.py` | [release_builder.py](<../source/bpdr_input/packaging/release_builder.py>) | Build a review snapshot from saved experiments |
| `completion_release.py` | [review_package.py](<../source/bpdr_input/packaging/review_package.py>) | Package a completed evidence/code/model snapshot |

## .

- [__init__.py](<../source/bpdr_input/__init__.py>)

## application

- [app.py](<../source/bpdr_input/application/app.py>)
- [legacy_interface.py](<../source/bpdr_input/application/legacy_interface.py>)
- [saved_results.py](<../source/bpdr_input/application/saved_results.py>)

## audio

- [analysis.py](<../source/bpdr_input/audio/analysis.py>)
- [contracts.py](<../source/bpdr_input/audio/contracts.py>)
- [model_pins.py](<../source/bpdr_input/audio/model_pins.py>)
- [pipeline.py](<../source/bpdr_input/audio/pipeline.py>)

## experiments/audio

- [anchor_feasibility.py](<../source/bpdr_input/experiments/audio/anchor_feasibility.py>)
- [audit_diagnostics.py](<../source/bpdr_input/experiments/audio/audit_diagnostics.py>)
- [boundary_comparison.py](<../source/bpdr_input/experiments/audio/boundary_comparison.py>)
- [boundary_renderer.py](<../source/bpdr_input/experiments/audio/boundary_renderer.py>)
- [context_ablation.py](<../source/bpdr_input/experiments/audio/context_ablation.py>)
- [context_ablation_experiment.py](<../source/bpdr_input/experiments/audio/context_ablation_experiment.py>)
- [context_ablation_scoring.py](<../source/bpdr_input/experiments/audio/context_ablation_scoring.py>)
- [controller_fixtures.py](<../source/bpdr_input/experiments/audio/controller_fixtures.py>)
- [demo.py](<../source/bpdr_input/experiments/audio/demo.py>)
- [estimated_harmonic_experiment.py](<../source/bpdr_input/experiments/audio/estimated_harmonic_experiment.py>)
- [estimated_harmonic_renderer.py](<../source/bpdr_input/experiments/audio/estimated_harmonic_renderer.py>)
- [framewise_diagnostics.py](<../source/bpdr_input/experiments/audio/framewise_diagnostics.py>)
- [framewise_fixtures.py](<../source/bpdr_input/experiments/audio/framewise_fixtures.py>)
- [harmonic_separation.py](<../source/bpdr_input/experiments/audio/harmonic_separation.py>)
- [harmonic_separation_experiment.py](<../source/bpdr_input/experiments/audio/harmonic_separation_experiment.py>)
- [known_control_diagnostics.py](<../source/bpdr_input/experiments/audio/known_control_diagnostics.py>)
- [known_source_control.py](<../source/bpdr_input/experiments/audio/known_source_control.py>)
- [phase_comparison.py](<../source/bpdr_input/experiments/audio/phase_comparison.py>)
- [phase_feasibility.py](<../source/bpdr_input/experiments/audio/phase_feasibility.py>)
- [phase_regression_diagnostics.py](<../source/bpdr_input/experiments/audio/phase_regression_diagnostics.py>)
- [phase_renderer.py](<../source/bpdr_input/experiments/audio/phase_renderer.py>)
- [prediction_preview.py](<../source/bpdr_input/experiments/audio/prediction_preview.py>)
- [renderer_calibration.py](<../source/bpdr_input/experiments/audio/renderer_calibration.py>)
- [renderer_candidates.py](<../source/bpdr_input/experiments/audio/renderer_candidates.py>)
- [segment_phase.py](<../source/bpdr_input/experiments/audio/segment_phase.py>)
- [segment_phase_experiment.py](<../source/bpdr_input/experiments/audio/segment_phase_experiment.py>)
- [segment_phase_scoring.py](<../source/bpdr_input/experiments/audio/segment_phase_scoring.py>)
- [source_estimation_experiment.py](<../source/bpdr_input/experiments/audio/source_estimation_experiment.py>)
- [source_estimator.py](<../source/bpdr_input/experiments/audio/source_estimator.py>)
- [source_transfer.py](<../source/bpdr_input/experiments/audio/source_transfer.py>)

## experiments/evaluation

- [baseline_evaluation.py](<../source/bpdr_input/experiments/evaluation/baseline_evaluation.py>)
- [cue_features.py](<../source/bpdr_input/experiments/evaluation/cue_features.py>)
- [environmental_noise_study.py](<../source/bpdr_input/experiments/evaluation/environmental_noise_study.py>)
- [generalization_probe.py](<../source/bpdr_input/experiments/evaluation/generalization_probe.py>)
- [humming_experiment.py](<../source/bpdr_input/experiments/evaluation/humming_experiment.py>)
- [humming_intake.py](<../source/bpdr_input/experiments/evaluation/humming_intake.py>)
- [humming_intake_extension.py](<../source/bpdr_input/experiments/evaluation/humming_intake_extension.py>)
- [noise_study.py](<../source/bpdr_input/experiments/evaluation/noise_study.py>)
- [reference_cue_ceiling.py](<../source/bpdr_input/experiments/evaluation/reference_cue_ceiling.py>)
- [rule_evaluation.py](<../source/bpdr_input/experiments/evaluation/rule_evaluation.py>)
- [runtime_cue_diagnosis.py](<../source/bpdr_input/experiments/evaluation/runtime_cue_diagnosis.py>)
- [transfer_study.py](<../source/bpdr_input/experiments/evaluation/transfer_study.py>)

## experiments/learning

- [comparison_preparation.py](<../source/bpdr_input/experiments/learning/comparison_preparation.py>)
- [controlled_study.py](<../source/bpdr_input/experiments/learning/controlled_study.py>)
- [expanded_comparison.py](<../source/bpdr_input/experiments/learning/expanded_comparison.py>)
- [expanded_experiment_preparation.py](<../source/bpdr_input/experiments/learning/expanded_experiment_preparation.py>)
- [feature_weighting_control.py](<../source/bpdr_input/experiments/learning/feature_weighting_control.py>)
- [fine_pitch.py](<../source/bpdr_input/experiments/learning/fine_pitch.py>)
- [fine_pitch_bpdr_check.py](<../source/bpdr_input/experiments/learning/fine_pitch_bpdr_check.py>)
- [fine_pitch_learning_curve.py](<../source/bpdr_input/experiments/learning/fine_pitch_learning_curve.py>)
- [fine_pitch_mse_check.py](<../source/bpdr_input/experiments/learning/fine_pitch_mse_check.py>)
- [fine_pitch_overfit.py](<../source/bpdr_input/experiments/learning/fine_pitch_overfit.py>)
- [fine_pitch_overfit_v2.py](<../source/bpdr_input/experiments/learning/fine_pitch_overfit_v2.py>)
- [fine_pitch_relative_check.py](<../source/bpdr_input/experiments/learning/fine_pitch_relative_check.py>)
- [frozen_development_evaluation.py](<../source/bpdr_input/experiments/learning/frozen_development_evaluation.py>)
- [humming_learning.py](<../source/bpdr_input/experiments/learning/humming_learning.py>)
- [input_response_diagnosis.py](<../source/bpdr_input/experiments/learning/input_response_diagnosis.py>)
- [learning_control.py](<../source/bpdr_input/experiments/learning/learning_control.py>)
- [matched_comparison.py](<../source/bpdr_input/experiments/learning/matched_comparison.py>)
- [multisource_control.py](<../source/bpdr_input/experiments/learning/multisource_control.py>)
- [plateau_comparison.py](<../source/bpdr_input/experiments/learning/plateau_comparison.py>)
- [plateau_objective.py](<../source/bpdr_input/experiments/learning/plateau_objective.py>)
- [real_input_diagnosis.py](<../source/bpdr_input/experiments/learning/real_input_diagnosis.py>)
- [real_learning_v2.py](<../source/bpdr_input/experiments/learning/real_learning_v2.py>)

## modeling

- [examples.py](<../source/bpdr_input/modeling/examples.py>)
- [losses.py](<../source/bpdr_input/modeling/losses.py>)
- [metrics.py](<../source/bpdr_input/modeling/metrics.py>)
- [model.py](<../source/bpdr_input/modeling/model.py>)
- [training.py](<../source/bpdr_input/modeling/training.py>)

## packaging

- [release_builder.py](<../source/bpdr_input/packaging/release_builder.py>)
- [review_package.py](<../source/bpdr_input/packaging/review_package.py>)

## preparation/diagnostics

- [coverage.py](<../source/bpdr_input/preparation/diagnostics/coverage.py>)
- [development_boundary_observability.py](<../source/bpdr_input/preparation/diagnostics/development_boundary_observability.py>)
- [review_bundle.py](<../source/bpdr_input/preparation/diagnostics/review_bundle.py>)
- [shared_screen_diagnostics.py](<../source/bpdr_input/preparation/diagnostics/shared_screen_diagnostics.py>)
- [support_acoustic_pilot.py](<../source/bpdr_input/preparation/diagnostics/support_acoustic_pilot.py>)
- [support_development_audit.py](<../source/bpdr_input/preparation/diagnostics/support_development_audit.py>)
- [support_fragmentation.py](<../source/bpdr_input/preparation/diagnostics/support_fragmentation.py>)
- [support_geometry.py](<../source/bpdr_input/preparation/diagnostics/support_geometry.py>)
- [training_support_audit.py](<../source/bpdr_input/preparation/diagnostics/training_support_audit.py>)

## preparation/pairs

- [development_preparation.py](<../source/bpdr_input/preparation/pairs/development_preparation.py>)
- [laboratory.py](<../source/bpdr_input/preparation/pairs/laboratory.py>)
- [shared_pair_audit.py](<../source/bpdr_input/preparation/pairs/shared_pair_audit.py>)
- [support_development_packaging.py](<../source/bpdr_input/preparation/pairs/support_development_packaging.py>)
- [training_reference_completion.py](<../source/bpdr_input/preparation/pairs/training_reference_completion.py>)
- [training_reference_completion_v2.py](<../source/bpdr_input/preparation/pairs/training_reference_completion_v2.py>)
- [training_reference_preparation.py](<../source/bpdr_input/preparation/pairs/training_reference_preparation.py>)
- [training_support_packaging.py](<../source/bpdr_input/preparation/pairs/training_support_packaging.py>)

## preparation/sources

- [data_builder.py](<../source/bpdr_input/preparation/sources/data_builder.py>)
- [data_sources.py](<../source/bpdr_input/preparation/sources/data_sources.py>)
- [dataset.py](<../source/bpdr_input/preparation/sources/dataset.py>)
- [inventory.py](<../source/bpdr_input/preparation/sources/inventory.py>)
- [jamendo_batch.py](<../source/bpdr_input/preparation/sources/jamendo_batch.py>)
- [jamendo_yield.py](<../source/bpdr_input/preparation/sources/jamendo_yield.py>)
- [separation_adapter.py](<../source/bpdr_input/preparation/sources/separation_adapter.py>)
- [shared_batch_intake.py](<../source/bpdr_input/preparation/sources/shared_batch_intake.py>)
- [shared_export_audit.py](<../source/bpdr_input/preparation/sources/shared_export_audit.py>)
- [shared_exports.py](<../source/bpdr_input/preparation/sources/shared_exports.py>)
- [verify_dataset.py](<../source/bpdr_input/preparation/sources/verify_dataset.py>)

## repair

- [framewise_audit.py](<../source/bpdr_input/repair/framewise_audit.py>)
- [framewise_renderer.py](<../source/bpdr_input/repair/framewise_renderer.py>)
- [interventions.py](<../source/bpdr_input/repair/interventions.py>)
- [model_runtime.py](<../source/bpdr_input/repair/model_runtime.py>)
- [noise_policy_v2.py](<../source/bpdr_input/repair/noise_policy_v2.py>)
- [noise_preview.py](<../source/bpdr_input/repair/noise_preview.py>)
- [prediction_adapter.py](<../source/bpdr_input/repair/prediction_adapter.py>)
- [repair_controller.py](<../source/bpdr_input/repair/repair_controller.py>)
- [rule_correction.py](<../source/bpdr_input/repair/rule_correction.py>)

## reporting

- [bpdr_showcase.py](<../source/bpdr_input/reporting/bpdr_showcase.py>)
- [notebook_evidence.py](<../source/bpdr_input/reporting/notebook_evidence.py>)
- [notebooks.py](<../source/bpdr_input/reporting/notebooks.py>)
- [paired_statistics.py](<../source/bpdr_input/reporting/paired_statistics.py>)
- [preparation_report.py](<../source/bpdr_input/reporting/preparation_report.py>)
