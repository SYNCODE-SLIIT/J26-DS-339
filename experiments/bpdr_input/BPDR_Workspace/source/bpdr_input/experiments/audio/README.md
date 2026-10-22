# experiments / audio

- anchor_feasibility.py
- audit_diagnostics.py
- boundary_comparison.py
- boundary_renderer.py
- context_ablation.py
- context_ablation_experiment.py
- context_ablation_scoring.py
- controller_fixtures.py
- demo.py
- estimated_harmonic_experiment.py
- estimated_harmonic_renderer.py
- framewise_diagnostics.py
- framewise_fixtures.py
- harmonic_separation.py
- harmonic_separation_experiment.py
- known_control_diagnostics.py
- known_source_control.py
- phase_comparison.py
- phase_feasibility.py
- phase_regression_diagnostics.py
- phase_renderer.py
- prediction_preview.py
- renderer_calibration.py
- renderer_candidates.py
- segment_phase.py
- segment_phase_experiment.py
- segment_phase_scoring.py
- source_estimation_experiment.py
- source_estimator.py
- source_transfer.py

Files live here once. Current source uses explicit subpackage imports, such as `bpdr_input.audio.pipeline` or `bpdr_input.modeling.model`. The former flat imports remain compatibility aliases to the same module objects.
