# experiments / evaluation

- baseline_evaluation.py
- cue_features.py
- environmental_noise_study.py
- generalization_probe.py
- humming_experiment.py
- humming_intake.py
- humming_intake_extension.py
- noise_study.py
- reference_cue_ceiling.py
- rule_evaluation.py
- runtime_cue_diagnosis.py
- transfer_study.py

Files live here once. Current source uses explicit subpackage imports, such as `bpdr_input.audio.pipeline` or `bpdr_input.modeling.model`. The former flat imports remain compatibility aliases to the same module objects.
