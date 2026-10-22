# repair

- framewise_audit.py
- framewise_renderer.py
- interventions.py
- model_runtime.py
- noise_policy_v2.py
- noise_preview.py
- prediction_adapter.py
- repair_controller.py
- rule_correction.py

Files live here once. Current source uses explicit subpackage imports, such as `bpdr_input.audio.pipeline` or `bpdr_input.modeling.model`. The former flat imports remain compatibility aliases to the same module objects.
