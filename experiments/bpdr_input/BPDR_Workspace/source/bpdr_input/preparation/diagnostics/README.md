# preparation / diagnostics

- coverage.py
- development_boundary_observability.py
- review_bundle.py
- shared_screen_diagnostics.py
- support_acoustic_pilot.py
- support_development_audit.py
- support_fragmentation.py
- support_geometry.py
- training_support_audit.py

Files live here once. Current source uses explicit subpackage imports, such as `bpdr_input.audio.pipeline` or `bpdr_input.modeling.model`. The former flat imports remain compatibility aliases to the same module objects.
