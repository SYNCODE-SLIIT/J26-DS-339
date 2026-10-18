# preparation / pairs

- development_preparation.py
- laboratory.py
- shared_pair_audit.py
- support_development_packaging.py
- training_reference_completion.py
- training_reference_completion_v2.py
- training_reference_preparation.py
- training_support_packaging.py

Files live here once. Current source uses explicit subpackage imports, such as `bpdr_input.audio.pipeline` or `bpdr_input.modeling.model`. The former flat imports remain compatibility aliases to the same module objects.
