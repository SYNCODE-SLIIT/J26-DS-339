# preparation / sources

- data_builder.py
- data_sources.py
- dataset.py
- inventory.py
- jamendo_batch.py
- jamendo_yield.py
- separation_adapter.py
- shared_batch_intake.py
- shared_export_audit.py
- shared_exports.py
- verify_dataset.py

Files live here once. Current source uses explicit subpackage imports, such as `bpdr_input.audio.pipeline` or `bpdr_input.modeling.model`. The former flat imports remain compatibility aliases to the same module objects.
