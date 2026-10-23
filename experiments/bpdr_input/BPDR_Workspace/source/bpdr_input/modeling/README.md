# modeling

- examples.py
- losses.py
- metrics.py
- model.py
- training.py

Files live here once. Current source uses explicit subpackage imports, such as `bpdr_input.audio.pipeline` or `bpdr_input.modeling.model`. The former flat imports remain compatibility aliases to the same module objects.
