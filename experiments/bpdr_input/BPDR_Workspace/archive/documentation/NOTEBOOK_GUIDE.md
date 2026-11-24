# Notebook guide

Six independently executable topics cover foundations, methodology, data, models, audio processing and verification. Start with `notebooks/index.html` or `Open-Notebooks.ps1`. Search a code-block ID or function name; source cards show file paths, functions and line numbers.

Use saved HTML or the executed notebook to read existing outputs. Read each explanation, result and interpretation in order; expand optional code only when needed. The full implementation remains in source modules and optional index source cards, rather than being printed under result cells. To execute, open `Run-Notebooks.ps1`, select BPDR local Python and run the topic from its setup cells. `Open-Notebooks.ps1 -Verify` explicitly re-executes the six notebooks before opening the index. No training starts automatically.

Notebook 04 now includes explicit `MODEL-FIT-NEW`, `MODEL-FIT-EXECUTE`, `MODEL-SAVE-CHECKPOINT`, `MODEL-RELOAD-CHECKPOINT`, `MODEL-VERIFY-DEMO` and `MODEL-ACTIVATE-DEMO` steps. To deliberately fit a new fixed ExtraTrees reproduction, set `TRAIN_NEW_MODEL = True` in the first step, or launch verification with `BPDR_TRAIN_NOTEBOOK_MODEL=1`. Default execution reloads the previously fitted notebook checkpoint. Each fit writes a new directory under `results/model_recovery/notebook_training`; it never overwrites the historical checkpoint.

After train-only fitting, save/reload checks and a complete delivered-audio development replay, a separate `results/notebook_training/current.json` pointer selects the model used by both notebook 05 and the working demo app. Existing calibration/pilot data and locked settings are reused: this is reproducibility/development evidence, not an independent final test. The demo remains experimental and user review is required before export.

Model training remains deliberate: `python -B -m bpdr_input.experiments.learning.model_recovery` creates a new declared run. Keep earlier experiments and partitions unchanged. These are engineering development results; natural intended-note accuracy, blinded listening and downstream note-accuracy improvements remain unestablished.

The local-file browser preview could not be automated under the browser tool policy. Offline links and JavaScript search can be checked separately; test local browser/audio playback on the actual machine. This limitation does not change recorded model metrics.
