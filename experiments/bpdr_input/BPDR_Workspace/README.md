# BPDR Workspace

This is the working project. Open [results/index.html](results/index.html), or run `Run-System.ps1` here. The source, original datasets, selected model, pinned model tools, notebooks, tests and documentation are physically under this folder.

For the working demo, run `Run-Demo.ps1`. Start with the **experimental learned repair** real-voice example and press **Process recording** to compare actual input and output. This uses a trained ExtraTrees model and saved weights. The historical Transformer, conservative baseline, original preservation and noise reduction remain available. The app does not train while presenting. [Notebook 05](notebooks/05_panel_demonstration.ipynb), the [offline comparison](results/panel_demo/index.html) and [panel instructions](documentation/Panel_Demo_Runbook_2026-10-18.md) retain earlier evidence; they do not contain the new learned-model result.

The [current workflow](documentation/Working_Demo_2026-10-19.md) includes provisional vocal-evidence review, explicit intention, a separate user-reviewed WAV package and an accepted-only GAME connection. Start with [notebook 07](notebooks/07_learned_stable_note_recovery.ipynb) ([executed HTML](notebooks/07_learned_stable_note_recovery.html)) for the new original-VocalSet experiment and audible model output. Two of four scoreable controlled interventions improved; four originals were preserved. Twelve of sixteen attempted interventions were unscorable, and broader validation criteria failed. This is an experimental preview, not a validated general singing model. [Notebook 06](notebooks/06_real_voice_pilot_and_usefulness.ipynb) preserves the earlier original-data pilot.

| Folder | Contents |
| --- | --- |
| `source/bpdr_input/application` | Current interface and verified input/output reader |
| `source/bpdr_input/audio` | Validation, acoustic analysis and handoff contracts |
| `source/bpdr_input/modeling` | Transformer, losses, training and metrics |
| `source/bpdr_input/repair` | Runtime correction, noise cleaning, permission and rollback |
| `source/bpdr_input/preparation` | Source intake, paired-data preparation and diagnostics |
| `source/bpdr_input/experiments` | Retained learning, audio and evaluation investigations |
| `source/bpdr_input/reporting` | Notebook execution, evidence reading and result reports |
| `source/bpdr_input/packaging` | Review/archive builders; not training models |
| `datasets` | Original/derived data and links to frozen training/evaluation material |
| `selected_model` | Working BPDR weights, verified against the immutable archive |
| `models_and_tools` | Pinned acoustic/separator/GAME models and component-local tool runtimes |
| `notebooks` | Input analysis, full study history, Transformer training/evaluation, panel demo and current real-data pilot |
| `results` | Executed reading views, implementation map and artifact checksums |
| `tests` | Verification sources |
| `documentation` | Current guides, proposal, plans and historical reports |
| `tools` | Extraction adapters and specialist launchers |

## Start and inspect

`Run-System.ps1` starts recording/upload, input/output comparison and saved results. `Run-Notebooks.ps1` opens editable notebook sources. [Implementation map](results/CODE_MAP.md) identifies every source file and explains the renamed `presentation_*` modules. They were a mixture of interface, model evaluation, statistics and packaging; they were not all notebook-generation code.

For the focused notebook walkthrough, follow **01 → 03 → 04**. Notebook 02 contains the detailed experiment history. `Run-Notebooks.ps1 -Check` verifies the environment; `-Execute` reruns the notebook suite without fitting models. See [notebook instructions](notebooks/README.md) for JupyterLab and VS Code kernel selection.

The selected model uses synthetic pretraining with real Vocadito/Jamendo adaptation. Controlled pitch-repair and real-source noise-cleaning examples are measured. General real-voice pitch accuracy and a BPDR advantage remain unproven. The original downstream working copy is retained; candidates are separate.

Older outputs, execution logs, recovery copies and references live in `../research_archive`. Original root folder paths remain compatibility junctions, so saved absolute paths continue to reach the same files. They are not duplicate datasets or source copies. Keep those compatibility links while old evidence is in use. The Python environment/configuration remains component-local at the parent level. This is one organized local workspace, not a portable environment distribution.
