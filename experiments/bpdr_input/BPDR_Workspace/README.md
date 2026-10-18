# BPDR Workspace

This is the working project. Open [results/index.html](results/index.html), or run `Run-System.ps1` here. The source, original datasets, selected model, pinned model tools, notebooks, tests and documentation are physically under this folder.

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
| `notebooks` | Exactly two maintained notebook sources |
| `results` | Executed reading views, implementation map and artifact checksums |
| `tests` | Verification sources |
| `documentation` | Current guides, proposal, plans and historical reports |
| `tools` | Extraction adapters and specialist launchers |

## Start and inspect

`Run-System.ps1` starts recording/upload, input/output comparison and saved results. `Run-Notebooks.ps1` opens editable notebook sources. [Implementation map](results/CODE_MAP.md) identifies every source file and explains the renamed `presentation_*` modules. They were a mixture of interface, model evaluation, statistics and packaging; they were not all notebook-generation code.

The selected model uses synthetic pretraining with real Vocadito/Jamendo adaptation. Controlled pitch-repair and real-source noise-cleaning examples are measured. General real-voice pitch accuracy and a BPDR advantage remain unproven. The original downstream working copy is retained; candidates are separate.

Older outputs, execution logs, recovery copies and references live in `../research_archive`. Original root folder paths remain compatibility junctions, so saved absolute paths continue to reach the same files. They are not duplicate datasets or source copies. Keep those compatibility links while old evidence is in use. The Python environment/configuration remains component-local at the parent level. This is one organized local workspace, not a portable environment distribution.
