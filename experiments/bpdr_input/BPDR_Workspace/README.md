# BPDR Workspace

This is the working project. Open [results/index.html](results/index.html), or run `Run-System.ps1` here. The source, original datasets, selected model, pinned model tools, notebooks, tests and documentation are physically under this folder.

**Current development — 28 October 2026:** use the [master plan](documentation/BPDR_Master_Plan.md) and [tracker](documentation/BPDR_Implementation_Tracker.md). **MOD-03 remains ACTIVE:** the [latest Transformer result](documentation/Transformer_Usability_Result_2026-10-28.md) improves 5/48 independent attempts and preserves all 12 originals, but fails usable coverage across four singers. The guarded adapter and repeatable accuracy check work; the candidate remains inactive and the demo remains ExtraTrees. Later features remain planned. The main research goal remains a matched test of boundary-invariance/counter-detuning learning, preservation and downstream extraction benefit.

For the working demo, run `Run-Demo.ps1`. Start with the **experimental learned repair** real-voice example and press **Process recording** to compare actual input and output. This uses trained ExtraTrees weights. The historical Transformer, conservative baseline, original preservation and noise reduction remain available. The app does not train while presenting. The [offline comparison](results/panel_demo/index.html) and [panel instructions](documentation/Panel_Demo_Runbook_2026-10-18.md) retain earlier evidence; they are separate from the new context-Transformer experiment.

The [existing workflow](documentation/Working_Demo_2026-10-19.md) includes provisional vocal-evidence review, explicit intention, a separate user-reviewed WAV package and an accepted-only GAME connection. In the earlier ExtraTrees delivered-WAV pilot, two of four scoreable interventions improved and four originals were preserved; twelve of sixteen attempts were unscorable and broader criteria failed. The separate [27 October Transformer experiment](documentation/Transformer_Recovery_2026-10-27.md) reached 5.839-cent raw error on ten of sixteen calibration attempts, but its recorded permission replay produced no useful correction on the scored shifted regions. Keep these denominators/results separate. Neither is a validated general singing model.

Use the [canonical six-notebook index](notebooks/index.html) for foundations, methodology, data, training/evaluation, audio/handoff and verification. Old notebook 07 and former 05/06 filenames in historical reports were superseded; see the [notebook guide](documentation/NOTEBOOK_GUIDE.md).

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
| `selected_model` | Historical selected Transformer weights; the current learned demo uses its separately guarded ExtraTrees card |
| `models_and_tools` | Pinned acoustic/separator/GAME models and component-local tool runtimes |
| `notebooks` | Input analysis, full study history, Transformer training/evaluation, panel demo and current real-data pilot |
| `results` | Executed reading views, implementation map and artifact checksums |
| `tests` | Verification sources |
| `documentation` | Current guides, proposal, plans and historical reports |
| `tools` | Extraction adapters and specialist launchers |

## Start and inspect

`Run-System.ps1` opens the laboratory interface; `Run-Demo.ps1` opens the focused working demo. `Open-Notebooks.ps1` opens the canonical notebook index. [Implementation map](results/CODE_MAP.md) identifies source responsibilities and explains historical module names; inspect actual source when using newer additions.

The current notebook order is **01 foundations → 02 methodology → 03 data → 04 training/evaluation → 05 audio/handoff → 06 verification**. From BPDR_Workspace, run `../.venv/Scripts/python.exe -X utf8 -B tools/execute_research_notebooks.py` for the canonical verification workflow. See [notebook instructions](notebooks/README.md) for kernel selection and explicit opt-in fitting steps.

The historical selected Transformer uses synthetic pretraining with real Vocadito/Jamendo adaptation. The guarded ExtraTrees preview and the new context Transformer use separate declared VocalSet investigations. Controlled pitch-repair and noise-cleaning examples are measured; general real-voice accuracy and a BPDR advantage remain unproven. The original downstream working copy is retained; candidates are separate.

Older outputs, execution logs, recovery copies and references live in `../research_archive`. Original root folder paths remain compatibility junctions, so saved absolute paths continue to reach the same files. They are not duplicate datasets or source copies. Keep those compatibility links while old evidence is in use. The Python environment/configuration remains component-local at the parent level. This is one organized local workspace, not a portable environment distribution.
