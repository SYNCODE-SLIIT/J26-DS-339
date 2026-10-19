# Notebooks

| Order | Notebook | Purpose |
| --- | --- | --- |
| 01 | [Input validation and analysis](01_input_validation_and_analysis.ipynb) | Original recordings, dataset coverage, waveform and acoustic features, technical acceptance |
| 02 | [Data review and paired experiments](02_data_review_and_paired_experiments.ipynb) | Comprehensive study history, paired preparation, diagnostics and all retained results |
| 03 | [Transformer training](03_transformer_training.ipynb) | Actual training data, four views, architecture, losses, schedules, all-seed learning curves and selected checkpoint |
| 04 | [Transformer evaluation](04_transformer_evaluation.ipynb) | Recomputed prediction metrics, matched controls, real-source transfer, gates, playable input/output and next validation |
| 05 | [Panel demonstration](05_panel_demonstration.ipynb) | Short executed walkthrough, selected-model inference, verified input/output, preservation, real-data failures and research limits |
| 06 | [Real-voice pilot and usefulness](06_real_voice_pilot_and_usefulness.ipynb) | Small original-data training results, separate source groups, provisional vocal review and actual accepted-only extractor connection |
| 07 | [Learned stable-note recovery](07_learned_stable_note_recovery.ipynb) | VocalSet source checks, trained ExtraTrees model, complete failed criteria and selected verified input/output audio |

For a focused walkthrough use **01 → 03 → 04**; 02 provides the full detailed study record.

For the two-minute panel demo, start with **05** and the [offline comparison](../results/panel_demo/index.html). The [self-contained notebook HTML](05_panel_demonstration.html) includes the executed plots and audio without a live kernel. Launch `../Run-Panel-Demo.ps1` for the panel interface, or see [the rehearsal runbook](../documentation/Panel_Demo_Runbook_2026-10-18.md).

For the current learned-model workflow, start with **07** and `../Run-Demo.ps1`. The ExtraTrees model provides an explicitly experimental preview; broader validation criteria remain failed. Notebook 06 preserves the earlier small-data negative result. See [the current workflow](../documentation/Working_Demo_2026-10-19.md).

## Version control and local assets

Commit notebook sources without saved outputs. Executed HTML, embedded audio/plots, datasets, weights and run receipts stay local and are ignored by Git. Before committing an executed notebook, run `../.venv/Scripts/python.exe -B tools/prepare_notebooks_for_git.py` from `BPDR_Workspace`. The helper first preserves and hashes exact executed copies under `results/notebook_output_backups/`, then clears source outputs. Existing execution receipts still describe those exact preserved copies; their old hashes should not be interpreted as hashes of the newly cleared sources.

The local executed HTML views remain available. A fresh checkout needs the separately retained datasets, pinned tools, weights and experiment receipts, or deliberate regeneration; committing the code does not distribute those assets. The official bounded VocalSet downloader is `bpdr_input.preparation.sources.vocalset_subset`.

## Open and run

From `BPDR_Workspace`, run `./Run-Notebooks.ps1`. It uses the installed component Python and opens JupyterLab in your browser. Select **BPDR local Python**, then **Restart Kernel and Run All Cells**. Keep the terminal open while using Jupyter; Ctrl+C stops it. Use `-NoBrowser` for manual browser opening and `-Port 8890` if needed.

In VS Code, select the component `.venv/Scripts/python.exe` as the kernel. The first code cell finds the source folder and reports missing packages with the required interpreter. Do not select a global Python just because it is the editor default.

`./Run-Notebooks.ps1 -Check` checks the environment without starting a server. `./Run-Notebooks.ps1 -Execute` executes every declared notebook in a fresh local kernel and saves full HTML/ipynb plus verification receipts. The previous successful execution remains the latest pointer if any notebook fails. Failure details are retained in that attempted run.

## Reading without execution

Open [results/index.html](../results/index.html) or run `../results/Open-Notebooks.ps1` for saved reading views with embedded plots and audio. These HTML files are static; use the `.ipynb` sources for executable cells. New training/evaluation notebooks load saved training and predictions; they do not rerun fitting. Notebook 01 performs one bounded technical input-analysis pass.

Historical executions, failed candidates and logs stay in `../../research_archive/evidence`. They are evidence, not disposable duplicate sources. Dataset provenance and failed real-source generalization are retained in the current explanations.
