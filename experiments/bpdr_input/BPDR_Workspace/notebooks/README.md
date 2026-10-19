# Notebooks

| Order | Notebook | Purpose |
| --- | --- | --- |
| 01 | [Input validation and analysis](01_input_validation_and_analysis.ipynb) | Original recordings, dataset coverage, waveform and acoustic features, technical acceptance |
| 02 | [Data review and paired experiments](02_data_review_and_paired_experiments.ipynb) | Comprehensive study history, paired preparation, diagnostics and all retained results |
| 03 | [Transformer training](03_transformer_training.ipynb) | Actual training data, four views, architecture, losses, schedules, all-seed learning curves and selected checkpoint |
| 04 | [Transformer evaluation](04_transformer_evaluation.ipynb) | Recomputed prediction metrics, matched controls, real-source transfer, gates, playable input/output and next validation |
| 05 | [Panel demonstration](05_panel_demonstration.ipynb) | Short executed walkthrough, selected-model inference, verified input/output, preservation, real-data failures and research limits |
| 06 | [Real-voice pilot and usefulness](06_real_voice_pilot_and_usefulness.ipynb) | Small original-data training results, separate source groups, provisional vocal review and actual accepted-only extractor connection |
| 07 | [Complete model walkthrough](07_learned_stable_note_recovery.ipynb) | Goal and model-change history; data cleaning/splits/labels; all 31 features; actual fitting and candidate selection; defined metrics and failed criteria; fresh audible output; export and actual downstream MIDI/CSV |

For a focused walkthrough use **01 → 03 → 04**; 02 provides the full detailed study record.

For the two-minute panel demo, start with **05** and the [offline comparison](../results/panel_demo/index.html). The [self-contained notebook HTML](05_panel_demonstration.html) includes the executed plots and audio without a live kernel. Launch `../Run-Panel-Demo.ps1` for the panel interface, or see [the rehearsal runbook](../documentation/Panel_Demo_Runbook_2026-10-18.md).

For the current learned-model workflow, start with **07** and `../Run-Demo.ps1`. The ExtraTrees model provides an explicitly experimental preview; broader validation criteria remain failed. Notebook 06 preserves the earlier small-data negative result. See [the current workflow](../documentation/Working_Demo_2026-10-19.md).

Notebook 07 is a complete 20-section explanation, with 44 cells, ten executed plots and five audio players. Its [HTML reading view](07_learned_stable_note_recovery.html) includes all code, explanations and outputs. `results/recovery_walkthrough/current.json` identifies the latest complete executed `.ipynb`, its source and hashes. Section **12B** compares the historical Transformer, frozen MLP, histogram gradient boosting and ExtraTrees on the same audited calibration examples, reporting MAE, accuracy within 5/10 cents, original alteration and permitted coverage. Section 3 explains the model change through observed correction response and preservation, without attributing it to dataset size. The original calibration ranking is retained: ExtraTrees was selected as the highest-ranked diagnostic fallback after all candidates failed broader criteria. Different training histories and reuse of calibration examples prevent general architecture-superiority or final-test accuracy claims.

The supplementary comparison receipt and frozen predictions are under `results/model_comparison/current.json`. `tools/compare_recovery_models.py` evaluates the saved checkpoints and reconstructs the previously unsaved HGB estimator from the exact original training arrays/settings, checking its recorded calibration metrics. That explicit tool performs one HGB fit; notebook Run All only verifies and rescores its saved predictions. Existing model selection, dataset sources and research-adoption flags are preserved.

## Version control and local assets

Commit notebook sources without saved outputs. Executed HTML, embedded audio/plots, datasets, weights and run receipts stay local and are ignored by Git. Before committing an executed notebook, run `../.venv/Scripts/python.exe -B tools/prepare_notebooks_for_git.py` from `BPDR_Workspace`. The helper first preserves and hashes exact executed copies under `results/notebook_output_backups/`, then clears source outputs. Existing execution receipts still describe those exact preserved copies; their old hashes should not be interpreted as hashes of the newly cleared sources.

The local executed HTML views remain available. A fresh checkout needs the separately retained datasets, pinned tools, weights and experiment receipts, or deliberate regeneration; committing the code does not distribute those assets. The official bounded VocalSet downloader is `bpdr_input.preparation.sources.vocalset_subset`.

`tools/execute_recovery_notebook.py` now keeps the notebook 07 source untouched and writes its complete executed copy to an ignored, unique `results/recovery_walkthrough/` run folder. It exports the local HTML reading view and an execution receipt. Manual Jupyter saves can still add outputs to source; use the preservation helper before committing those saves. `tools/build_complete_recovery_notebook.py` maintains the full explanatory source, and `tools/update_recovery_notebook.py` delegates to that builder.

## Open and run

From `BPDR_Workspace`, run `./Run-Notebooks.ps1`. It uses the installed component Python and opens JupyterLab in your browser. Select **BPDR local Python**, then **Restart Kernel and Run All Cells**. Keep the terminal open while using Jupyter; Ctrl+C stops it. Use `-NoBrowser` for manual browser opening and `-Port 8890` if needed.

In VS Code, select the component `.venv/Scripts/python.exe` as the kernel. The first code cell finds the source folder and reports missing packages with the required interpreter. Do not select a global Python just because it is the editor default.

`./Run-Notebooks.ps1 -Check` checks the environment without starting a server. `./Run-Notebooks.ps1 -Execute` executes every declared notebook in a fresh local kernel and saves full HTML/ipynb plus verification receipts. The previous successful execution remains the latest pointer if any notebook fails. Failure details are retained in that attempted run.

## Reading without execution

Open [results/index.html](../results/index.html) or run `../results/Open-Notebooks.ps1` for saved reading views with embedded plots and audio. These HTML files are static; use the `.ipynb` sources for executable cells. New training/evaluation notebooks load saved training and predictions; they do not rerun fitting. Notebook 01 performs one bounded technical input-analysis pass.

Historical executions, failed candidates and logs stay in `../../research_archive/evidence`. They are evidence, not disposable duplicate sources. Dataset provenance and failed real-source generalization are retained in the current explanations.
