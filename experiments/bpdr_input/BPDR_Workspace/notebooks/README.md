# Research notebooks

Open [index.html](index.html) to search code blocks, view executed output or expand source functions. Run `../Open-Notebooks.ps1` to open the local index.

| Order | Notebook |
|---|---|
| 01 | [Research foundations and music fundamentals](01_research_foundations.ipynb) |
| 02 | [BPDR methodology, Transformer and objectives](02_bpdr_methodology.ipynb) |
| 03 | [Data preparation, features and interventions](03_data_preparation.ipynb) |
| 04 | [Model training, comparison and evaluation](04_model_training_and_evaluation.ipynb) |
| 05 | [Audio processing, verification and handoff](05_audio_processing_and_handoff.ipynb) |
| 06 | [Evidence verification, statistics and tests](06_verification_and_statistics.ipynb) |

Use the component `.venv` with the **BPDR local Python** kernel. Results, interpretation, plots and audio are visible on first open. Code is collapsed in HTML and marked collapsed in the notebook; expand it only when needed. Run all cells in a topic once before rerunning a dependent block. No fitting or dataset downloads run automatically.

Search IDs: `BPDR-LOSSES`, `DATA-CLEAN`, `DATA-FEATURES`, `MODEL-ACCURACY`, `MODEL-GATES`, `DEMO-RUN`, `RENDER-AUDIT`, `HANDOFF-EXTRACTOR`, `VERIFY-TESTS`.

Verified: 6 notebooks, 125 code cells, 0 errors, 65 passing tests, 14 plots and 8 audio players. Scientific acceptance failures remain visible.

Run `../.venv/Scripts/python.exe -X utf8 -B tools/execute_research_notebooks.py` from BPDR_Workspace to verify these six notebooks. Current execution receipt: `results/research_notebooks/current.json`. Superseded notebooks and exact user-edited originals were checksum-verified and archived under the location recorded in `results/notebook_cleanup/current.json`; they are no longer in the active notebook folder.

See [environment pins](../documentation/requirements-notebooks.txt), [usage guide](../documentation/NOTEBOOK_GUIDE.md), and [verification report](../documentation/EXPERIMENT_VERIFICATION_REPORT.md). Data, weights, historical receipts and generated outputs remain local and ignored by Git.
