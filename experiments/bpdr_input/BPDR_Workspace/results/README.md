# Current implementation and results

Open **[index.html](index.html)**. Start the local system with `Run-System.ps1` in this folder, then use http://127.0.0.1:8501.

| Item | Purpose |
| --- | --- |
| [Notebook 1](reports/01_input_validation_and_analysis.html) | Original Vocadito coverage, data inspection, acoustic features, technical checks and unchanged acceptance |
| [Notebook 2](reports/02_data_review_and_paired_experiments.html) | Dataset roles, preparation, four-view Transformer, real-data adaptation, input/output and all measured results |
| [Implementation map](CODE_MAP.md) | Current application, inference, model, training, data helpers and retained experiments |
| [Source application](../source/bpdr_input/application/app.py) | Recording/upload interface and verified recorded comparisons |
| [Result reader](../source/bpdr_input/application/saved_results.py) | Checks saved input/output identity and computes paired noise improvement |
| [Validation guide](VALIDATION.md) | Metric meanings, implemented requirements, best practices and remaining validation |
| [Current artifacts](CURRENT_ARTIFACTS.json) | Pinned files, checksums and example provenance |

## Dataset and example identity

Vocadito is the proposed original dataset; Jamendo contributes separated vocals and predicted metadata. The selected Transformer uses synthetic pretraining followed by Vocadito/Jamendo adaptation with synthetic replay. HumTrans is an additional real-humming source; its later learning candidate was not selected.

The local pitch-error comparison is a generated phrase with a known injected shift. The noise comparison uses real HumTrans audio mixed with recorded rain. These are useful checked examples, not general real-singer accuracy. The notebook retains the full evaluation, unsuccessful seeds/candidates and relevant validation limits.

## Reading views and full record

The report views omit executable cell inputs for readability and retain every markdown cell and result output. The only two maintained notebook sources remain in [notebooks/](../notebooks/). Full executed copies, code and historical outputs remain under `runs/notebooks/`; no result or log was rewritten to improve the reported performance.

Original accepted downstream audio remains unchanged. Verified pitch/noise candidates are separate playable/downloadable previews. No reference audio is supplied to live inference. Input/output must be compared at the same playback volume.

## Use and maintain

Run `../Run-Notebooks.ps1` to edit sources. `Open-Notebooks.ps1` opens saved reading views. After a deliberate source change, execute both notebooks with the component runner and then run `python -P -B BPDR_Workspace/results/build_views.py` in the existing environment. This refreshes views and their checksum manifest without fitting models or changing research decisions.

Complete notes and proposal: [docs/](../documentation/README.md). Experiments/logs: [runs/](../../research_archive/evidence/README.md). Historical module names and compatibility launchers are preserved for recovery and reproducibility; they are not the current entry point.
