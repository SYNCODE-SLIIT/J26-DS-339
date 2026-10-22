# BPDR vocal input workspace

**Use [BPDR_Workspace](BPDR_Workspace/README.md).** Open [the results index](BPDR_Workspace/results/index.html), or run `./Run-System.ps1`.

- `BPDR_Workspace/`: actual source, datasets, selected model, tools, notebooks, tests, documentation and current result views.
- `research_archive/`: all older runs/logs, recovery records, references and legacy launchers.
- `Run-System.ps1`, `Run-Notebooks.ps1`, `Open-Results.ps1`: short convenience entry points.
- `.venv`, `.tools`, `.jupyter`, `.streamlit`, `pyproject.toml`, `uv.lock`: retained local runtime/configuration.

The old `src`, `data`, `checkpoints`, `notebooks`, `tests`, `docs`, `extraction`, `final`, `runs` and `private` directories are compatibility junctions to their new physical locations. They contain no separate copies. Existing absolute dataset/evidence paths still resolve. Do not remove those links casually or clean through both paths as though they were different datasets.

The source package root now has only its initialization file and clear responsibility folders. The ambiguous `presentation_*` filenames were renamed; stable old imports remain compatible. No historical log, research result or original dataset was deleted to create this layout. See [the migration record](runs/reorganization/current.json) for hashes, recovery and the path map.
