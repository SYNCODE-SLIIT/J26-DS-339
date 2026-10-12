# BPDR vocal input component — PP1 experiments

Owner: Liyanage L. N. P., IT23141506. All component changes stay in `experiments/bpdr_input`.

Phase 1 implements timing-preserving technical checks and frozen acoustic analysis. It preserves original bytes, creates a mono 22,050 Hz working copy, measures performed pitch and exports checked accepted audio. It does not infer intended notes, train BPDR, repair pitch or establish musical-usability accuracy. `ACCEPT_UNCHANGED` means no musical correction; channel selection, resampling and PCM encoding are logged technical operations.

## Run a recording

From this folder in PowerShell:

```powershell
.\Run-Phase1.ps1 -InputAudio '.\data\vocadito\Audio\vocadito_10.wav'
```

The local setup includes an ignored portable uv in `.tools`. On another machine, install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. The script selects this component project and uses its own environment. Python 3.12 or newer is required; `uv.lock` records dependencies. First setup needs network access for dependencies. PESTO's packaged checkpoint is then available offline.

With uv on PATH, equivalent commands are:

```powershell
uv sync --locked
uv run --locked bpdr-process 'C:\path\recording.wav'
uv run --locked python -m unittest discover -s tests -v
uv run --locked bpdr-data-audit --dataset-root 'data\vocadito' --output 'runs\dataset_inventory.json'
uv run --locked python -m bpdr_input.verify_dataset --dataset-root 'data\vocadito'
```

Each call prints its status and a new `runs/<run-id>` directory. Open `report.html` in a browser for original/output playback and plots on accepted runs. Inspect `diagnostics.json` for reasons on stopped runs. Exit code is 0 for accepted audio and 1 for a controlled stop/failure. Raw recordings stay in this folder's ignored `data/` directory (`data/vocadito`) and are never committed; pass another `--dataset-root`, or set `BPDR_DATASET_ROOT` for the notebooks, to use a copy elsewhere.

## Supported behavior

- WAV/FLAC, mono/stereo, 0.25–20 seconds. Longer recordings stop without cropping. Phase 2 will create explicit mapped phrase crops.
- Empty/silent, overloaded and out-of-scope recordings request re-recording. Corrupt, invalid-sample, unsupported-format or software/analyzer failures return `TECHNICAL_FAILURE`.
- Stereo selects the highest-RMS channel to avoid antiphase cancellation. This provisional policy does not separate sources or detect solo vocals.
- Resampling retains time origin and internal rests. Rounded duration differs by at most one target-rate sample. Resampling overshoot receives logged peak-protection gain if needed.
- Accepted audio is mono PCM24 WAV. The original bytes/SHA-256 remain separate. `validate_package` checks actual checksum, metadata, finite samples and bounds before downstream use. Stopped results have no accepted path/export.
- Phase 1 emits `ACCEPT_UNCHANGED`, `RERECORD_REQUIRED` and `TECHNICAL_FAILURE`. `ACCEPT_PREPARED` is reserved for later verified musical repair.

Silence RMS (1e-5), selected-channel clipping limit (1% at absolute sample >=0.999), duration scope and voicing confidence are provisional engineering settings, not calibrated research results. A noisy or musically unsuitable recording may still pass these technical checks. Acceptance does not prove intent, musical correctness or monophony.

## Saved acoustics and conventions

`features.npz` contains performed F0 in Hz, native pitch timestamps/confidence, aligned voicing, pitch-change cents/validity, 64 log-mel bands, energy, onset and a provisional boundary score. Aligned rows use `times_seconds`, starting at original time zero with a 441-sample/20 ms hop. Log-mel and energy use centered 1024-sample windows; edge padding does not extend the exported recording.

PESTO pitch maps by nearest native timestamp without interpolation across unvoiced gaps. Unvoiced plotted F0 is NaN, never a zero-Hz note. Model features use masked zero pitch with confidence/voicing retained. Pitch changes are valid only across adjacent voiced frames. Boundary scores combine positive spectral flux, pitch change and voicing transitions: heuristics, not final note boundaries.

Feature order: 64 log-mel dB, pitch cents relative to A4=440 Hz, confidence, voiced flag, pitch-change cents, log-energy dB, onset score = 70 features. Appending provisional boundary gives 71. Normalization, targets and training belong to later phases.

Frozen `mir-1k_g7` PESTO loads once per process, runs on CPU in evaluation/inference mode and disables gradients. Diagnostics record dependency versions and checkpoint SHA-256. See [official PESTO](https://github.com/SonyCSLParis/pesto), [SoundFile docs](https://python-soundfile.readthedocs.io/) and [librosa docs](https://librosa.org/doc/latest/).

## Research and assessment

Read the [P0 scope, claims and draft hand-off](docs/P0_Scope_Claims_and_Decisions.md), [master plan](docs/PP1_Master_Plan.md), [phase plan](docs/PP1_Phase_Execution_Plan.md), [evidence tracker](docs/PP1_Evidence_Tracker.md) and [Phase 1 verification](docs/Phase1_Verification.md). The phase plan carries an 11 October status and corrective schedule. The local P0 scope is documented; teammate acceptance of the interface and actual data-sharing permission remain open.

Phase 1 demonstrates the foundation of FR1/FR2/FR6/FR7. BPDR repair, usability calibration, response losses, model comparisons and team integration remain later work. Dataset execution counts are not accuracy or completion percentages. Unit tests use injected pitch fixtures for controlled failure/alignment checks; a separate smoke test uses real frozen PESTO. Full-dataset replay uses real PESTO and no annotation targets.

## Phase 2 and notebooks

The Phase 2 laboratory validates F0/A1/A2 references, partitions by singer before cropping, creates mapped <=20-second crops, prepares two-reviewer sheets and generates audited A/B/C/D interventions. It has no trained model. The current pilot is exploratory; human approvals are pending and training eligibility remains zero. See [Phase 2 verification](docs/Phase2_Verification.md) and [the review guide](docs/Reference_Review_Guide.md).

At the 30 September status check, the 57 mapped crops comprise 30 training, 14 development, 9 PP1-evaluation and 4 final-holdout crops; all 114 review rows remain pending. The 18 exploratory pairs come from six shifts and three boundary corruptions of **one phrase from one singer**, not 18 independent performances. The laboratory boundary feature is built from A1 annotation timing, while the operating pipeline emits an audio-derived provisional boundary score. Controlled pair integrity does not establish runtime boundary robustness. Model/loss code can be checked on constructed fixtures while reviewers work, but research training and comparative claims require actual independent approvals and audit-passing examples.

Launch the notebooks from this component folder:

```powershell
.\Run-Notebooks.ps1
```

Open the local URL printed by Jupyter and select the **BPDR component (uv)** kernel. `01_input_validation_and_analysis.ipynb` covers technical input, playback and performed pitch. `02_data_review_and_paired_experiments.ipynb` covers source quality, splits, review, controlled boundary changes and measured detuning. Run All works from a fresh kernel. Sources have no saved outputs; executed copies and HTML exports are generated under ignored `runs/notebooks/`. The launcher keeps kernel registration local to this component.

With uv on PATH:

```powershell
uv sync --locked --group notebooks
uv run --locked --group notebooks python -m bpdr_input.notebooks execute --dataset-root 'data\vocadito'
uv run --locked bpdr-lab build --dataset-root 'data\vocadito'
```

`runs/phase2/latest.json` points to a data run with `manifest.json`, `data_summary.json`, `reviews.csv` and `crops/`. The build command creates a new immutable run: keep the old run if it contains reviews. Inspect its printed path and run:

```powershell
# Replace the path below with the actual data run; <data-run> is a placeholder.
uv run --locked bpdr-lab interventions --data-run '<data-run>' --exploratory --limit 1
# After two actual reviews: omit --exploratory to enforce the review gate.
uv run --locked bpdr-lab interventions --data-run '<data-run>' --limit 2
```

An unreviewed collection produces no training-ready examples. Two distinct complete approvals, their interval intersection and a successful realized-shift audit are required. `require_training_example` refuses exploratory/failed examples and checks actual waveforms, review/audit/target integrity and paired inputs. References and correction targets stay separate from model-input arrays. Development/evaluation/final holdout are not used for these initial renderer pilots.

The intervention renderer applies a constant local cents shift with waveform crossfades and exact outside-support samples. It is an experiment generator, not the completed repair renderer. Ramp targets are deliberately excluded. Audit v0.3 uses independent YIN for fine shifts (5-cent median, 10-cent 90th-percentile errors), PESTO for coarse consistency (20/40 cents), and explicit plateau/outside coverage and outside-drift-tail checks. The earlier v0.1 failure and v0.2 pilot remain recorded. The 30 September current run passed six shifts and 18 pairs, but all examples remain exploratory because reviewer decisions are pending. One singer and repeated boundary variants cannot establish accuracy or novelty. A check on 11 October showed why the pilot stays at one singer: audit v0.3 needs a steady note of about 1.06 seconds, and only 3 of the 30 training crops, all from that singer, have one. The phase plan records the decision this requires.

## Phase 3 model code

The three learned conditions share one small encoder (`model.py`), the proposal's losses (`losses.py`), a loader (`examples.py`), measurements (`metrics.py`) and a training command (`training.py`). All of it is checked on constructed fixtures in `tests/test_phase3.py`. No model has been trained to a result.

```powershell
# Replace <pairs-run> with a Phase 2 experiment folder that holds summary.json.
.\.venv\Scripts\python.exe -m bpdr_input.training --pairs-run '<pairs-run>' --condition bpdr --mode engineering
```

`--condition` is `augmentation_only`, `frame_only` or `bpdr`. The default `--mode research` accepts only reviewed, audit-approved pairs and therefore refuses everything today. `--mode engineering` reads the unreviewed pilot pairs for debugging, writes under `runs/engineering` and marks every output `exploratory_not_research`. Each run saves a checkpoint and a record with its seed, settings, parameter count and input hashes.

Training is about 18 times faster on an Intel Arc GPU. Use the GPU environment and `--device xpu`:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\checkpoints\melody-extraction\.venv-xpu\Scripts\python.exe -m bpdr_input.training --pairs-run '<pairs-run>' --condition bpdr --mode engineering --device xpu --output-root runs
```

## Second data source and melody extraction

Since 11 October the component also holds 1,604 vocal-tagged MTG-Jamendo tracks and an unedited copy of the team's melody-extraction pipeline. These are full accompanied songs with predicted notes only. They supplement vocadito and do not replace it; read the [extraction check](docs/Jamendo_Extraction_Verification.md) before using them.

```powershell
.\Setup-Jamendo.ps1                      # mirror metadata and pipeline code into data\mtg-jamendo
.\Run-Extraction.ps1 -IndexOnly          # write manifest.csv and catalog.json; needs no model
.\Setup-ExtractionTools.ps1 -Stage Full  # download pinned FFmpeg, GAME, separator, SheetSage2 and MERT (about 3.5 GB plus packages)
.\Run-Extraction.ps1 -First '1079195,1348174' -KeepStems -Threads 8   # complete pipeline on a named sample
.\Run-Extraction.ps1 -Limit 2 -VocalOnly -KeepStems                   # next two songs, vocal notes and stems only
```

`Setup-Jamendo.ps1` copies from the repository's `datasets/mtg-jamendo` and expects the MP3 folders in `data\mtg-jamendo\vocal-tagged-audio`. The older `melody-extraction` pilot is not used. `data/` and `checkpoints/` are ignored, so the mirror, model tools and outputs stay local. `-Stage Vocal` installs only what `-VocalOnly` needs (about 0.9 GB).

The pipeline was written for an Apple Silicon Mac. `extraction/run_local.py` runs its files unchanged and supplies a Windows file lock, the PyTorch build of the same vocal separator, and the environment paths. Each substitution is logged in the pipeline's `history.log`. Stems from this launcher are not expected to match the Mac run sample for sample. The launcher stops before the pipeline starts if a tool is missing or the separator cannot be imported, because the pipeline would otherwise record every selected song as failed.

**Speed.** On this CPU the separator is the bottleneck: tens of seconds of computing per second of music, so a full-length song takes over an hour, while GAME runs at about real time. Add `-Gpu` to run the separator on an Intel Arc GPU: about ten times faster with the same result, so a normal song takes 10 to 15 minutes. It needs `.\Setup-ExtractionTools.ps1 -IntelGpu` once (about 1.4 GB). Keep the laptop awake; standby pauses a run. All 1,604 songs still belong on the team's Mac or a CUDA machine, where the launcher also runs. `-Threads 8` was the fastest thread setting measured here. `-SeparatorOverlap 2` overlaps fewer chunks: 7 instead of 13 for a 30-second track, about a third less time. On the one track compared, its notes differed from the default no more than two GAME runs on the same stem differ from each other; see the [extraction check](docs/Jamendo_Extraction_Verification.md). It is logged and should not be mixed with default-setting stems.

For BPDR, read only `outputs/game_vocal`, and read its `model_midi_pitch`: the published pitch is raised 12 semitones and scores zero against vocadito's annotated pitch. `-KeepStems` saves each separated vocal in `data\jamendo-vocal-stems`, which the pipeline otherwise deletes. Outputs are predictions, not annotations. The dataset is for non-commercial research, and 353 of the local tracks carry NoDerivatives licenses.

Three checks back the plan's gates. Run the first and third with the launcher environment `checkpoints\melody-extraction\.venv-run\Scripts\python.exe`, and the second with this component's own environment:

```powershell
python extraction\check_game.py                      # J2: GAME on vocadito training/development singers against A1 and A2
python extraction\check_stems.py measure --game-run '<J2 run>'   # J3: kept stems against vocadito crops
python extraction\compare_overlap.py 1079195 --overlap 2         # accuracy cost of the faster separator setting
```
