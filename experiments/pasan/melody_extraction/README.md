# Melody extraction experiments

This folder turns an MP3 into the same research artifacts through several
interchangeable pipelines:

- `melody.mid` — note events for listening or symbolic models.
- `notes.parquet` — ML-ready rows: onset, offset, MIDI pitch, confidence, source.
- `notes.csv` — human-readable copy of the note table.
- `f0.npz` — frame times, F0 in Hz, confidence, and voiced mask (when available).
- `metadata.json` — global key estimate, settings, runtime, tool versions, and paths.

The global key is always estimated from the original mix. This is intentional:
a vocal stem alone often does not contain enough harmonic evidence to distinguish,
for example, C major from A minor.

## Quick start

Run these commands from the repository root:

```bash
uv sync

# Confirm the default dataset and installed tools.
uv run python -m experiments.pasan.melody_extraction.main doctor

# Fast first test: no model download and no stem separation.
uv run python -m experiments.pasan.melody_extraction.main run \
  --pipelines pyin_fullmix --limit 1

# Compare the original-mix baseline with separated-vocal pYIN.
# The MLX checkpoint downloads once on the first use.
uv run python -m experiments.pasan.melody_extraction.main compare \
  --pipelines pyin_fullmix mlx_pyin_vocals --limit 1
```

Results default to `outputs/<pipeline>/<relative song path>/`. Separated vocal
stems are cached under `outputs/_stems/`, so multiple vocal pipelines reuse them.
The comparison command also writes `outputs/comparison.csv` and `.json`.

After a batch, build one index that an ML dataloader can read:

```bash
uv run python -m experiments.pasan.melody_extraction.main manifest
```

This writes `outputs/manifest.parquet` and `outputs/manifest.csv`, with one row
per song/pipeline and paths to its note, MIDI, F0, key, and metadata artifacts.

## Changing the dataset path

The current default is:

`experiments/pasan/dataset/mtg-jamendo/vocal-tagged-audio`

Use a CLI flag for one run:

```bash
uv run python -m experiments.pasan.melody_extraction.main run \
  --pipelines pyin_fullmix \
  --input-dir /path/to/new/audio \
  --output-dir /path/to/results
```

Or set `MELODY_INPUT_DIR`, `MELODY_OUTPUT_DIR`, and
`MELODY_MODEL_CACHE_DIR`. All path defaults are centralized in `paths.py`.

Use `--pattern '14/*.mp3'` to select a subset, `--limit N` for a small test,
and `--overwrite` to replace an existing result.

## Available pipelines

```bash
uv run python -m experiments.pasan.melody_extraction.main list
```

The useful first comparison is:

1. `pyin_fullmix` — cheap baseline; unreliable when chords dominate.
2. `mlx_pyin_vocals` — clean continuous vocal F0 after MLX separation.
3. `mlx_basic_pitch_vocals` — vocal note events from Spotify Basic Pitch.
4. `mlx_game_vocals` — stronger singing-specific note transcription.
5. `rmvpe_fullmix` — vocal F0 directly from the polyphonic mix, without separation.
6. `mlx_rmvpe_vocals` — RMVPE after separation, for an A/B test of that choice.
7. `mlx_rosvot_vocals` — an alternative singing-specific note transcriber.
8. `essentia_melodia_fullmix` — predominant melody including instruments.
9. `sheetsage_fullmix` — audio-aligned lead melody from the full mix; chords disabled.
10. `hybrid_vocal_melodia` — vocal F0 when confident, full-mix melody otherwise.

`basic_pitch_fullmix` is also included as a polyphonic baseline. Basic Pitch may
return accompaniment notes on a full mix, so do not automatically treat every
returned note as the lead melody.

SheetSage's code is installed, but its official checkpoint host returned HTTP
403 during the August 2026 smoke test. The pipeline reports `unavailable`
instead of crashing a comparison; it will work when the host is restored or the
official assets are placed under `.models/sheetsage`.

## Optional tool environments

MLX, librosa, Parquet, MIDI, and evaluation libraries are in the root uv
environment. Basic Pitch and Essentia cannot share it cleanly:

- Basic Pitch does not currently support the root's Python 3.12 setup.
- The compatible Apple-Silicon Essentia 3.12 wheel was compiled for NumPy 1.x,
  while MLX Audio Separator requires NumPy 2.x.

The adapters therefore call small worker scripts in isolated uv environments.

### Basic Pitch

```bash
uv venv --python 3.10 .venv-basic-pitch
uv pip install --python .venv-basic-pitch/bin/python basic-pitch
export MELODY_BASIC_PITCH_PYTHON="$PWD/.venv-basic-pitch/bin/python"
```

### Essentia Melodia

Use the current Python 3.14 Apple-Silicon wheel, which is compatible with NumPy
2, in a separate uv environment:

```bash
uv venv --python 3.14 .venv-essentia
uv pip install --python .venv-essentia/bin/python essentia numpy
export MELODY_ESSENTIA_PYTHON="$PWD/.venv-essentia/bin/python"
```

### GAME

GAME is a repository plus a checkpoint rather than a normal library. Keep its
dependencies isolated, follow its official installation instructions, then set:

```bash
export MELODY_GAME_ROOT=/path/to/GAME
export MELODY_GAME_MODEL=/path/to/model.pt
export MELODY_GAME_PYTHON=/path/to/game-venv/bin/python
```

The adapter copies each vocal stem to a temporary directory because GAME writes
MIDI beside its input. It never writes into or changes the source dataset.

### RMVPE

RMVPE packages are not standardized, so the adapter uses the common
`from rmvpe import RMVPE` API in an isolated interpreter. Set:

```bash
export MELODY_RMVPE_PYTHON=/path/to/rmvpe-venv/bin/python
# Optional when that package does not download its own model:
export MELODY_RMVPE_MODEL=/path/to/rmvpe.pt
```

The worker expects 16 kHz audio and writes a 10 ms F0 contour. RMVPE does not
return calibrated confidence, so its voiced frames receive confidence `1.0`.

### ROSVOT

ROSVOT is tested upstream with Python 3.9/CUDA and old dependency versions, so it
must not be merged into the root MLX environment. Follow the official inference
setup, including checkpoints, then set:

```bash
export MELODY_ROSVOT_ROOT=/path/to/ROSVOT
export MELODY_ROSVOT_PYTHON=/path/to/rosvot-venv/bin/python
```

## File responsibilities

- `paths.py` and `config.py`: paths and tunable parameters.
- `mlx_separator.py`: MLX Audio Separator only.
- `librosa_melody.py`: pYIN only.
- `basic_pitch_adapter.py`, `essentia_adapter.py`, `game_adapter.py`: tool wrappers.
- `rmvpe_adapter.py`, `rosvot_adapter.py`: optional vocal research-model wrappers.
- `sheetsage_adapter.py`: SheetSage full-mix melody only (no chord output).
- `fusion.py`: vocal-first/full-mix-fallback contour fusion.
- `pipelines.py`: combinations of the separate tools.
- `artifacts.py`: common MIDI/Parquet/NPZ/JSON output format.
- `manifest.py`: a single ML-ready index over completed songs.
- `main.py`: CLI; `compare.py`: shared multi-tool comparison runner.

## Research caution

These outputs are pseudo-labels, not ground truth. Keep confidence and source
columns, manually inspect a sample from every genre, and evaluate against a
melody-annotated set before training. Full-mix instrumental melody extraction is
especially uncertain when the arrangement contains several simultaneous lead
lines.

Primary tool references: [MLX Audio Separator](https://github.com/ssmall256/mlx-audio-separator),
[Basic Pitch](https://github.com/spotify/basic-pitch),
[Essentia Melodia](https://essentia.upf.edu/tutorial_pitch_melody.html),
[GAME](https://github.com/openvpi/GAME), and
[SheetSage](https://github.com/chrisdonahue/sheetsage).
