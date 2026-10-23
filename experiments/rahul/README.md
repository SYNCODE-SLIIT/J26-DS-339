# Chord refinement: feature and target construction

The from-scratch PyTorch architecture is implemented in `src/model/`.
See [TRANSFORMER_ARCHITECTURE.md](TRANSFORMER_ARCHITECTURE.md) and the executed
[05_transformer_architecture.ipynb](notebooks/05_transformer_architecture.ipynb).
Full training is implemented in
[06_train_transformer.ipynb](notebooks/06_train_transformer.ipynb), using the
explicitly selected 17 input tensors, validation-based early stopping and a saved
best checkpoint. Notebook 05 remains the architecture-only walkthrough.

## Current four-bar variable-length dataset

The active build is `outputs/model_preprocessing/final_all_tracks_quality_v6_multibar/`.
Load `train_sequences.json`, `validation_sequences.json` or `test_sequences.json`
with `SavedSequenceDataset`; the indices reference numerical NPZ shards.
Each beat remains an individual prediction token, grouped into four source bars
by default. DataLoader padding is batch-local and all melody notes are retained.

```powershell
.\.venv\Scripts\python.exe -m experiments.rahul.src.rebuild_multi_bar_dataset
```

The command reuses the verified v5 data/vocabularies and writes a new immutable
snapshot, including a verified readable CSV. `--bars-per-sequence` changes grouping.
See [FINAL_DATASET.md](FINAL_DATASET.md) for actual counts and regeneration details,
[MODEL_PREPROCESSING.md](MODEL_PREPROCESSING.md) for schemas, and
[04_multibar_dataset_inspection.ipynb](notebooks/04_multibar_dataset_inspection.ipynb)
for executable inspection. Earlier notebooks and builds below remain historical.

## Historical one-bar and pilot workflows


## Final multi-track dataset

The completed multi-track build is in `outputs/model_preprocessing/final_all_tracks_quality_v5_bars_full_melody/`.
Use its `train_sequences.npz`, `validation_sequences.npz`, and `test_sequences.npz`
with the feature and target vocabularies from that same folder. See
[FINAL_DATASET.md](FINAL_DATASET.md) for actual counts, loading instructions,
source exclusions, compatibility fixes and reproducibility details.

The batch runner discovers all extraction track folders and reuses the existing
Phase 1 builder, encoders and PyTorch Dataset. Build a new immutable snapshot with:

```powershell
.\.venv\Scripts\python.exe -m experiments.rahul.src.build_final_dataset --run-name another_snapshot
```

Incomplete tracks are recorded, not silently substituted. The old Track 382
development exports remain intact. A finished run has `FINAL_DATASET_READY.json`;
`--resume` continues an interrupted, unfinished run and never replaces a completed one.

## Phase 2: model-ready numerical preprocessing

For a step-by-step all-track walkthrough using
`outputs/model_preprocessing/final_all_tracks_quality_v5_bars_full_melody/all_tracks_sequences_readable.csv`,
with source-to-encoded examples, masks, tensor shapes and plots, open
[02_dataset_preprocessing.ipynb](notebooks/02_dataset_preprocessing.ipynb).
Select the project `.venv` kernel and use Restart Kernel and Run All. Additional
notebook dependencies are listed in `notebooks/requirements.txt`. The notebook
scans CSV metadata in chunks and retains full tensors only for selected sequences
from each split. It reads existing data without rebuilding or fitting vocabularies.

```powershell
.\.venv\Scripts\python.exe -m experiments.rahul.src.preprocess_for_transformer --track-id 382 --development-only
.\.venv\Scripts\python.exe -m unittest experiments.rahul.tests.test_model_preprocessing -v
```

This writes separate inputs, targets, masks, train-only vocabularies, chronological
sequence tensors and reports under `outputs/model_preprocessing/`. It leaves
Phase 1 unchanged and does not implement or train a Transformer. See
[MODEL_PREPROCESSING.md](MODEL_PREPROCESSING.md) for the complete tensor contract,
safe silence handling, saved-data loading and multi-track split commands.

## Existing-extraction synthetic baseline

To build the requested baseline from `combined.csv`, detailed chords, Roman
events, and source beats (without re-extracting audio), run from repository root:

```powershell
python -m experiments.rahul.src.build_synthetic_dataset --track-id 382
python -m unittest experiments.rahul.tests.test_build_synthetic_dataset -v
```

Repeat `--track-id` for multiple tracks. New outputs live exclusively under
`outputs/synthetic_dataset/track_0000382/`: intermediate chord events, training
events, validation report, full column schema and build summary. This is a
**synthetic chord-refinement baseline**: basic chords are simplified reference
pseudo-labels, so inputs are not independent of targets. It is distinct from
the older independent harmony builder described below. Detailed target fields
are excluded from the schema's explicit input list. See the generated summary
for policies, actual counts, timing limitations and future sequence guidance.
Optional `--member2-csv` replaces basic inputs with independently supplied
`track_id,beat_index,basic_chord` predictions. It requires one basic triad per
source beat. No production extraction or aligned dataset is overwritten.

This isolated component currently performs one task only:

```text
original MTG-Jamendo mixed audio
    -> LV-Chordia large-vocabulary recognition
    -> time-aligned reference chord events
```

The component also has a leakage-free, one-track feature builder:

```text
original mix -> existing pYIN notes + beat grid + global key
official MTG metadata ---------------------------> genre tags
notes + two-beat windows + key -> independent basic diatonic triads
basic chords -> one-beat model-event grid by greatest overlap
existing reference chords -> overlap alignment after all input generation
                         -> canonicalized time-aligned training events
```

It does not train or implement the refinement Transformer.

## Configuration

Defaults are in `config.py`. `AUDIO_DIR` points to the existing repository
dataset and never copies or modifies it. The settings can also be overridden:

```powershell
$env:CHORD_AUDIO_DIR = "path\to\audio"
$env:CHORD_OUTPUT_DIR = "path\to\outputs"
$env:CHORD_VOCABULARY = "submission"
$env:CHORD_NUMBER_OF_FILES = "1"
```

The supported LV-Chordia dictionaries in the installed package are
`submission`, `ismir2017`, `full`, and `extended`. The default is
`submission`, the recognizer's recommended general-purpose vocabulary.

## Run the one-song test

From the repository root, using the existing project environment:

```powershell
.\.venv\Scripts\python.exe -m experiments.rahul.src.extract_reference_chords
```

The default limit is one file. Increase it later through configuration only
after the first result has been reviewed.

## Outputs

- `outputs/chords/<track_id>.json`: events for one track.
- `outputs/chords/reference_chord_events.csv`: combined event table.
- `outputs/logs/extraction.log`: progress and validation details.

Each event contains `track_id`, `start_time`, `end_time`, `raw_chord`, `root`,
`quality`, `bass_degree`, and `reference_chord`. A batch also creates
`outputs/chords/vocabulary_report.json`.

`raw_chord` is LV-Chordia's unmodified output. `reference_chord` is a
conservative display normalization such as `A:min7` -> `Am7`. Unknown chord
formats remain unchanged rather than being guessed or corrupted.

`basic_chord` is intentionally absent. It will later come from another team
component and is not the same thing as a chord detected from the recording.

## Build one-track training events

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m experiments.rahul.src.build_training_events `
  --track-id 1400601 --basic-beats-per-event 2 --model-beats-per-event 1
```

The implementation imports the team's existing `pyin_fullmix` extraction and
F0-to-note segmentation functions without modifying them. pYIN runs directly
on the original mix; vocal separation is not used, so all timestamps remain
seconds from time zero of the source MP3. Note filtering uses the existing
60 ms minimum duration and pYIN voiced mask. Confidence is the mean pYIN voiced
probability in the segmented note and is not a calibrated probability.

The beat grid uses `librosa.beat.beat_track`. Basic harmony generation remains
at two detected beats per event. A separate production model grid uses one beat
per event; each model event receives the already-generated basic chord with the
greatest time overlap. The leading intro and trailing remainder are retained.
Since downbeats are not estimated, bar and beat-position fields are null.

For each window, the basic harmonizer scores the seven diatonic triads using
duration-weighted melody/chord-tone compatibility, a small melody-on-root bonus,
and a small common-tone transition score from the previous generated chord.
Minor mode uses common-practice major V and diminished vii°. Candidate scores
are retained. Previous/next basic context is recomputed on the one-beat model
sequence. Reference chords are never read until every basic chord and all other
input-side features have been generated.

Reference targets are selected by greatest interval overlap. All positive-
overlap targets and their ratios are retained for ambiguity analysis. Melody
notes crossing a window boundary appear in every overlapping event: original
timing is preserved and overlap duration is stored. A relative start may thus
be negative.

Outputs for track `<id>` are:

- `outputs/melody/<id>_melody.json`
- `outputs/beats/<id>_beats.json`
- `outputs/metadata/<id>_metadata.json`
- `outputs/basic_chords/<id>_basic_chords.json`
- `outputs/aligned/<id>_aligned.json`
- `outputs/aligned/training_events.csv`

JSON is canonical because melody and reference candidates are naturally nested.
The CSV is a convenience table whose list-valued columns contain JSON arrays.
The command accepts another `--track-id`, so the same isolated stages can later
be wrapped for controlled 5-song and 50-song batches.

The production 10-track pilot is rebuilt with:

```powershell
.\.venv\Scripts\python.exe -m experiments.rahul.src.build_pilot_dataset
```

It validates the complete candidate before atomically replacing
`training_events.csv`, updates the ten aligned JSON files, and writes
`outputs/aligned/production_migration_report.json`.

## Add canonical chord columns

Canonical fields are now added automatically during event construction. The
standalone command remains available for older output files:

```powershell
.\.venv\Scripts\python.exe -m experiments.rahul.src.canonicalize_chords
```

This adds `basic_chord_canonical`, `reference_chord_canonical`,
`basic_root_pitch_class`, and `reference_root_pitch_class` to the combined CSV
and per-track aligned JSON files. Only the root spelling is normalized; quality
and inversion/bass-degree suffix text is preserved exactly. `N` stays `N` and
has a null root pitch class.


The active v6 dataset groups up to four consecutive source bars per variable-length
sequence and retains full melody arrays without four-note truncation. Beats are
padded only within a batch. Notebook `10_complete_preprocessing_walkthrough.ipynb`
explains preprocessing; notebook `11_comprehensive_model_evaluation.ipynb`
compares the three saved model experiments.

## Committed trained model

The selected multi-task inference checkpoint is included in Git at:

`outputs/multitask_transformer/multitask_20261020_164527_051309/best_multitask.pt`

Its `chord_quality_mapping.json` is included alongside it. The matching
`feature_vocabularies.json` and `target_vocabulary.json` are included under
`outputs/model_preprocessing/final_all_tracks_quality_v6_multibar/`.
Keep these mappings with the checkpoint; category IDs from other snapshots are
not interchangeable.

From the repository root:

```python
from pathlib import Path
from experiments.rahul.src.model.multitask_transformer import load_multitask_checkpoint

base = Path("experiments/rahul/outputs")
model = load_multitask_checkpoint(
    base / "multitask_transformer/multitask_20261020_164527_051309/best_multitask.pt",
    base / "model_preprocessing/final_all_tracks_quality_v6_multibar",
    device="cpu",
)
```

The matching model/loader implementation and dependencies in
`requirements-transformer.txt` are committed too. Install notebook tools from
`notebooks/requirements.txt` when running notebooks.
The dataset shards, extraction files, logs and review archives remain local;
copy or rebuild them before running data-dependent notebooks. The baseline and
weighted checkpoints are not part of this selected model upload, so notebooks
that compare all three models also require those separate checkpoints.
