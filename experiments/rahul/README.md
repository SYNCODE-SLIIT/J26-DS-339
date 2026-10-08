# Chord refinement: feature and target construction

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

