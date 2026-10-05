# Chord refinement: reference-chord extraction

This isolated component currently performs one task only:

```text
original MTG-Jamendo mixed audio
    -> LV-Chordia large-vocabulary recognition
    -> time-aligned reference chord events
```

It does not extract melody, separate vocals, generate basic chords, refine
chords, process genre, calculate Roman numerals, or align events to beats.

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
.\.venv\Scripts\python.exe -m chord_refinement.src.extract_reference_chords
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

