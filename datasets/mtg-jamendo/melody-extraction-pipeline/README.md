# Full-song melody extraction

This pipeline processes every MP3 currently in `../vocal-tagged-audio/` (1,604
files here). It never takes excerpts. It saves three separate note
predictions for each song: GAME on a separated vocal stem, SheetSage2's
instrumental melody from the original MP3, and GAME-first combined melody.
GAME pitches are raised **12 semitones** in its published MIDI, CSV, and JSON.
The original GAME pitch remains in `model_midi_pitch` for audit.

## Run and resume

From the repository root, with the existing root, GAME, and SheetSage2 Python
environments and pinned model files set up:

```bash
.venv/bin/python datasets/mtg-jamendo/melody-extraction-pipeline/run.py --limit 10
```

Run that same command again for the next 10 incomplete songs. Omit `--limit 10`
to process all remaining songs. Ctrl-C can stop a run; completed files and
in-progress vocal stems remain available for the next run. Songs with a recorded
failure are skipped on ordinary reruns; use `--retry-failed` to try them again.
Only one extraction process should run at a time.

The environments and checkpoints currently live in `../melody-extraction/` and
are reused without copying their large files. Set `MELODY_MODEL_ASSETS` to a
different directory with the same `.venv-game/`, `.venv-sheetsage2/`, `models/`,
and `tools/GAME/` layout if you move them. The root environment runs the MLX
separator and output conversion. The separator requires an Apple Silicon Mac;
a different separation implementation is needed on other hardware. FFmpeg and
ffprobe must be on `PATH`. Model versions, source commits, and settings are in
[`provenance.json`](provenance.json).

## Saved files

```text
melody-extraction-pipeline/
  run.py                       one resume command
  viewer.html                  fixed listening page
  manifest.csv                 one row per local MP3, metadata, splits, state
  catalog.json                 songs with at least one saved prediction for the page
  history.log                  append-only timestamped extraction events
  outputs/
    game_vocal/               {numeric_track_id}.mid/.json/.csv
    sheetsage2_instrumental/   {numeric_track_id}.mid/.json/.csv
    combined/                 {numeric_track_id}.mid/.json/.csv
  work/                        temporary stems and incomplete inference files
  code/                        shared format, SheetSage2 worker, combination
```

`track_id` is the decimal Jamendo number in the MP3 filename, such as `382`.
`mtg_track_id` is the matching annotation ID, such as `track_0000382`. The
manifest also keeps the relative MP3 and annotation paths, artist/album IDs,
all five official autotagging split labels, tags, durations, status, and note
counts. This provides direct joins to `../annotations/autotagging.tsv` and
`../splits/split-*/autotagging-*.tsv` without renaming the audio. Tracks absent
from the local MP3 subset are not queued.

Every note JSON and CSV uses seconds from the **start of the full song**, MIDI
semitone pitch, rounded MIDI pitch, velocity, and source. JSON also records the
method and schema version. The combined JSON records its beat-based gap rule.
The CSV has one row per note; an empty prediction still gets a valid empty
MIDI, JSON, and header-only CSV. These are predictions, not human annotations.
No chord, SheetSage2 vocal, YourMT3+, audio preview, or score files are saved.

## Listen while extraction runs

In a second terminal, serve the dataset directory:

```bash
python3 -m http.server 8765 --directory datasets/mtg-jamendo
```

Open [the viewer](http://localhost:8765/melody-extraction-pipeline/viewer.html).
It lists songs as soon as a prediction is saved and marks songs still in progress.
Unavailable methods cannot be selected until their files are ready. The page
plays the original full MP3 and synthesizes the
selected MIDI notes as a plain tone with its own play, seek, mute, and volume
controls. Seeking the MIDI bar while playing both also seeks the original song.
The song list refreshes every 10 seconds
without replacing the page. The local server is needed because browsers often
block `file://` pages from reading neighboring JSON files. The page downloads
each source's MIDI, JSON, and CSV.

## Source and model references

The [MTG-Jamendo dataset](https://github.com/MTG/mtg-jamendo-dataset) uses the
track/artist/album IDs and autotagging splits above. [GAME](https://github.com/openvpi/GAME)
exports note MIDI and CSV; [SheetSage2](https://huggingface.co/m-a-p/SheetSage2)
provides a separate instrumental melody MIDI and beat times. This pipeline
keeps only those note results plus the beat period needed for combination.
SheetSage2's model weights have a CC BY-NC 4.0 license; check its model card
before using predictions outside noncommercial research.
