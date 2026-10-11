# Full-song melody + chord extraction

One command extracts **five separate results for each complete local MP3** in
`../vocal-tagged-audio/`, in this order:

1. GAME vocal notes, from a separated vocal stem, shifted up 12 semitones.
2. SheetSage2 instrumental melody from the original MP3.
3. A monophonic combined melody: GAME first; SheetSage2 only in long vocal gaps.
4. LV-Chordia chords decoded with its `submission` vocabulary.
5. The same LV-Chordia model decoded with its `full` vocabulary.

This folder contains its own extraction code and results. It **reuses the
existing pinned model files and GAME/SheetSage2 environments** in
`../melody-extraction/`; it does not copy their large checkpoints or write
results into either earlier pipeline. See [`provenance.json`](provenance.json)
for model versions, source revisions, and the melody combination rule.

## Setup

From the repository root, use the Python 3.12 `.venv` already used by the
melody pipeline. On a fresh checkout, install the root project's locked
dependencies first, then the pinned chord package:

```bash
uv sync --locked
uv pip install --python .venv/bin/python -r datasets/mtg-jamendo/melody-chord-extraction-pipeline/requirements.txt
```

The existing GAME and SheetSage2 environments/checkpoints must also be set up
as described in `../melody-extraction-pipeline/README.md`. By default the
runner looks for them in `../melody-extraction/`; set `MELODY_MODEL_ASSETS` to
another directory with the same layout if necessary. `ffmpeg` and `ffprobe`
must be on `PATH`. The current vocal separator uses Apple Silicon/MLX, so this
specific setup needs a compatible Mac. The SheetSage2 worker remains loaded
across songs in one run, but the runner still finishes and records all five
stages for one song before starting the next.

## Run, stop, resume

```bash
.venv/bin/python datasets/mtg-jamendo/melody-chord-extraction-pipeline/run.py --limit 10
```

Run the **same command again** to process the next 10 incomplete songs. Omit
`--limit` to process all remaining songs, or use `--track-id 382` to process
one song. Ctrl-C stops the run; completed stages remain saved and the next run
resumes the current song. Failed songs are logged and skipped on the usual
next-song run; `--retry-failed` retries them. An explicit `--track-id` also
retries that track. Only one runner can hold the pipeline lock at a time.

## Saved data

```text
melody-chord-extraction-pipeline/
  run.py                    song-by-song runner and resume
  code/                     shared format, GAME, SheetSage2, combination, chords
  provenance.json           pinned model and method information
  requirements.txt          pinned chord package
  viewer.html               original audio, MIDI melody player, chord timeline
  manifest.csv              every local track, annotation IDs, splits, status
  catalog.json              available results for the viewer
  history.log               append-only stage history and errors
  outputs/
    track_0000382/
      melody/
        game_vocal.mid/.json/.csv
        sheetsage2_instrumental.mid/.json/.csv
        combined.mid/.json/.csv
      chords/
        submission.json/.csv
        full.json/.csv
  work/                     temporary stems and logs
```

`track_id` in every result is the decimal MP3 number (e.g. `382`), and
`mtg_track_id` is the official ID and output directory (e.g.
`track_0000382`). `source_mp3` links each result to the original file.
`manifest.csv` includes artist/album IDs, annotation path, tags, and all five
official autotagging split memberships. Only locally present MP3s are queued.

Melody JSON and CSV save full-song note onsets/offsets in seconds, MIDI pitch,
velocity, and source; the MIDI is a listening/DAW copy. GAME's original model
pitch is retained as `model_midi_pitch` in the JSON/CSV alongside the shifted
pitch. Combined melody metadata includes the beat-derived minimum vocal gap.
Chord JSON and CSV save full-song segment times and the **exact** LV-Chordia
label, plus parsed root, quality, and bass degree. `N` means no chord.
`submission` and `full` use the same pretrained model but different allowed
label vocabularies; full's rare labels are not automatically more accurate.
No synthetic chord MIDI is saved because a chord symbol does not determine a
unique voicing. None of these outputs is a human-verified annotation.

Each stage writes its own files atomically, with JSON as the final commit
marker. Resume checks the output triplet/pair, content, and source MP3 size and
modification time; missing or damaged stages are recomputed. The `history.log`
and manifest record failures without discarding completed stages.

## Listen while it runs

In another terminal, serve the dataset directory:

```bash
python3 -m http.server 8765 --directory datasets/mtg-jamendo
```

Open [the viewer](http://127.0.0.1:8765/melody-chord-extraction-pipeline/viewer.html).
It lists a song as soon as one stage is saved and refreshes every 10 seconds.
Select GAME, SheetSage2, or combined MIDI melody; listen to the original MP3
and the synthesized melody with separate controls; and follow the timed
`submission` or `full` chord labels. Click a chord row to seek. The page also
links to each available MIDI/JSON/CSV file.
