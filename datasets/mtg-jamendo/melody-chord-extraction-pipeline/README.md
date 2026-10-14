# Full-song melody + chord extraction

One command extracts **ten resumable results for each complete local MP3** in
`../vocal-tagged-audio/`, in this order:

1. GAME vocal notes, from a separated vocal stem, shifted up 12 semitones.
2. SheetSage2 instrumental melody from the original MP3.
3. A monophonic combined melody: GAME first; SheetSage2 only in long vocal gaps.
4. SheetSage2 beat times, downbeats, and meter from the original full MP3.
5. A key estimate from the combined **melody only**.
6. LV-Chordia chords decoded with its `submission` vocabulary.
7. The same LV-Chordia model decoded with its `full` vocabulary.
8. An independent submission-chord key estimate and disagreement diagnostic.
9. Key-relative basic chords from the submission labels.
10. Key-relative basic chords from the full labels.

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
specific setup needs a compatible Mac. GAME, SheetSage2, and LV-Chordia now run
in separate processes for each song, releasing their model memory before the
next song.
The runner retains only result counts and metadata in memory. Already saved
melody and chord stages are reused when adding beat, key, and basic chord
results.

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
If the SheetSage2 worker runs out of memory or exits, the runner retries that
song once with a fresh worker. If the retry also fails, it records the error
and stops the batch instead of producing partial results for later songs.
It also stops if GAME cannot access Metal or a GAME/LV-Chordia worker runs out
of memory.
Songs that already have melody and chords but no beat files are selected for
beat backfill; the other validated stages are reused. SheetSage2 must run
once again for those songs because its original temporary `beat.lab` files
were discarded after melody extraction. New songs save beats during the same
SheetSage2 transcription as instrumental melody.

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
      beats/
        beats.json/.csv         SheetSage2 beat grid and meter
      chords/
        submission.json/.csv
        full.json/.csv
      key.json               key estimated only from combined melody
      key_comparison.json    chord diagnostic and review flags
      roman/
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
`beats.json` and `beats.csv` save one row per SheetSage2 beat: time from the
start of the full MP3, zero-based beat and bar indexes, beat position within
the bar, predicted meter, and a downbeat flag. JSON also retains the separate
`downbeat.lab` times and reports whether they match the beat-position flags.
The median beat period and its reciprocal in beats per minute summarize the
event rate; in compound meters such as 6/8 this rate is not necessarily the
musical quarter-note tempo. Beats, downbeats, meter, and bars are model
predictions, not verified annotations. The beat files are linked to the same
track ID, MP3 path, and source fingerprint as the other outputs.
Chord JSON and CSV save full-song segment times and the **exact** LV-Chordia
label, plus parsed root, quality, and bass degree. `N` means no chord.
`submission` and `full` use the same pretrained model but different allowed
label vocabularies; full's rare labels are not automatically more accurate.
`key.json` ranks 24 major/minor keys using duration-weighted melodic pitch
classes and a small phrase-ending tonic tie-break. It records the top three
candidates, score gap, and a qualitative confidence. These confidence labels
measure score separation, not calibrated correctness. This file is produced
even if chord extraction fails. `key_comparison.json` estimates the key from
submission chords as a **diagnostic only**. `key_agreement=disagree` and
low-confidence estimates set review flags in the manifest and comparison
file; the chord estimate never replaces the melody key and
is never used to make Roman labels. Both estimates are predictions, and a
low-confidence key may be wrong. The estimator uses C#/Db-equivalent pitch
classes and writes one canonical spelling for each; it does not infer notation
or detect modulations.

The `roman/` files preserve the original chord label and times, and add the
root's chromatic distance from the **melody-derived tonic**, a Roman root,
and the proposal's basic family: `MAJ`, `MIN`, `DOM7`, `DIM`, `HDIM7`, `DIM7`,
`AUG`, `SUS2`, or `SUS4`. `N` stays separate. For example, in C major,
`G:9` becomes `V:DOM7` and `A:min7` becomes `vi:MIN`; bass inversions and
extensions remain available in the original chord file. These labels encode
relative root and coarse chord structure, not a full functional analysis.
Power chords, rootless labels, and other unsupported qualities remain marked
`unsupported_quality` with no invented basic label. If melodic evidence is
insufficient, Roman labels are marked `key_unavailable` instead of using the
diagnostic chord key. Original chords are never altered.

No synthetic chord MIDI is saved because a chord symbol does not determine a
unique voicing. None of these outputs is a human-verified annotation.

Each stage writes its own files atomically, with JSON as the final commit
marker. Resume checks the output triplet/pair, content, and source MP3 size and
modification time; missing or damaged stages are recomputed. The `history.log`
and manifest record failures without discarding completed stages.
Derived key, comparison, and Roman outputs record hashes of their input JSONs
so a changed upstream result invalidates and rebuilds only its dependents.

## Listen while it runs

In another terminal, serve the dataset directory:

```bash
python3 -m http.server 8765 --directory datasets/mtg-jamendo
```

Open [the simple viewer](http://127.0.0.1:8765/melody-chord-extraction-pipeline/simple-viewer.html).
It keeps a fixed list of the tracks included when the HTML was generated, with
native players for the original song and synthesized combined melody. Opening
one track loads only its beat and chord JSON. The timeline shows predicted
beats, downbeats, and exact `submission` or `full` chord labels; its chord rows
and timeline seek both players. Use **Refresh data** after extracting a missing
beat file while the page is open. Newly extracted track IDs require updating
the HTML song list; existing listed tracks need no HTML update when their beat
or chord files change.
