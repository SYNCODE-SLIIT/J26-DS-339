# Full-song melody + chord extraction

One command extracts **ten core results plus BPDR for each eligible local MP3** in
`../vocal-tagged-audio/`, in this order:

1. GAME vocal notes, from a separated vocal stem, shifted up 12 semitones.
2. A BPDR vocal crop and feature export for license-eligible voice-tagged songs.
3. SheetSage2 instrumental melody from the original MP3.
4. A monophonic combined melody: GAME first; SheetSage2 only in long vocal gaps.
5. SheetSage2 beat times, downbeats, and meter from the original full MP3.
6. A key estimate from the combined **melody only**.
7. LV-Chordia chords decoded with its `submission` vocabulary.
8. The same LV-Chordia model decoded with its `full` vocabulary.
9. An independent submission-chord key estimate and disagreement diagnostic.
10. Key-relative basic chords from the submission labels.
11. Key-relative basic chords from the full labels.

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

Run the **same command again** to process the next 10 incomplete songs. Completed
older songs missing BPDR are backfilled first, then new songs run all stages
one song at a time. Omit
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
  simple-viewer.html        original audio, MIDI melody, chords, BPDR crop
  manifest.csv              every local track, annotation IDs, splits, status
  catalog.json              available results and per-track BPDR links
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
      bpdr/                  only on license-eligible voice-tagged tracks
        record.json          crop mapping, screening, hashes and provenance
        vocal.wav            listenable vocal crop, when exported
        features.npz         71-column boundary features, when exported
        boundaries.json      crop GAME notes, when exported
        masks.npz            screen masks, when exported
        reviews.csv          pending review slots, when applicable
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
It keeps its original song list and accepts any newer extracted track through
the **Track ID** box; that lookup loads only the selected song's combined
melody JSON. Native players compare the original song and synthesized melody.
Opening one track loads only its beat, chord, and BPDR record JSON. The timeline shows predicted
beats, downbeats, and exact `submission` or `full` chord labels; its chord rows
and timeline seek both players. Use **Refresh data** after extracting a missing
beat file while the page is open. Newly extracted tracks can be opened by ID
without regenerating the HTML song list.

## BPDR vocal-input export

BPDR runs as an ordinary resumable stage for each local voice-tagged track with a
recognized Creative Commons BY/BY-SA/BY-NC/BY-NC-SA license. There is no
artist selection, five-track cap, run ID, or separate output root. Vocadito is
not used. Existing songs with the ten core stages complete are visited first
to add BPDR; each new song then runs GAME, BPDR, SheetSage2, beats, chords, key,
and Roman numerals before moving to the next song. Every stage uses the same
`history.log`, `manifest.csv`, `catalog.json`, runner lock, and retry behavior.
A failed BPDR attempt stops the batch after logging the error; run again with
`--retry-failed` once the cause is fixed. Ctrl-C preserves completed results.

Set up the isolated feature environment once:

```bash
uv venv --python 3.12 datasets/mtg-jamendo/melody-chord-extraction-pipeline/.venv-bpdr
uv pip install --python datasets/mtg-jamendo/melody-chord-extraction-pipeline/.venv-bpdr/bin/python \
  -r datasets/mtg-jamendo/melody-chord-extraction-pipeline/bpdr-requirements.txt
```

Use the normal run command from above; no BPDR flags are needed. BPDR reuses
the vocal stem made by the GAME stage for a new song. A song extracted before
BPDR existed needs its stem separated once more. The full temporary stem is
deleted only after the BPDR record is committed. GAME's published +12-semitone
melody stays unchanged; BPDR uses unshifted model pitch to choose and decode a
crop. GAME, separator, PESTO, and chord work run in separate processes, so the
runner does not retain their model tensors across songs. System memory limits
can still stop a worker; the runner saves prior stages and stops on a BPDR or
known model-memory failure.

Each eligible song gets `outputs/<mtg_track_id>/bpdr/record.json` with source,
model, crop, screen, and artifact hashes. An exported crop also gets
`vocal.wav` (mono 22,050 Hz PCM24), `features.npz`, `boundaries.json`, and
`masks.npz`. A withheld crop may have only its record. Candidate crops get a
per-track `reviews.csv`; a human's saved decisions are never replaced by
resume. The viewer loads the record only when its track card is opened and
shows a player for the vocal crop when a WAV exists. This is a short excerpt,
starting at the original-song offset written in `record.json`; it is not a
full-song melody or ground-truth annotation.

The crop policy chooses complete GAME-note spans bounded by predicted rests
when possible, at most 20 seconds, with at least five predicted singing
seconds. A continuous vocal passage can instead use note boundaries and is
flagged for review. The record maps original stem sample indexes and seconds
to the exported WAV. Technical checks cover finite audio, supported channels,
duration, silence, and clipping. They do not establish one singer, musical
acceptability, or correct pitch. No denoising, time stretch, intonation
correction, or section concatenation occurs.

`features.npz` uses the frozen BPDR `analysis.py` implementation: columns
0–63 are log-mel dB; 64 pitch cents relative to A4; 65 PESTO confidence; 66
voiced flag; 67 adjacent voiced pitch change in cents; 68 log RMS dB; 69
onset score; and 70 a crop-GAME note-edge score. The 70-column frame-only
control is a slice of this matrix. Unvoiced `f0_hz` is NaN in the archive;
the masked model input uses zero pitch only in column 64. The audio-derived
heuristic boundary is separately named in `masks.npz` and is never silently
substituted for GAME's column 70. PESTO uses frozen `mir-1k_g7`, 20 ms frames,
1024-sample centered windows, 64 mel bands, and 0.8 voicing confidence.

All BPDR exports remain `exploratory_not_research` and
`training_eligible=false`: they are model-derived inputs, not approved
training pairs. Human review and an independent validation step would be
needed before training. The five original pilot crops were migrated to their
track folders, hash-checked, and the separate pilot folder removed.

### Windows adapter

The BPDR route resolves `Scripts/python.exe`, uses a Windows file lock, and
runs the same checkpoint through a separate PyTorch `audio-separator` worker
with CUDA when available or CPU otherwise. The worker is isolated from the
main and PESTO environments. Set up that worker separately on Windows:

```powershell
uv venv --python 3.12 datasets/mtg-jamendo/melody-chord-extraction-pipeline/.venv-bpdr
uv pip install --python datasets/mtg-jamendo/melody-chord-extraction-pipeline/.venv-bpdr/Scripts/python.exe -r datasets/mtg-jamendo/melody-chord-extraction-pipeline/bpdr-requirements.txt
uv venv --python 3.12 datasets/mtg-jamendo/melody-extraction/.venv-separator-windows
uv pip install --python datasets/mtg-jamendo/melody-extraction/.venv-separator-windows/Scripts/python.exe -r datasets/mtg-jamendo/melody-chord-extraction-pipeline/bpdr-windows-separator-requirements.txt
```

The existing GAME, SheetSage2, LV-Chordia and shared asset setup is still
needed for a combined run. PyTorch versus MLX can differ numerically; records
include model and backend provenance. The Windows adapter has not been executed or compared against Mac
outputs in this checkout. See the [audio-separator project](https://github.com/nomadkaraoke/python-audio-separator)
for Windows backend support and the [PESTO project](https://github.com/SonyCSLParis/pesto)
for the frozen pitch model.
