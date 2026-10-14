# Full-song chord extraction

This folder extracts **chords from the original full MP3s** in
`../vocal-tagged-audio/`. It uses only LV-Chordia 1.1.0, the model used in
Rahul's experiment. It does not use the melody results or split songs into
excerpts. Its labels are model predictions, not verified chord annotations.

LV-Chordia predicts root, chord structure, and bass as a sequence. The default
`submission` vocabulary includes major/minor, suspended, diminished,
augmented, seventh, ninth, eleventh, thirteenth, and common bass inversions.
It keeps labels such as `C:sus4`, `D:min7`, and `F:maj/3` **exactly as the model
returned them**. `/3` is a bass degree relative to the root; it is not the
name of a bass note. `N` means the model predicts no chord. We do not reduce
the labels to major/minor, fill gaps, quantize to beats, or derive chords from
the extracted melody.

The optional `full` vocabulary can decode many rarer labels, but rare qualities
are difficult to recognize reliably. It writes to a separate output directory
so its predictions cannot overwrite `submission` results. Start with
`submission` for the complete dataset and compare `full` on selected songs
before deciding whether to use it for training.

## Setup

Use the repository's Python 3.12 `.venv`, which already has PyTorch and
Librosa. Install the pinned chord package in that environment:

```bash
uv pip install --python .venv/bin/python -r datasets/mtg-jamendo/chord-extraction-pipeline/requirements.txt
```

FFmpeg and `ffprobe` must also be on `PATH`. LV-Chordia's five pretrained
checkpoints are bundled with the normal package installation. The runner
checks that all five are present before processing a song, since inference
with missing weights can silently produce worthless `N` predictions.

## Run and resume

From the repository root:

```bash
.venv/bin/python datasets/mtg-jamendo/chord-extraction-pipeline/run.py --limit 10
```

Run the same command again for the **next 10 incomplete songs**. Omit
`--limit` to process all remaining songs. Ctrl-C stops after the current
interruption, keeping every completed JSON and CSV. Failed songs are recorded
and skipped on normal reruns; add `--retry-failed` to try them again. Use
`--track-id 382` to process one local song. Only run one extraction process
at a time.

For experimental high-detail decoding:

```bash
.venv/bin/python datasets/mtg-jamendo/chord-extraction-pipeline/run.py --track-id 382 --vocabulary full
```

The default and full results have separate manifests and catalogs. The
`--limit` count includes only songs actually selected for inference. The
runner checks both result files and their source MP3 fingerprint on resume;
partial, corrupt, or mismatched results are recomputed.

## Files

```text
chord-extraction-pipeline/
  run.py                   extraction and resume command
  code/chords.py           dataset joins, exact-label parsing, validation, output
  requirements.txt         pinned model dependency
  viewer.html              original audio plus synchronized chord timeline
  manifest.csv             every local MP3: annotations, five splits, status
  catalog.json             completed songs for the viewer
  history.log              append-only starts, completions, failures
  outputs/submission/      one {numeric_track_id}.json and .csv per song
  outputs/full/            optional high-detail results, same format
  manifest-full.csv        optional full-vocabulary status
  catalog-full.json        optional full-vocabulary viewer list
```

The generated manifests, catalogs, logs, and output folders are ignored by
Git. Keep or back them up when moving computers. `track_id` is the numeric MP3
filename (for example `382`), while `mtg_track_id` is the official annotation
ID (`track_0000382`). The manifest also includes annotation path, artist and
album IDs, duration, tags, and the labels from all five official autotagging
splits. This lets researchers join predictions back to MTG-Jamendo without
guessing from filenames.

Each per-song JSON has model/version/vocabulary provenance, original audio
path and fingerprint, full-song duration, a list of quality flags, and `events`.
Each event records `start_sec`, `end_sec`, `chord_label`, `root`, `quality`, and
`bass_degree`. The CSV has the same event fields plus `track_id`. Times are
seconds from the **start of the full song**. `chord_label` is the authoritative
lossless value. Parsed fields are convenient for filtering, and unknown
future label syntax is kept intact in `chord_label` and flagged for review.
`all_no_chord` is also flagged rather than silently accepted as a healthy
prediction. The model does not provide calibrated confidence per segment.

No MIDI is saved: a chord symbol identifies harmony, but does not specify a
unique voicing or instrument performance. A synthetic MIDI rendering would
add invented notes that the model did not extract.

## Listen and inspect

In another terminal, from the repository root:

```bash
python3 -m http.server 8765 --directory datasets/mtg-jamendo
```

Open [the chord viewer](http://127.0.0.1:8765/chord-extraction-pipeline/viewer.html).
It plays the original song, highlights the predicted chord at the current
time, and lets you click a segment to seek. It refreshes the completed-song
list every 10 seconds and offers JSON/CSV downloads. Use the vocabulary
selector to inspect any separately extracted `full` results.

## Why this model

The [original ISMIR 2019 paper](https://archives.ismir.net/ismir2019/paper/000078.pdf)
models chord root/triad, bass, and extensions to handle larger vocabularies.
The [LV-Chordia package](https://github.com/openmirlab/lv-chordia) provides the
pretrained ensemble and a modern API. By comparison,
[Essentia's ChordsDetection](https://essentia.upf.edu/reference/std_ChordsDetection.html)
outputs only major and minor triads. This pipeline uses LV-Chordia's normal
package install and original MP3 input, then adds full-song validation,
dataset linkage, durable per-song files, and reliable resume behavior.
