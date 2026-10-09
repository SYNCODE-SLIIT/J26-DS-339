# Repeatable 20-song melody comparison

From the repository root, run:

```bash
.venv/bin/python datasets/mtg-jamendo/melody-extraction/code/next_comparison.py
```

Each run randomly selects 20 tracks from 20 different artists, makes 30-second
excerpts, runs vocal separation, GAME on the vocal stem and mix, SheetSage2,
YourMT3+, and the combined GAME vocal + SheetSage2 instrumental line. It updates
this same [`compare.html`](compare.html) after all predictions pass validation.
An already open page checks for a completed new run every 10 seconds and reloads
automatically. On the first use of this feature, refresh the old page once.

The script builds the next set in a temporary sibling directory, so the current
page remains available during inference. It remembers every selected track ID in
`seen_tracks.csv` and excludes it from later runs, along with the original pilot
and earlier round-two set. The current `sample.csv` records each excerpt's start
in its full MP3. The script prints the seed; `--seed NUMBER` reproduces the same
random choice when the exclusion history is the same. `--count` and `--seconds`
change the number and length of excerpts.

The combined MIDI uses GAME notes from the separated vocal stem and fills only
vocal-free gaps lasting at least two SheetSage2 beats and 0.8 seconds with the
SheetSage2 instrumental melody. Its JSON records each note's source. The MIDI
has one note at a time. Empty instrumental predictions stay empty.

The audio previews use a plain sine tone. Predictions are unreviewed; listen
against the original before treating any MIDI as a training label. `summary.csv`
lists the results and `review.csv` is the blank review sheet for the current set.
If an inference stage fails, its log remains in the temporary directory and the
current comparison is preserved. Rerunning the command starts a fresh set.
