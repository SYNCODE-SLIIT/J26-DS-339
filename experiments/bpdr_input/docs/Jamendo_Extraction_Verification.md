# MTG-Jamendo source and melody-extraction check — 11 October 2026

**Confirmed on this machine:** the second data source is mirrored inside the component; the team's pipeline code runs unedited through a launcher; the complete pipeline produced all three outputs for a two-song sample; and the team's listening page shows them. **Also found:** the vocal separator is far too slow on this CPU for full-length songs, and the GAME note model reads performed pitch correctly only through its unshifted field. The two sample tracks were chosen for being short, so they say nothing reliable about the stems of typical songs. Nothing here is a BPDR result or a training-ready collection.

## What was added

```text
experiments/bpdr_input/
  Setup-Jamendo.ps1            builds the mirror below from the repository's datasets/mtg-jamendo
  Setup-ExtractionTools.ps1    installs the pinned model tools into ignored checkpoints/, with hash checks
  Run-Extraction.ps1           runs the pipeline; -IndexOnly needs no model
  extraction/run_local.py      launcher: platform substitutions, tool check, stems, sample selection, speed options
  extraction/check_game.py     gate J2: GAME against vocadito's two annotators
  extraction/check_stems.py    gate J3: kept stems against vocadito crops
  extraction/compare_overlap.py  accuracy cost of the faster separator setting
  data/mtg-jamendo/            ignored, local only
    annotations/  splits/  LICENSE  README.md  audio_licenses.txt
    vocal-tagged-audio/NN/*.low.mp3
    melody-extraction-pipeline/
      code/  .gitignore  README.md  provenance.json  run.py  viewer.html     copied, unedited
      manifest.csv  catalog.json  history.log  outputs/  work/              written by real pipeline runs
  data/jamendo-vocal-stems/    separated vocals kept for BPDR, which the pipeline otherwise deletes
```

The mirror follows the team's layout so that the pipeline finds its dataset as a sibling folder. The older `melody-extraction` pilot is deliberately not mirrored. All 91 copied metadata and pipeline files match the repository copies by SHA-256. The 1,604 MP3s (4.21 GB) moved from `data/vocal-tagged-audio` to `data/mtg-jamendo/vocal-tagged-audio`. Everything under `data/` and `checkpoints/` stays out of Git, so the launcher and scripts are tracked but the mirror, tools and outputs are not.

## Observed dataset facts

Read from the local files and `annotations/autotagging.tsv`; nothing in this table comes from listening.

| Property | Observed value |
|---|---|
| Local recordings | 1,604 `.low.mp3` files, every one present in `autotagging.tsv` |
| Content | Full released tracks with accompaniment; mono, 44.1 kHz, about 90 kbit/s MP3 |
| Duration | 101.2 hours; minimum 30 s, median 216 s, 90th percentile 336 s, maximum 1,703 s |
| Voice evidence | All 1,604 carry the track-level `instrument---voice` tag. It is not a vocal-activity, note or F0 annotation. |
| Independent groups | 257 artists, 402 albums; no singer identifier |
| Balance | One artist has 174 tracks; the median is 3; 85 artists have one |
| Styles | Rock 526, indie 332, pop 315, alternative 287, electronic 275, hip hop 216, folk 180 (tags overlap) |
| Human references | None: no F0, no notes, no second annotator |
| Official splits | Five train/validation/test splits; no artist appears in more than one part of any split |
| Licenses | BY-NC-SA 687, BY-SA 450, BY-NC-ND 316, BY 103, BY-ND 37, BY-NC 10, one unparsed. 353 tracks forbid derivatives; 1,251 permit them. |

Applying the component's existing stable-hash grouping (`singer_splits`, seed 339) to `artist_id` gives 129 training, 51 development, 51 PP1-evaluation and 26 final-holdout artists. With at most three tracks per artist that is 290/118/117/45 tracks. Restricted to licenses that permit derivatives: 98/40/40/20 artists and 222/94/87/37 tracks. An artist is a proxy for a singer, not a verified singer identity. The voice tag covers more than solo singing: the first artist in the inventory is, by its name, a children's choir.

## What the pipeline produces

For each full song the team's `run.py` decodes the MP3 with FFmpeg, separates a vocal stem with a BS-RoFormer checkpoint, transcribes that stem with GAME-1.0-medium, transcribes an instrumental melody from the original MP3 with SheetSage2, and writes a GAME-first combined melody. Each method saves MIDI, JSON and CSV with onset and offset in seconds from the start of the full song.

Points that matter for BPDR:

- `game_vocal` is the only output that describes the singing. `sheetsage2_instrumental` and `combined` contain instrument melody.
- Published GAME pitch is raised 12 semitones; the page labels it "one octave higher". The model's own value stays in `model_midi_pitch`, and that is the one matching performed pitch (gate J2 below).
- GAME pitch is fractional, so a note carries a pitch center in cents, not only a semitone.
- The vocal stem is deleted after note extraction. BPDR needs that audio, so the launcher's `--keep-stems` links it into `data/jamendo-vocal-stems/` first.
- All outputs are model predictions. The pipeline README says so; they are pseudo-labels, not annotations.

## Running it on this machine

The pipeline is written for an Apple Silicon Mac: it imports `fcntl`, separates with `mlx-audio-separator`, starts its model environments through `bin/python`, and calls `ffmpeg` and `ffprobe`. Its README states that other hardware needs a different separation implementation. This machine is Windows 11 with an Intel Core Ultra 5 125H, 15.4 GB of memory and no CUDA GPU.

`extraction/run_local.py` leaves the pipeline files untouched and supplies:

| Need | Substitution on Windows | Effect on outputs |
|---|---|---|
| Single-run lock | `msvcrt.locking` in place of `fcntl.flock` | None |
| Vocal separator | `audio-separator` 0.47.0 (PyTorch, CPU) in place of the MLX package, same checkpoint file | Stems are not expected to be sample-identical to the Mac run |
| `bin/python` | A `bin` junction to `Scripts` inside each model environment | None |
| Text encoding | Python UTF-8 mode | Same encoding as the Mac |
| FFmpeg | Component-local Windows build 9.0.2 | Decoder version differs |

Every substitution is written to the pipeline's `history.log` at the start of a run. The launcher refuses to start, with exit code 2, when a tool is missing or the separator cannot be imported. That matters because the pipeline records such a problem as a failed song and then skips the song on later runs.

### Tools installed and checked

`Setup-ExtractionTools.ps1 -Stage Full` installed three Python environments and the models into `checkpoints/melody-extraction` (8.9 GB on disk). Every checkpoint matched its pinned SHA-256:

| File | Check | Result |
|---|---|---|
| `GAME-1.0-medium.zip` | GitHub release digest `8c5b3e53…` | Match |
| `GAME-1.0-medium/model.pt` | Team provenance `e9904159…` | Match |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | Team provenance `5b84f37e…` | Match |
| SheetSage2 `model.safetensors` | Team provenance `b235f680…` | Match |
| MERT-v2-FullSong `model.safetensors` | Team provenance `e6dd2ab1…` | Match |

GAME is at the team's pinned commit `f423934`. Resolved package versions are saved beside the tools in `environment.venv-*.txt`, and `local-provenance.json` records the differences from the team's Mac provenance.

Four problems surfaced during setup, and each one is now handled by the scripts:

1. **Mislabelled pin.** The team's `provenance.json` calls `e9904159…` the GAME archive hash. It is the hash of `model.pt` inside the archive. The setup script checks it against that file. The team file is left unedited; tell its author.
2. **Undeclared dependency.** `audio-separator` 0.47.0 imports `audioread` without declaring it. The first run therefore failed both selected songs, and the pipeline logged them as failed. The setup now installs `audioread`, and the launcher imports the separator before the pipeline starts.
3. **Silently incomplete download.** This network resets a share of connections to Hugging Face. `snapshot_download` then returned a partly filled MERT folder without an error, and the SheetSage2 worker crashed. The setup now fetches and confirms every file in the pinned listing.
4. **Slow FFmpeg host.** The setup takes the same build from its GitHub mirror and checks it against the checksum published on gyan.dev.

### What ran

`history.log` holds the full sequence, including the failures:

| Run | What happened |
|---|---|
| Index | `-IndexOnly`: 1,604 rows written, each with an artist ID and all five split labels |
| 1 | Tracks 382 and 383 failed on the missing `audioread` module. They remain `failed` in the manifest until a run with `-RetryFailed` succeeds. |
| 2 | Retry of track 382 (211 s) stopped by hand after 7 of 103 separator chunks in 7 min 41 s |
| 3 | `-First '1079195,1348174'`: vocal stage completed for both; the SheetSage2 worker then crashed on the incomplete download |
| 4 | Same command resumed: SheetSage2 ran on CPU and both songs reached `complete` |

Result for the two-song sample, both from training-partition artists with licenses that permit derivatives:

| Track | Length | GAME vocal notes | SheetSage2 notes | Combined notes |
|---|---:|---:|---:|---:|
| 1079195 | 30.5 s | 37 | 74 | 94 |
| 1348174 | 33.7 s | 1 | 57 | 46 |

Each track has MIDI, JSON and CSV under all three `outputs/` folders, `manifest.csv` shows 2 complete, 2 failed and 1,600 pending, and `catalog.json` lists the two completed tracks. Served locally, `viewer.html` lists both tracks with the three melody choices and reports no console error. The folder now holds every item in the team's layout, including the `__pycache__` folders Python writes when bytecode caching is on.

**Gate J1: the process works here on a small sample.** It does not scale here; see the next section.

## Speed

| Stage | Measured on this CPU | Per second of music |
|---|---|---:|
| Separation, pipeline settings, 8 threads | 913 s for the 30.5 s track | 30 s |
| Separation, 211 s song, default threads | 7 of 103 chunks in 7 min 41 s; about 1 h 50 min projected | 31 s |
| GAME on solo recordings | 527 s for 600 s of vocadito audio, start-up included | 0.9 s |
| SheetSage2 on CPU | Both sample songs in about 4 minutes, model load included | — |

The separator is the whole problem. It processes 8-second chunks every 2 seconds, and one chunk takes 45 to 100 seconds here. At that rate the 101 hours of local audio would need about 3,000 hours. GAME by itself is practical locally.

Ways to go faster, with what each does to accuracy:

| Option | Speed | Accuracy | Status |
|---|---|---|---|
| Same pipeline on a GPU: the team's Mac, or a CUDA machine | Not measured here; the pipeline was built for it | Unchanged: same models and settings | Recommended for any volume. The launcher runs there unchanged; the setup script is Windows-only. |
| `-Threads 8` | 164 s against 207 s (default 14) and 281 s (4) for a 10 s excerpt | Unchanged | Use it. About 20% faster. |
| `-SeparatorOverlap 2` | 635 s against 913 s on the same track; 7 chunks instead of 13 | Stem differs from the default stem by 29.9 dB below signal level. GAME notes agree with the default-stem notes at onset F1 0.79, against 0.81 for two GAME runs on the identical stem. | No cost visible on one track beyond GAME's own run-to-run variation. The comparison built the separator directly; the launcher option was confirmed to pass the same setting but has not produced pipeline outputs. Logged when used. Do not mix with default-setting stems. |
| Separate only a short excerpt of each song | About six times less work for a 35 s excerpt of a median song | Unchanged for the excerpt | Not built. The team pipeline is full-song by design; this would be a BPDR-side step. |
| A lighter separator model | Probably much faster on CPU | Unknown; a different model | Not tested. Needs another download and the same comparison as above before use. |

For BPDR the practical answer is that the slow stage is rarely needed. The extractor and natural-boundary roles below run GAME on solo audio, which needs no separation. Only the extra-voices role needs stems.

## Gate J2: GAME against vocadito

`extraction/check_game.py` ran the pinned GAME on the 28 full vocadito recordings of training and development singers (20 singers, 600 s). The PP1-evaluation and final-holdout singers were not read. Scores use `mir_eval` defaults: 50 ms onset tolerance, 50 cents pitch tolerance.

| Mean over 28 recordings | Onset only F1 | Onset and pitch F1 | With offset F1 |
|---|---:|---:|---:|
| A2 scored against A1 (the two annotators) | 0.82 | 0.73 | 0.65 |
| GAME, model pitch, against A1 | 0.75 | 0.63 | 0.30 |
| GAME, model pitch, against A2 | 0.85 | 0.77 | 0.42 |
| GAME, published pitch, against A1 | 0.75 | 0.00 | 0.00 |

What this shows:

- **Pitch field.** Of 1,120 onset-matched notes, 1,116 lie in the annotated octave, with a median difference of +6 cents. The published value, one octave up, scores zero. BPDR reads `model_midi_pitch`.
- **Segmentation.** On onset and pitch, GAME agrees with A2 (0.77) at least as well as the two annotators agree with each other (0.73), and less well with A1 (0.63). A1 simply marks more notes: 1,636 against 1,354 for A2 and 1,373 for GAME. The extractor's disagreement with a human is the same kind and size as the disagreement between two humans. That is the split-and-merge ambiguity BPDR studies.
- **Offsets.** GAME ends a note where the next begins in 87% of note pairs; the annotators do so in 15% and 16%. Offset-aware scores are therefore low by convention, and a boundary track from GAME is essentially an onset track.
- **Natural boundary variation.** On the eight shortest recordings, decoding the same waveform twice with identical settings gives onset F1 0.95 between the two results, because GAME samples its boundaries. A boundary threshold of 0.1 gives 36.0 notes per recording and 0.4 gives 30.6, against 33.5 by default. These are real alternative segmentations of one unchanged waveform.

**Gate J2 passes for a pilot.** GAME is adequate as a candidate fixed extractor and as a natural boundary source, read through `model_midi_pitch` and with offset-aware results labelled. Twenty-eight recordings do not make a benchmark.

## Gate J3: first look at stems

`extraction/check_stems.py` cuts each kept stem into windows of at most 20 seconds where GAME predicts the most singing, runs the Phase 1 path, and applies a reliability mask with no human reference: inside a GAME note, PESTO voiced, YIN within 50 cents of PESTO, and PESTO within 100 cents of the note. The same mask was applied to one crop from each of the 14 vocadito training singers.

| Median per crop | Jamendo stem | vocadito training crops |
|---|---:|---:|
| Crops, independent groups | 1 crop, 1 artist | 14 crops, 14 singers |
| Accepted by Phase 1 | 1 of 1 | 14 of 14 |
| PESTO voiced inside GAME notes | 0.26 | 0.84 |
| Reliable inside GAME notes | 0.17 | 0.76 |
| Longest reliable span | 0.18 s | 0.40 s |
| Crops able to host an audited shift | 0 | 1 |

**This comparison is uninformative about typical songs.** The two sample tracks were picked because they are the shortest eligible ones, which is what made a complete local run possible. By their titles and tags they are a 30-second "song loop" and an ambient piece, not ordinary songs with a sung melody. Track 1348174 gave no crop: GAME found a single 5-second note. In track 1079195, GAME reported 26 notes in the window while PESTO found a voice in a quarter of those frames. On the same stem, two GAME runs agreed at onset F1 0.81 with 37 and 47 notes, against 0.95 on vocadito. One vocadito crop also scored only 0.08 for voicing inside notes, so the mask depends on PESTO's voicing decision as well as on the audio.

**Gate J3 is open.** One crop from an atypical track cannot characterize the stems of 257 artists, in either direction. What the look does establish is that a voice tag plus predicted notes is not evidence of usable singing, so any Jamendo material needs automatic screening before review. A fair sample needs stems from full songs by many artists, which this machine cannot produce.

## A finding about the existing vocadito plan

The same measurement exposed a limit that is independent of the new dataset. The v0.3 renderer needs one reference-reliable span of at least 0.86 seconds: a 0.30 s plateau, two 0.08 s ramps and four 0.10 s guards. That is a steady note about 1.06 seconds long. Among the 30 vocadito training crops from 14 singers, the human-reference mask offers such a span in **3 crops, all from one singer**. The median longest span is 0.40 seconds. This is an upper bound, before the tracker-agreement checks.

So the planned multi-singer training subset cannot be generated under the current policy, whatever the reviewers decide. A 0.5-second requirement would admit 11 crops from 6 singers. Changing the geometry is a protocol revision on training material: it needs a new audit policy version, a repeat of the identity and realized-shift audits, and disclosure, as with v0.1 to v0.3.

## Suitability for BPDR

BPDR accepts a short unaccompanied single voice and learns a relative correction from references judged suitable to preserve.

| BPDR requirement | vocadito | MTG-Jamendo subset |
|---|---|---|
| Unaccompanied single voice | Yes | No. Requires separation; stems may keep bleed, reverb and backing voices, and the tag includes choirs. |
| At most 20 seconds | 21 of 40 directly; the rest through mapped crops | No track qualifies. Every one needs crops from a stem. |
| Format accepted by `bpdr-process` | WAV | No. MP3 is rejected; stems are WAV and were accepted. |
| Performed-pitch reference | Dataset F0 annotation | None. Only pitch trackers run on a stem. |
| Note boundaries | Two human annotators | One model's predictions |
| Independent groups | 29 singers | 257 artists as a proxy |
| Reviewed acceptable anchors | Sheets prepared; 114 rows pending | Nothing prepared |
| Resembles supported input | Closer: short solo vocal excerpts | Produced, released performances; unlikely to show novice errors |
| Local cost | Seconds per crop | About 30 seconds of computing per second of music |

**Verdict.** The subset cannot replace vocadito. `build_dataset` and `reference_mask` cannot run on it, and Phase 1 would stop every track for duration. It supplements vocadito in three roles, which the checks now rank clearly:

1. **Fixed downstream extractor: supported.** Running the same frozen GAME before and after preparation gives the proposal's extraction comparison and a real D6 hand-off. Fractional pitch allows that comparison in cents, where note F1 may not move for 10 to 30-cent changes. It needs GAME only, on BPDR's own solo audio, at about real-time speed. The melody-extraction owner still has to confirm GAME as the agreed extractor.
2. **Natural boundary condition: supported.** The current boundary tracks come from A1 annotation timings, while runtime has only an audio-derived score. GAME gives model-estimated boundaries and, through repeated or re-thresholded decoding, alternative segmentations of one unchanged waveform. That is the natural-analyzer condition the proposal asks to test separately, and a source for the controller's "other reasonable boundary versions" in offline evaluation.
3. **More independent voices: not supported yet.** Extraction is impractical locally, there is no human reference, and no informative stem sample exists. It stays optional, off the critical path, and behind J3 and J4. The shortage it was meant to ease is real, though: see the vocadito finding above.

## Gates

| Gate | Check | Status on 11 October |
|---|---|---|
| J1 | The pipeline completes on local songs; time recorded | Works on a two-song sample. Not practical for volume on this CPU. |
| J2 | GAME on vocadito against A1 and A2; pitch field fixed | Passed for a pilot. Read `model_midi_pitch`; label offset-aware results. |
| J3 | Stems cut to 20-second crops; reliability and shift audit against vocadito | Open. The one crop measured came from a track chosen for being short and is uninformative. Needs stems from full songs by many artists. |
| J4 | Two-reviewer sheet for a declared Jamendo crop subset | Not started. Depends on J3. |

Until J3 and J4 pass, Jamendo crops may support engineering runs that are labelled exploratory. They do not enter the reviewed comparison.

## License notes

MTG-Jamendo is released for non-commercial research and academic use, with per-track Creative Commons licenses listed in `audio_licenses.txt`. Detuned or separated audio is a derivative. Intervention material should therefore come only from the 1,251 tracks whose license permits derivatives, and no NoDerivatives track should supply demonstration or presentation audio. SheetSage2 and MERT-v2-FullSong weights are CC BY-NC 4.0. Attribution for any played excerpt comes from `audio_licenses.txt`. None of this is legal advice; confirm the intended presentation use with the supervisor.

## Evidence locations

All under ignored folders, so they stay on this machine: `data/mtg-jamendo/melody-extraction-pipeline/history.log`, `manifest.csv` and `outputs/`; `checkpoints/melody-extraction/local-provenance.json`; and `runs/extraction_checks/` with `0e9888a4…/game_on_vocadito.json` (J2), `3fa67dcd…/stems_vs_vocadito.json` (J3) and `c357df60…/overlap_comparison.json`. Rerun the commands in the README to regenerate them.
