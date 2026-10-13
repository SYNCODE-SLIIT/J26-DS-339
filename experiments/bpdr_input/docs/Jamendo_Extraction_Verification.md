# MTG-Jamendo source and melody-extraction check — 11 October 2026

**Confirmed on this machine:** the second data source is mirrored inside the component; the team's pipeline code runs unedited through a launcher; the complete pipeline produced all three outputs for a two-song sample; and the team's listening page shows them. **Also found:** the vocal separator is far too slow on this CPU for full-length songs but runs about ten times faster, with the same result, on the laptop's Intel Arc GPU; and the GAME note model reads performed pitch correctly only through its unshifted field. The first two sample tracks were chosen for being short and are not typical; one normal-length song has since been measured. Nothing here is a BPDR result or a training-ready collection.

**Added 12 October:** three further checks and a decision are recorded in [their own section](#12-october-hand-off-crop-and-excerpt-checks). The team's GAME reads BPDR's accepted audio as well as the original, notes decoded from a 20-second crop match notes cut from a full recording, and one excerpt separated by itself matched the full-song stem. The owner decided that BPDR builds its own data path on the same pinned models, using from each dataset only what suits the component.

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
  extraction/check_handoff.py  gate H1, added 12 October: GAME on BPDR's accepted audio against the original
  extraction/check_excerpt.py  gate J5, added 12 October: an excerpt separated by itself against the full-song stem
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
| 5 | `-First '1038720' -Gpu -VocalOnly -KeepStems`: a 162-second song separated on the Intel GPU; GAME found 292 notes; the song is `partial` because SheetSage2 was skipped |

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
| Separation on the Intel Arc GPU, pipeline settings | 16.6 s for a 10 s excerpt, against 163.5 s on the CPU; 78 chunks of a 162 s song at about 7.9 s each | 4 s |
| GAME on solo recordings | 527 s for 600 s of vocadito audio, start-up included | 0.9 s |
| SheetSage2 on CPU | Both sample songs in about 4 minutes, model load included | — |
| One BPDR training step (8 pairs, 940 frames, four views) | 13.2 s on the CPU, 0.73 s on the Intel Arc GPU | — |

The separator is the whole problem. It processes 8-second chunks every 2 seconds, and one chunk takes 45 to 100 seconds on the CPU. At that rate the 101 hours of local audio would need about 3,000 hours. On the Intel Arc GPU a chunk takes about 8 seconds, so a normal song takes 10 to 15 minutes and a sample of 20 to 30 songs fits in a night. All 1,604 songs would still need about 400 hours. GAME by itself is practical locally.

**Standby pauses a run.** During run 5 the laptop entered Modern Standby several times, which stretched 10 minutes of computing to 41 minutes on the clock. Keep the laptop awake and on power for long runs.

Ways to go faster, with what each does to accuracy:

| Option | Speed | Accuracy | Status |
|---|---|---|---|
| `-Gpu`: the laptop's Intel Arc GPU through the Intel-GPU build of PyTorch | About 10 times faster than the CPU on the same excerpt | Unchanged: same model and settings. The GPU stem differs from the CPU stem by 101 dB below signal level. Peak GPU memory 3.75 GB. | Use it. Install with `Setup-ExtractionTools.ps1 -IntelGpu`. The package's own DirectML route keeps this model on the CPU, so the launcher sets the device itself and logs it. |
| Same pipeline on the team's Mac or a CUDA machine | Not measured here; the pipeline was built for it | Unchanged: same models and settings | Still the choice for all 1,604 songs. The launcher runs there unchanged; the setup script is Windows-only. |
| The NPU | — | — | Not attempted. Neither model exists in a form it can run. |
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

**One normal-length song, added after the GPU option worked.** Track 1038720 (162 seconds, a training-partition artist, license permits derivatives) was separated on the GPU and gave two crops. Phase 1 accepted both.

| Per crop | 1038720 at 55 s | 1038720 at 125 s | vocadito median |
|---|---:|---:|---:|
| PESTO voiced inside GAME notes | 0.68 | 0.71 | 0.84 |
| Reliable inside GAME notes | 0.50 | 0.40 | 0.76 |
| Longest reliable span | 0.30 s | 0.40 s | 0.40 s |
| Able to host an audited shift | No | No | 1 of 14 |

So a real song's stem is usable by the input path and gives roughly half to two thirds of vocadito's reliable share, with reliable spans of similar length. Neither crop has a note long enough for the v0.3 shift, which is the same limit most vocadito crops hit. This is one song by one artist.

**Gate J3 is open.** One song cannot characterize the stems of 257 artists, in either direction. What the checks do establish is that the path from song to accepted crop works, and that a voice tag plus predicted notes is not evidence of usable singing, so any Jamendo material needs automatic screening before review. A fair sample needs stems from full songs by many artists; with the GPU option this machine can now produce 20 to 30 of them in a night.

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
3. **More independent voices: not supported yet.** There is no human reference, and only one normal song has been measured. Extraction of a sample is now practical on the laptop's GPU. The role stays optional, off the critical path, and behind J3 and J4. The shortage it was meant to ease is real, though: see the vocadito finding above.

## 12 October: hand-off, crop and excerpt checks

Three questions decide how BPDR should take data from this source, and whether the next component can read what BPDR delivers. All three were measured with the pinned GAME model. GAME samples its note boundaries and its `extract` command has no seed option, so two decodes of the same audio differ. Every comparison below is read against that run-to-run figure, called the noise floor. Scores use `mir_eval` defaults (50 ms onset, 50 cents pitch); offsets are not scored.

### Does GAME read BPDR's accepted audio like the original? (gate H1)

BPDR delivers a mono 22,050 Hz PCM24 WAV, while GAME works at 44,100 Hz and gate J2 fed it the original files. `extraction/check_handoff.py` decoded the 17 crops of the J3 run three times: the crop as given to Phase 1, the same crop again, and the Phase 1 accepted WAV.

| Mean per crop: onset F1 / onset-and-pitch F1 | vocadito: 14 crops, 14 training singers | Jamendo stems: 3 crops, 2 tracks |
|---|---:|---:|
| Same crop decoded twice (noise floor) | 0.96 / 0.91 | 0.90 / 0.87 |
| Accepted WAV against the crop as given | 0.96 / 0.93 | 0.91 / 0.87 |
| Notes from the full recording or full-song stem, cut to the window, against the crop | 0.95 / 0.91 | 0.90 / 0.84 |

The same decodes of the vocadito crops, scored against the two annotators:

| GAME input | Against A1 | Against A2 |
|---|---:|---:|
| Crop as given | 0.73 / 0.60 | 0.83 / 0.74 |
| Same crop, second decode | 0.72 / 0.59 | 0.83 / 0.73 |
| Accepted WAV | 0.74 / 0.61 | 0.84 / 0.75 |
| Full recording, cut to the window | 0.73 / 0.60 | 0.84 / 0.73 |

What this shows:

- **Hand-off.** GAME's notes on the accepted WAV agree with its notes on the original at the noise floor, and they are no worse against either annotator. Converting to mono 22,050 Hz costs nothing measurable here. **Gate H1 passes for technical preparation, as a pilot.** Every one of these crops was `ACCEPT_UNCHANGED`: no repair exists yet, so the check must be repeated on prepared audio.
- **Crop or full recording.** Notes decoded from a 20-second crop agree with notes cut from the full recording at the noise floor on vocadito. On the three Jamendo crops the onset-and-pitch figure is 0.84 against a floor of 0.87, which three crops cannot settle. BPDR can therefore decode each accepted crop itself. That is the audio the next component will receive.
- **Cost.** One decode of the 17 crops (275 seconds of audio) took 500 and 565 seconds when nothing else was running, about 2 seconds per second of audio. That is about twice as slow as the 0.9 seconds measured in gate J2 on full recordings, and the cause is not established. Until it is measured again, budget the data path's boundary step, which decodes every accepted crop and then repeats it, at the slower rate. The first decode took 1,056 seconds because the excerpt check below ran beside it.
- **Reproducibility.** With no seed, a nominal boundary track is one recorded decode. Repeat decodes are real alternative segmentations and have to be saved, not regenerated.

### Does an excerpt separated by itself match the full-song stem? (gate J5)

Separation is the slow stage, and BPDR needs crops of at most 20 seconds. `extraction/check_excerpt.py` separated seconds 45 to 85 of track 1038720 on the Intel GPU with the pipeline's settings, then compared seconds 55 to 75 with the same window of the stem kept from the full-song run.

| Measure | Result |
|---|---|
| Difference between the two stems | 40 dB below signal level; best-match gain 1.000 |
| GAME, full-song stem decoded twice (noise floor) | Onset F1 0.97, onset-and-pitch 0.94; 59 and 58 notes |
| GAME, full-song stem against excerpt stem | Onset F1 0.98, onset-and-pitch 0.97; 59 and 57 notes |
| Audio separated | 40 seconds in place of 162, a quarter of the song |
| Separation time | 221 seconds, taken while the hand-off check was running, so it is not a clean speed figure |

No cost is visible beyond GAME's own variation. This is one window of one track. **Gate J5 is open** until the same holds on several windows from several artists. Excerpt stems are logged as such and are not mixed with full-song stems before then.

Separating excerpts raises a question that is not solved: the J3 crop rule picks the windows with the most predicted singing from full-song notes, which an excerpt run does not have. The window has to be chosen before separation by a fixed rule, such as fixed positions in the song or a cheap measure on the mix. None has been measured. See the phase plan.

### The team's newer pipeline, examined and not adopted

On 12 October the team merged `melody-chord-extraction-pipeline` (repository commit `f2ad4f0`). It runs nine stages per song: the same three melody stages, then chords from LV-Chordia, a key estimate and key-relative chord labels. It was read, not run.

- Its vocal stage (`code/melody.py`) is the same separation and GAME code as the mirrored `melody-extraction-pipeline`, with the same checkpoints, the same 12-semitone shift and the same `model_midi_pitch` field. It gives BPDR no more accurate notes.
- Chords, key and Roman labels describe the accompaniment. BPDR's input is an unaccompanied voice, so they are not BPDR inputs.
- It still deletes the vocal stem after note extraction.
- It does not run through the launcher as it stands. Its start-up check looks for `bin/python` without `.exe`; it needs `lv-chordia` 1.1.0, which is in neither launcher environment; the launcher's index option, vocal-only option and tool check call names it no longer has; and it has no way to run the vocal stage alone without recording the other stages as errors.
- Its outputs move to `outputs/<mtg_track_id>/melody/game_vocal.json`.

### Decision of 12 October

The owner decided that BPDR uses from each dataset only what suits the component, and builds its own data path instead of running either team pipeline for its data. The measurements above support the following shape, which the [phase plan](PP1_Phase_Execution_Plan.md) turns into tasks:

- **Same models.** The pinned BS-RoFormer checkpoint and the pinned GAME commit and checkpoint stay exactly as the team has them. BPDR's notes then mean the same as the melody component's notes. A different separator or note model would need its own J2 and H1 before use.
- **Own orchestration.** Selection by artist and license, separation with stems kept, crops of at most 20 seconds, Phase 1 on every crop, GAME on the accepted crop, screening and packaging all live in the component.
- **Both datasets through one path.** vocadito keeps its human references and stays the only evaluation material. Jamendo adds voices for exploratory training material. Each example records which kind of reference it has.
- **The mirror stays.** `data/mtg-jamendo/melody-extraction-pipeline` remains unedited as the reference for the team's format and as the working route to full-song stems.

Nothing in this decision changes what makes an example training eligible. Jamendo crops still need gates J3 and J4.

## Gates

| Gate | Check | Status on 12 October |
|---|---|---|
| J1 | The pipeline completes on local songs; time recorded | Works on a two-song sample, and on a normal-length song through the GPU option. A sample of tens of songs is practical on the GPU; all 1,604 are not. |
| J2 | GAME on vocadito against A1 and A2; pitch field fixed | Passed for a pilot. Read `model_midi_pitch`; label offset-aware results. |
| J3 | Stems cut to 20-second crops; reliability and shift audit against vocadito | Open. One normal song gave 40 to 50% reliable frames against 76% for vocadito. Needs stems from full songs by many artists, which the GPU option now allows. |
| J4 | Two-reviewer sheet for a declared Jamendo crop subset | Not started. Depends on J3. |
| J5 | An excerpt separated by itself matches the full-song stem | Open. One window of one track matched within GAME's own variation. Needs several windows from several artists. |
| H1 | GAME on BPDR's delivered audio agrees with GAME on the original within its own variation, and is no worse against A1 and A2 | Passed for technical preparation on 14 vocadito training crops and 3 Jamendo crops. To be repeated whenever preparation changes and once a repair exists; the final run uses the PP1-evaluation singers and belongs to P5. |

Until J3 and J4 pass, Jamendo crops may support engineering runs that are labelled exploratory. They do not enter the reviewed comparison.

## License notes

MTG-Jamendo is released for non-commercial research and academic use, with per-track Creative Commons licenses listed in `audio_licenses.txt`. Detuned or separated audio is a derivative. Intervention material should therefore come only from the 1,251 tracks whose license permits derivatives, and no NoDerivatives track should supply demonstration or presentation audio. SheetSage2 and MERT-v2-FullSong weights are CC BY-NC 4.0. Attribution for any played excerpt comes from `audio_licenses.txt`. None of this is legal advice; confirm the intended presentation use with the supervisor.

## Evidence locations

All under ignored folders, so they stay on this machine: `data/mtg-jamendo/melody-extraction-pipeline/history.log`, `manifest.csv` and `outputs/`; `checkpoints/melody-extraction/local-provenance.json`; and `runs/extraction_checks/` with `0e9888a4…/game_on_vocadito.json` (J2), `3fa67dcd…/stems_vs_vocadito.json` and `ed4b3433…/stems_vs_vocadito.json` (J3, before and after the normal-length song) and `c357df60…/overlap_comparison.json`. The 12 October checks are in `be8e05a2…/handoff_and_crop_check.json` (H1 and crop against full recording) and `d0a56681…/excerpt_comparison.json` (J5). The GPU probe's timings are recorded in this file only. Rerun the commands in the README to regenerate them.
