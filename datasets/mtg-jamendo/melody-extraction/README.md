# MTG-Jamendo melody extraction pilot

This folder holds a reproducible **25-song pilot**. `sample.csv` lists the original
MP3s and the deterministic, 30-second excerpt selected from each one. The input
`../vocal-tagged-audio/` tree is read only. All timestamps in result files are
relative to each excerpt; add `clip_start_sec` from `sample.csv` to locate a note
in the original song.

[`round-2/`](round-2/README.md) contains a separate 20-song comparison. It uses
the same code and model files while keeping its clips, stems, predictions, and
listening page separate. Set `MELODY_RUN_DIR` to that directory when running any
of the commands below for the second sample. Its combined MIDI takes GAME vocal
notes and adds SheetSage2 instrumental notes in longer vocal-free sections;
`code/combine_melody.py` creates that output after both model stages finish.

## Folder layout

| Path | Purpose |
| --- | --- |
| `code/` | Our sampling, inference, conversion, and reporting code |
| `tools/` | Official upstream model source checkouts; ignored by Git |
| `models/` | Downloaded checkpoints; ignored by Git |
| `clips/` | Decoded WAV excerpts from the original MP3s |
| `stems/` | Separated vocal WAVs with the same track IDs |
| `results/game_vocal/` | GAME from separated vocals: MIDI, raw CSV, canonical JSON, preview WAV |
| `results/game_mix/` | GAME directly from the original mix, kept separate |
| `results/yourmt3/` | YourMT3+ multi-instrument MIDI and canonical JSON |
| `results/yourmt3/parts/` | One flat MIDI/JSON pair per non-drum instrument track |
| `results/sheetsage2/` | SheetSage2 instrumental and vocal melody MIDI, canonical JSON, previews, and raw events/rhythm/chords |
| `summary.csv` | One row per selected song, including note counts and review hints |
| `review.csv` | Persistent, editable listening-review sheet; regeneration leaves it intact |
| `report.html` | Local audio players for the mix, vocal stem, and each MIDI preview |
| `compare.html` | Compact song selector and A/B player for the original excerpt and MIDI previews |

GAME's raw CSV preserves fractional MIDI pitch (cents). MIDI rounds that pitch
to semitones for conventional playback; the canonical JSON retains the CSV
value. YourMT3+ JSON retains instrument program, drum flag, timing, and velocity
from its multitrack MIDI. Predictions are **unreviewed pseudo-labels**; note
counts and stem loudness do not establish note accuracy. A silent stem may mean
an instrumental excerpt, a failure of separation, or a quiet singer.
`summary.csv` also gives relative paths to every available JSON result and
individual instrument part, so a training loader can join notes to the frozen
song selection without searching the filesystem. GAME JSON note events contain
`onset_sec`, `offset_sec`, fractional `midi_pitch`, and `part`; YourMT3+ events
add `velocity`, `program`, and `is_drum`.
SheetSage2's instrumental and vocal melodies are separate, single-part MIDI and
JSON files. Its `events.json`, beat/downbeat and chord LAB files are saved beside
the melodies for later rhythm-aligned analysis. No melody selection or gap
filling is applied in this pilot.

## Reproduce the pilot

Run from the repository root. The root `.venv` provides the MLX separator,
PrettyMIDI, NumPy, and SoundFile. GAME, YourMT3+, and SheetSage2 each use their own
environment because their dependency versions differ.

```bash
.venv/bin/python datasets/mtg-jamendo/melody-extraction/code/pipeline.py sample --count 25 --seconds 30 --seed 20261008
.venv/bin/python datasets/mtg-jamendo/melody-extraction/code/pipeline.py prepare
.venv/bin/python datasets/mtg-jamendo/melody-extraction/code/pipeline.py separate
.venv/bin/python datasets/mtg-jamendo/melody-extraction/code/pipeline.py game_vocal
.venv/bin/python datasets/mtg-jamendo/melody-extraction/code/pipeline.py game_mix
datasets/mtg-jamendo/melody-extraction/.venv-yourmt3/bin/python datasets/mtg-jamendo/melody-extraction/code/yourmt3_worker.py
.venv/bin/python datasets/mtg-jamendo/melody-extraction/code/pipeline.py yourmt3_parts
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 datasets/mtg-jamendo/melody-extraction/.venv-sheetsage2/bin/python datasets/mtg-jamendo/melody-extraction/code/sheetsage2_worker.py
.venv/bin/python datasets/mtg-jamendo/melody-extraction/code/pipeline.py review_template
.venv/bin/python datasets/mtg-jamendo/melody-extraction/code/pipeline.py summary
.venv/bin/python datasets/mtg-jamendo/melody-extraction/code/pipeline.py report
.venv/bin/python datasets/mtg-jamendo/melody-extraction/code/pipeline.py compare
```

`sample.csv` saves the frozen selection. The `sample` command will
refuse to replace it unless `--overwrite-sample` is supplied. The stages can be
resumed; existing audio and result files are reused. `run_status.json` records
per-song stage outcomes. GAME may emit no file when it finds no notes; the
pipeline records a valid empty MIDI and JSON with status `empty_prediction`.

The separator uses the BS-RoFormer checkpoint
`model_bs_roformer_ep_317_sdr_12.9755.ckpt`. GAME uses the official
`GAME-1.0-medium` release checkpoint. YourMT3+ uses the official Space's
`YPTF.MoE+Multi (noPS)` checkpoint. SheetSage2 uses its pinned Hugging Face
checkpoint plus the pinned MERT-v2-FullSong parent model. GAME and YourMT3+
source is under `tools/`; SheetSage2 source and all checkpoints are under
`models/`. Song predictions remain under `results/`.
Exact upstream commits, model hashes, and key inference settings are in
`provenance.json`.

For SheetSage2, install its official `requirements.txt` plus `soundfile` in `.venv-sheetsage2`
using Python 3.10 or 3.11, then download the two pinned Hugging Face snapshots
under `models/hf-home/` (set `HF_HOME` to that path when downloading):

```bash
export HF_HOME="$PWD/datasets/mtg-jamendo/melody-extraction/models/hf-home"
datasets/mtg-jamendo/melody-extraction/.venv-sheetsage2/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("m-a-p/SheetSage2", revision="32c7c7473e7b59e1ed36bd8a8fefbd11ae8d468a",
                  ignore_patterns=["render_assets/*", "benchmarks/*", "tests/*"])
snapshot_download("m-a-p/MERT-v2-FullSong", revision="d8ba1c745e733b3908ce6ad16ebeb17ac7600a42")
PY
```

The worker
uses local files only and saves both instrumental and vocal melodies for each
excerpt. Its `--limit 1` option is useful for a first check. The audio previews
are simple sine renders of MIDI, not SheetSage2's timbre rendering.
The run used the official full-task prompts and five-minute model window on each
30-second original-mix excerpt. In this sample it produced 917 instrumental notes
across 21 excerpts and 504 vocal notes across 12 excerpts. Empty lines and note
counts need listening review; neither is an accuracy score.

For a fresh checkout, install the root dependencies with `uv sync`, then clone
the two upstream repositories into `tools/GAME` and `tools/YourMT3-Space` using
the commits listed in `provenance.json`. Install GAME's `requirements.txt` with
PyTorch in `.venv-game`; install PyTorch 2.8, torchaudio 2.8, NumPy 1.26,
Transformers 4.45.1, Lightning, librosa, mido, mir-eval, mirdata, einops,
deprecated, wandb, smart-open, SoundFile, and SciPy in `.venv-yourmt3`.
Download the two checkpoints from the URLs in `provenance.json` into `models/`.
Unzip GAME to `models/GAME-1.0-medium/` and point the YourMT3+ Space's
`amt/logs/2024/.../checkpoints/last.ckpt` at the downloaded checkpoint. The
separator obtains its named BS-RoFormer model through `mlx-audio-separator` if
it is absent from `models/mlx-audio-separator/`. Model inference here used the
Mac's Metal GPU; a CPU run will take longer.

## Review before training

Open `compare.html` in a browser for quick A/B listening, or `report.html` for all
tracks on one page. Listen to the original, separated vocal, and
synthesized note previews for each ID. In particular, review rows marked
`quiet_vocal_stem` in `summary.csv`, and any large disagreement between GAME
on the stem and on the mix. Record decisions in `review.csv`; running the
pipeline again will preserve that sheet. The report is a listening aid, not an accuracy
measurement. For training, correct a held-out set by hand and compare onset,
pitch, and offset scores before selecting or filtering pseudo-labels. Keep
artist-level splits and the original timed JSON; derive beat-aligned transformer
tokens as a separate dataset view.
