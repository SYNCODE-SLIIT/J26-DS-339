# Perera E. V. — preprocessing and melody baseline

This section implements the first executable foundation for proposal SO1/SO5
and CFR-01/02/03/04/08/09. These objectives are only partially fulfilled:
this is an acoustic baseline, not the proposed jointly trained CNN-Transformer.

## Run

From the repository root:

```sh
uv run python scripts/analyze_vocal.py datasets/humtrans/downloads/wav_data_sync_with_midi/F01_0001_0001_1_D.wav --output datasets/humtrans/derived/baseline/example.json
uv run python -m unittest discover -s tests -v
```

The output path must not already exist. `datasets/` is ignored by Git.
Use another output name for repeated runs. No recording is modified.

## Decisions

- Preserve the complete recording by default. Optional `--trim` records the
  offset and keeps output timestamps relative to the original audio.
- Convert to mono and resample to 22,050 Hz; normalize peak to 0.95.
  No denoising, pitch correction, or time stretching is applied.
- Use an 80-band Mel spectrogram and pYIN with 4096-sample frames and
  256-sample hops. Pitch search covers C1–C7.
- Group voiced frames with the same rounded MIDI pitch; discard events shorter
  than 60 ms. This baseline can fragment vibrato and merge repeated notes.
- JSON v0.1.0 carries notes, frame F0, voiced probabilities, original timeline,
  configuration and library versions. Inference time naturally varies.
- A duration-weighted pitch-class distribution is derived from predicted notes.
  It is not a learned harmony representation or a chord annotation.
- CFP, beats, tempo, key, phrases, and embeddings are explicitly unsupported/null.
  Voiced probability must not be interpreted as note correctness confidence.

## Research boundary and next stage

The proposal (pages 26, 30, 39–43) places vocal musical representation in this
component; chord planning, refinement and accompaniment remain other components.
Correction (2026-10-21): the initial 834 MIDI-tail flags were caused by the
notebook tempo parser. Independent Mido/PrettyMIDI timing agrees on all 14,614
pairs, with zero tails beyond 50 ms. This validates decoding, not perceptual
alignment. See docs/vishwa/research_progress.md for the preserved correction. Preserve official splits;
use training data for development, validation for tuning and reserve test data.
Next: reproducible pair manifest, annotation alignment checks, then baseline
note/frame evaluation using supported labels. CFP and the supervised model follow
once feature and target alignment is established. Beat/key/phrase supervision
has not been established by the supplied HumTrans files.
