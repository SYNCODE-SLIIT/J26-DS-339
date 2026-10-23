# Vishwa — PP1 dataset and implementation audit (17 Oct 2026)

Status: Planning / verification only. No trained model or evaluation results are claimed.

## Component contract
Input: accepted monophonic vocal audio (or directly supplied solo vocal audio for standalone testing).
Output: versioned JSON containing note events (pitch, onset, duration, confidence), F0 trajectory, beat/tempo/key estimates when supported, phrase boundaries when supported, and optional harmony features/embedding. Unsupported fields must be null with provenance, not invented.

## Dataset audit
| Dataset | Verified annotation strengths | Missing/unverified for unified model | PP1 decision |
|---|---|---|---|
| HumTrans | WAV humming audio, paired MIDI note labels, official train/valid/test keys | Beat, tempo, key, phrase and chord-label supervision not established | Primary short sample for audio-to-note checks; respect source split and noncommercial license |
| MIR-1K | Singing-vocal channel and manual pitch contours, voiced/unvoiced annotations | Beat, key, phrase, harmony annotations not established | Pitch evaluation / spectrogram inspection |
| MedleyDB | Multitrack audio, selected melody and stem-pitch annotations | Label coverage varies per track; solo-vocal suitability and access need screening | Optional vocal-stem evaluation; not mandatory for PP1 |

Sources:
- https://huggingface.co/datasets/dadinghh2/HumTrans
- https://zenodo.org/records/3532216
- https://github.com/marl/medleydb
- https://medleydb.weebly.com/downloads.html

## First dataset audit fields
recording ID, singer/source ID, dataset, audio path (local only), duration seconds, sample rate, channels, peak amplitude, clipping %, voiced %, pitch annotation available, MIDI annotation available, license, split, preprocessing log.

## Leakage controls
- Retain HumTrans official split; do not place alternate renditions of the same source across experimental splits.
- Group splits by recording/singer/source where applicable, before augmentation.
- Keep raw audio, participant recordings, private data, and large datasets out of Git.
- Report which metrics are actually supported by labels.

## PP1 acceptance criteria
1. On a small local solo vocal sample: decode and normalize audio without time drift.
2. Save waveform, mel-spectrogram and F0 visualization.
3. Produce time-aligned baseline F0 and note events; document limitations.
4. Serialize reproducible JSON and provide smoke tests.
5. Report metrics only on genuinely annotated data, alongside data-analysis plots.
6. Prepare a 10-minute group demo/video and group presentation with clear individual contribution.

## Research scope note
Full jointly trained CNN-Transformer, phrase supervision, and harmonization-optimized embeddings remain proposed research goals until implemented and experimentally tested.
