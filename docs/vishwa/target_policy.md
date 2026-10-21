# Target policy — weak-score-chroma-v1

Recorded 2026-10-21, decisions D013–D015, experiments R006–R008.

## Separate target meanings

- **Performed pitch:** actual acoustic fundamental frequency. pYIN and Praat are estimates with method provenance, not manually verified truth.
- **Score pitch:** original downloaded MIDI annotation. Preserve pitch, onset and offset unchanged. Its relationship to the performed octave and precise timing remains under review.
- **Current feasibility target:** original MIDI pitch class plus rest: 0=rest, 1=C, 2=C♯, …, 12=B. Octave invariance discards a relevant part of the eventual transcription task. It does not solve performed-pitch prediction.

## Frame and split policy

Use original timestamps, untrimmed audio, 22050 Hz, 4096 FFT, 256 hop and 80 Mel bands. Log power is relative to each recording's maximum. Per-band mean/std is fitted on TRAIN frames only. The fixed pilot uses 20 TRAIN and 10 VALID recordings from the original official splits. VALID was inspected previously and is development evidence; TEST is untouched.

Exclude supervision within 100 ms of note and recording boundaries. This heuristic removes roughly 42% of frames, including some short notes, and does not certify the remainder. The frame target assumes monophony; overlapping active reference labels are rejected. No estimator-based octave or timing correction is applied.

## Model interpretation

CNN and CNN–Transformer prototypes predict these 13 classes. Report masked frame metrics separately from strict onset-sensitive note metrics. These prototypes do not implement the full proposal's CFP, multitask beat/key/phrase outputs or learned harmony embeddings. They are not integrated as replacements for the existing baseline.

## Next evidence required

Listen to the eight shared-octave TRAIN cases and inspect the waveform, spectrum, both F0 tracks and original MIDI. Record reviewer identity, confidence, evidence and any independently justified annotation decision. Do not mark algorithm agreement as human verification. Define reviewed targets before a larger preregistered, repeated-seed experiment. Do not tune repeatedly to the existing ten validation examples.
