### Historical evidence — separate from this upload

- HumTrans audit: 14,614 paired recordings, approximately 56.218 audio hours and 391,595 reference note events.
- Official split: 13,080 TRAIN / 765 VALID / 769 TEST. Singer overlap limits claims about unseen singers.
- An earlier MIDI timing-parser error produced 834 false flags; corrected parsing and independent cross-checking removed these flags.
- Fixed feasibility cohort: 20 TRAIN recordings and 10 VALID recordings. Final TEST evaluation was not performed.
- Recording-macro weak score-class accuracy: majority 12.10%, pYIN 48.55%, CNN 42.53%, CNN–Transformer 39.45%. This is not note transcription F1 and is not measured on the current upload.
- Strict validation note F1 was approximately 0.0281: exact note boundaries remain a substantial limitation.
- Eight listening cases have preliminary user timing observations. These are not verified corrected annotations and are never applied automatically to new uploads.

The graphs below show actual outputs from the selected run. Model agreement is not evidence of annotation correctness.
