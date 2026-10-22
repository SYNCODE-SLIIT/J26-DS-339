# R009: eight-case listening protocol

## Open the packet

From the project root:

```sh
.venv/bin/python scripts/build_listening_review.py
open datasets/humtrans/derived/research/listening_review/index.html
```

The generated page contains all eight original recordings, waveform/pitch plots,
and sine references at MIDI −24, −12, 0 and +12 semitones. All generated audio and
images stay in the ignored derived-data directory. Rebuilding preserves existing
decision records. Source hashes must match the earlier R006 audit.

## Review each case

1. Listen to the original before opening the automatic diagnostic panel. Note
   timestamps of stable notes, ambiguous regions, noise or register changes.
2. Compare the original-score reference first, then the octave alternatives.
   Sine tones have different timbre; perceptual similarity alone is insufficient.
3. Inspect both pitch tracks and MIDI over the same times. Distinguish octave
   differences from timing errors, unstable voice and estimator mistakes. Do not
   infer a correction from the filename suffix or the strongest spectral band.
4. Enter a record in `experiments/vishwa/results/listening_review_decisions.json`.
   Set status to `reviewed` only after listening. Provide your reviewer name,
   ISO date/time, both listening confirmations, confidence (`low`, `medium`,
   `high`), and evidence with timestamps. Leave ambiguous cases `unresolved`.
5. Run `.venv/bin/python scripts/check_listening_review.py` to validate the form.

Allowed outcomes: `consistent_with_original`,
`possible_score_performance_octave_difference`, `possible_estimator_error`,
`timing_or_mixed_issue`, `unresolved`. Suggested shift may be null, −24, −12, 0 or +12;
this is a hypothesis and does not modify labels. A second independent reviewer
is useful for disputed cases. The checker checks completeness, not truthfulness
or musical correctness.

## Decision boundary

No recordings or MIDI are edited by these scripts. A completed review does not
itself authorize relabeling. Record a separate target-policy decision with its
supporting evidence before creating versioned derived annotations. Keep original
score and performed pitch separate. Do not enlarge model training merely because
a transposed reference agrees better with two algorithms.

The assistant prepared automatic evidence and visual checks; human listening
remains pending. The page is a playback aid and does not save form entries.
