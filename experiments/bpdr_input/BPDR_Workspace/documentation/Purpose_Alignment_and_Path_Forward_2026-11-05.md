# Purpose alignment and path forward (5 Nov 2026)

## Where BPDR sits
Project system (repo `system.md`): vocal melody -> melody understanding -> genre-conditioned harmonic planning -> chord refinement -> validation.
BPDR is the audio-preparation component: audio integrity, acoustic evidence, bounded local relative-waveform repair, protection/rollback/review,
accepted-WAV provenance. It does NOT own final notes/durations, key, tempo, genre or harmony (objective file `goal-objective.md`). Its only reason to exist
downstream is that cleaner audio improves note extraction without damaging correct audio.

## What the evidence now says (this session; none independently replayed)
- Learned Transformer on pitch-derived features cannot separate injected short detunes from natural movement (folder names `v70_detection_diagnosis`,
  `v71_oracle_level`, `v72`, `v73`; AUROC .32-.64; abstain gate protects originals but catches <1%).
- NUMBERING: this session's `v70_detection_diagnosis` and `v71_oracle_level` collide in label with the project's own `v70_original_coverage` and
  `v71_coverage_delivery`. Refer to the session's studies by folder name. No files overwritten.
- Literature (reports/Short note pitch repair literature.md): every corrector gets its target from outside the audio (scale, score, accompaniment, human);
  natural singing is ~19-25c from a note target; trackers' own error (2-14c) is near the verification floor on 0.24-0.5 s spans.
- Oracle reference (exact original) upper bound: macro error 24.9 -> 8.6c, 294/445 verified beneficial (`v76`); with a realistic stand-in reference
  (semitone snap of the singer's own notes, `v79`) 33% of natural note spans would be falsely flagged -> silent repair not viable; review-only is.
- Purpose check (`results/transformer_usability/purpose_check.py`, descriptive, 434 notes from 53 original windows): a 10/20/30c detune changes the nearest-semitone
  label of 7.6/13.2/24.5% of notes on an absolute grid (3.9/10.1/14.9% with per-recording tuning). So even perfect repair of every 20-30c error could change only
  roughly 10-25% of transcribed notes; an unwanted edit of similar size changes a similar share. Downstream benefit has NOT been measured.

## Decision
1. Stop model-architecture/loss work for short-note repair; it is information-limited, not capacity-limited.
2. Promise review-only behaviour: measure deviation against an externally supplied target, flag sustained spans with time ranges, repair only on user accept
   with existing rollback/decoded-WAV verification/provenance; abstain without a reference. Held-note path unchanged.
3. Highest-value next measurement (not yet run): does repairing toward a reference change the downstream note labels/F-score on the same clips
   (notes owner's extractor, offline MIDI only)? If the gain is small, BPDR should prioritize input-preparation feedback (re-record guidance, take comparison)
   instead of repair. Needs the notes owner's extractor or a declared surrogate; do not use guide MIDI as performed-pitch truth.
4. Ask the key/notes owners for the handoff contract (note targets or a key grid with confidence). Without it BPDR cannot flag reliably.
5. Before any claim: independent replay of the exact-reference delivery (`v75`-`v77`) and a fresh, untouched evaluation source. Not done.
