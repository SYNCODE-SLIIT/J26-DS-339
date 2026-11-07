# Fresh controlled confirmation: reference construction failed

Updated 2 November 2026. Engineering feasibility result; no usable-accuracy claim.

## What was actually checked

The V37 reservation fixed twelve files from four source performers before acoustic
inspection: six VocalSet straight vowels (male10/male11), three CSD songs from one
female source, and three JSUT-song recordings from one female source. Publisher
source-category labels are not verified participant demographics. The first three
lexically sorted nonhidden regular JSUT-song WAV entries were selected before
waveform decoding; no outcome-based replacement occurred.

All twelve sources were downloaded into ignored private storage and checked for
archive length/CRC and file hashes. JSUT audio has redistribution restrictions;
it is not a public demo asset. Fixed preprocessing used the highest-energy native
channel, the first at most twenty seconds, mono 22,050 Hz PCM24, with no padding
or normalization. Working-copy identity does not mean identity to the full raw file.

The unchanged V6 reference-construction procedure attempted four local deviations
per file: -30, -20, +20 and +30 cents. **48 attempted, zero scoreable, 48 excluded**,
all with `NO_STABLE_HELD_REGION_WITH_TWO_SIDED_CONTEXT`.
No physical detuned variants were constructed and no correction-Transformer
inference ran. **This is not 0% Transformer accuracy: accuracy is unassessed.**

Independent replay recreated all twelve preprocessings, reinferred 180 original
feature arrays and reproduced all 48 exclusions. Separating adjacent acoustic
regions did not rescue a case: zero joined-region runs and zero separately eligible
regions. That proposed explanation is unsupported for this cohort.

## What prevented a test

The pair builder requires approximately 1.80 seconds of a single stable source
region after allowing for its minimum deviation, context and trimming. None of
the twelve recordings had an acoustic region that long under the fixed acoustic
reliability/segmentation rules. The largest was 1.72 seconds. The source-readiness
plot shows fragmentation and ordinary short melodic movement, rather than a
correction-model prediction failure. No claim that excessive vibrato caused every
exclusion is supported: no region reached the duration needed for that comparison.

![All twelve source readiness profiles](../results/transformer_usability/2026-11-01_v37_fresh_confirmation/preparation/source_readiness.png)

## Consequences

- Retain V35's 29/200 training-side verified outputs and V36's 11/48 known-development
  outputs, with all 50 and 12 original controls unchanged respectively. These remain
  narrow, repeatedly inspected controlled-recovery results.
- V37 does not confirm general usability, natural intended-pitch recovery or listening
  quality. It supplies a failed reference-feasibility result, not an accepted-model result.
- These sources have now been acoustically inspected. If used to design the short-phrase
  protocol they are diagnostic/development material; reserve another cohort for final
  confirmation after the new model and policy are fixed.
- The owner selected **5–15-second melodic hums with short notes**. The successor is
  [the short-phrase implementation plan](Short_Phrase_Transformer_Plan_2026-11-02.md),
  rather than another held-tone training sweep.
- The current Transformer stays inactive; the shared demo/model selection is unchanged.

## Reproducible evidence

Root: `results/transformer_usability/2026-11-01_v37_fresh_confirmation/`.
Pipeline/model/cohort locks and the 210-file component-source closure are retained.
`preparation/integrity_receipt.json` records the independent checks;
`preparation/pairs.json` retains every attempt;
`preparation/source_readiness.json` retains per-file geometry.
Initial preprocessing preflight and verifier-output serialization failures are
retained with amendment receipts; neither supplies a missing successful result.
