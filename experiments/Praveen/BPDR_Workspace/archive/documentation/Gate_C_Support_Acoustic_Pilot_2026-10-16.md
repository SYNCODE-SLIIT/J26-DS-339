# Gate C fixed-support acoustic pilot for Gate E - 16 October 2026

The declared **four-shift acoustic pilot is complete: two passed, two failed**. Both original-audio identities reproduce every saved acoustic array and saved YIN track exactly. The two phrases use their already fixed supports and unchanged final trusted masks; no alternate interval, renderer or threshold was tried.

**Passed:** vocadito_2_p00 at -30 cents and vocadito_5_p00 at +30 cents. Their independent YIN median / 90th-percentile shift errors are 1.440 / 3.381 cents and 0.896 / 2.593 cents respectively. These are errors in known synthetic shift delivery, not errors of a learned repair model.

**Failed and retained:** vocadito_2_p00 at +30 cents fails the YIN 90th-percentile error limit: 1202.733 cents versus 10 allowed. Three plateau frames at 3.30–3.34 seconds show an octave-sized YIN estimate jump while PESTO reports about +33.334 cents. This estimator disagreement does not establish which pitch is physically correct; no octave folding or label repair is applied. vocadito_5_p00 at -30 cents keeps only 9/10 valid plateau frames; PESTO confidence is 0.422 at 5.04 seconds, below the unchanged 0.8 voiced threshold. Its fine-shift errors are small, but it still fails the required minimum of ten common frames. Both failed attempts have all-false anchor masks and zero targets.

All four attempts preserve sample count, float-WAV roundtrips and every outside-support sample exactly. Outside audit coverage is 100% and median/90th-percentile measured pitch drift is zero in all four attempts. Plateau coverage is 100%, 100%, 90%, 100% in declared order. Neither source passes both shift directions, so this is a **partial support-protocol result**, not a passed corpus gate or accuracy percentage.

**306 regression tests pass**. Independent replay verifies all four original audits, separate threshold decisions, private targets, exact outside samples and 40 plateau frame records. The first launch stopped before shift measurement because the historical runs/coverage directory shadowed an optional library. Its evidence/log remain preserved; Python safe-path launch (-P) resolved the import without changing packages, source algorithms or audit thresholds.

This step generates two identity controls and four synthetic test WAVs plus diagnostic measurements. It creates **zero paired examples and zero optimizer updates**. The engineering pool remains 936 training variants / 285 Jamendo-only development variants. The original comparison remains 900 updates and earlier diagnostics 800; no learned repair audio, new fitting, shared-pipeline execution or policy adoption occurs. The 28 pinned common-pipeline files and all directly used original artifacts remain unchanged. PP1/final holdout acoustics remain reserved. Research-ready examples remain zero; Gate C support/audit work and Gate E validation remain open, D/J4 review deferred, Gate F open.

## All four declared attempts

| Phrase | Injected cents | Common / trusted plateau frames | Fine median error | Fine p90 error | Coarse p90 error | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| vocadito_2_p00 | -30 | 10 / 10 | 1.439558 | 3.381345 | 30.666561 | pass |
| vocadito_2_p00 | +30 | 10 / 10 | 1.882918 | 1202.732767 | 5.999902 | FINE_P90_ERROR |
| vocadito_5_p00 | -30 | 9 / 10 | 0.540590 | 0.966312 | 3.333752 | PLATEAU_FRAME_COUNT |
| vocadito_5_p00 | +30 | 10 / 10 | 0.896105 | 2.592661 | 3.332960 | pass |

## Validation assessment: share with caveats

The declaration precedes identity extraction and rendering. Fixed intervals are 2.78–3.52 seconds for vocadito_2_p00 and 4.26–5.22 seconds for vocadito_5_p00, in crop time. The named pair from the prior plan is authoritative; selection follows original numeric manifest order with distinct recording/singer IDs. The prior prose saying first by phrase ID does not authorize lexical re-selection. Sources remain in development; they are never fed to training.

Every existing pitch-audit limit, 80 ms ramp, 100 ms analysis guard, 300 ms minimum physical plateau, 75% coverage criterion and ten-frame minimum remains fixed. Original trusted masks are replayed from human annotations and saved original voicing/tracker agreement. The exact identity check compares all saved feature arrays with equal-NaN handling for unvoiced F0, not an approximate tolerance. Model checkpoint hashes are pinned; PESTO remains frozen CPU inference. YIN and PESTO are estimates, not human verification.

The four independent checks replay the original audit implementation exactly, then separately verify frame counts, shift statistics, decision thresholds, waveform support and target suppression. The failed +30-cent case is not repaired by removing its three octave-jump frames. The nine-frame case is not promoted because 90% coverage exceeds 75%: both coverage and absolute minimum count must pass. A pilot pass does not create a corpus entry or certify perceptual quality. Confidence, audio leakage/musical review and model repair benefit remain separate concerns.

All four attempted directions and both failures remain visible. Two successes are not a 50% repair-accuracy result. The two original development geometry refusals remain unchanged in the preceding 14-phrase evidence. The six playable files are identity controls and synthetic corruptions, never learned repair outputs.

Run: `runs/support_acoustic_pilot/f71cccdef29941edbdc8d2e449412114`. `declaration.json`, `evidence.json`, `independent_verification.json`, `plateau_frame_trace_NOT_MODEL_INPUT.json` and `launch_recovery.json` preserve protocol, raw audits, per-frame diagnoses and the unmeasured failed launch. The old 936/285 prepared examples and previous fitted checkpoints are retained, without silently adding these pilot measurements to selection metrics.

## Next — Gate C prerequisite for Gate E

Gate C prerequisite for Gate E: separately declare completion of the same -30/+30-cent audit on the remaining ten development phrases with locked multi-interior geometry. Reuse these four measured pilot attempts; do not rerun them or search alternate supports/renderers/thresholds. Keep the two geometry-refused development phrases, every failed direction and all 24 prospective shift outcomes visible. Audit completion must precede any paired-view packaging or model fit. Training candidates still require final acoustic masks; main comparison and diagnostic fits are unchanged. No musical approval, model adoption or repair-accuracy claim. Gate E open, D/J4 deferred and Gate F open.
