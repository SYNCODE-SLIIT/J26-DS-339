# Gate C fixed-support training acoustic audit - 16 October 2026

**The declared fixed-support training acoustic audit is complete.** The twenty-three selected phrases contributed forty-six declared shift keys: **43 measured shifts, 28 passes, 15 measured failures, two identity-blocked keys and one renderer-blocked key**. The seven original geometry-refused phrases remain visible as fourteen additional unattempted keys, so the complete frozen thirty-phrase inventory contains sixty source/direction keys. No case is omitted, retried with a different interval, rendered with a fallback or accepted under relaxed tolerances.

**The twenty-eight passed cases cover eighteen phrases, fourteen recordings and eleven singers.** Eleven negative and seventeen positive shifts pass. Ten phrases pass both directions, covering seven recordings and six singers. Cases are not independent recordings, and these technical counts are not model accuracy or musical approval. The existing support intervals, human/tracker masks, natural pauses, 0.8 voicing confidence threshold and all acoustic thresholds are unchanged.

**Twenty-two unchanged-audio identities pass exactly.** One phrase, **vocadito_37_p00**, fails the declared all-feature identity check, so both shifts are unattempted. Its decoded waveform samples, measured F0, native pitch, confidence, voicing, YIN, timestamps and checkpoint match exactly. Five feature arrays differ slightly: log-mel, onset score, provisional boundary score and the derived 70/71-column arrays. The largest absolute difference is **7.62939453125e-6**, in log-mel/derived feature values; it is not a pitch error in cents. The numerical cause is not established. This run adopts no compatibility tolerance, regenerates no original baseline and changes none of the eighteen historical Vocadito variants already stored in the 936-example pool. A separately declared compatibility decision is needed before another fit.

The **vocadito_13_p01 / -30-cent** renderer attempt is stopped by the unchanged non-finite-or-clipping guard before a shifted WAV or audit target is saved. There is no automatic gain adjustment or alternate renderer. The fifteen measured failures retain their artifacts with **all anchors false and all correction targets zero**. Failure reason occurrences overlap: nine plateau-frame-count failures, four plateau-coverage failures, three fine median-error failures, seven fine p90-error failures and one coarse median-error failure. These counts describe audit conditions, not twenty-four distinct failed shifts.

**358 regression tests pass.** Independent replay verifies all forty-three measured audits and their private targets, independently recomputes frame masks and threshold decisions, and records **430 plateau-frame trace rows**. All forty-three saved shifted WAVs preserve outside-support samples exactly. The twenty-two accepted identities reproduce every original cached acoustic array and YIN track exactly; the separate identity-refusal review verifies the remaining source and both blocked keys. All 461 frozen inputs and twenty-eight common-pipeline files remain unchanged. This audit performs twenty-three identity extractions and forty-three shifted extractions, producing sixty-six component-local FLOAT WAVs. Original accepted packages, reviews and source partitions are preserved. No development, PP1-evaluation or final-holdout acoustic/reference processing occurs in this audit.

**No paired-training examples, optimizer updates or model adoption are added.** Private passed-case audit targets exist, but no new paired catalog is built. The current engineering pool stays 936 training variants, 304 development variants and 88 original controls; the historical comparison/diagnostic update counts stay 900/800. Research-ready examples remain zero. The previously evaluated models still have no demonstrated useful repair benefit; an acoustic audit of artificial shifts does not change that result. Gate C has completed this declared training-audit step, with every refusal and failure retained. Gate E useful preservation-safe learned correction remains open, D/J4 musical review stays owner-deferred and Gate F remains open.

## Every source/direction outcome

| Phrase | Cents | State | Passed | Reasons |
| --- | ---: | --- | --- | --- |
| vocadito_10_p00 | -30 | measured | False | FINE_MEDIAN_ERROR; FINE_P90_ERROR; COARSE_MEDIAN_ERROR |
| vocadito_10_p00 | +30 | measured | False | FINE_P90_ERROR |
| vocadito_13_p00 | -30 | measured | False | FINE_P90_ERROR |
| vocadito_13_p00 | +30 | measured | False | FINE_P90_ERROR |
| vocadito_13_p01 | -30 | execution_refused | False | Rendered audio is non-finite or clips; no automatic normalization |
| vocadito_13_p01 | +30 | measured | True |  |
| vocadito_14_p00 | -30 | measured | True |  |
| vocadito_14_p00 | +30 | measured | False | PLATEAU_FRAME_COUNT; PLATEAU_COVERAGE |
| vocadito_18_p00 | -30 | measured | True |  |
| vocadito_18_p00 | +30 | measured | True |  |
| vocadito_18_p01 | -30 | measured | True |  |
| vocadito_18_p01 | +30 | measured | True |  |
| vocadito_1_p00 | -30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_1_p00 | +30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_1_p01 | -30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_1_p01 | +30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_20_p00 | -30 | measured | True |  |
| vocadito_20_p00 | +30 | measured | True |  |
| vocadito_25_p00 | -30 | measured | False | PLATEAU_FRAME_COUNT; PLATEAU_COVERAGE; FINE_P90_ERROR |
| vocadito_25_p00 | +30 | measured | False | PLATEAU_FRAME_COUNT; PLATEAU_COVERAGE |
| vocadito_26_p00 | -30 | measured | False | PLATEAU_FRAME_COUNT |
| vocadito_26_p00 | +30 | measured | True |  |
| vocadito_27_p00 | -30 | measured | True |  |
| vocadito_27_p00 | +30 | measured | True |  |
| vocadito_28_p00 | -30 | measured | True |  |
| vocadito_28_p00 | +30 | measured | True |  |
| vocadito_29_p00 | -30 | measured | False | PLATEAU_FRAME_COUNT |
| vocadito_29_p00 | +30 | measured | True |  |
| vocadito_32_p00 | -30 | measured | False | PLATEAU_FRAME_COUNT |
| vocadito_32_p00 | +30 | measured | True |  |
| vocadito_32_p01 | -30 | measured | False | PLATEAU_FRAME_COUNT; PLATEAU_COVERAGE |
| vocadito_32_p01 | +30 | measured | True |  |
| vocadito_35_p00 | -30 | measured | False | PLATEAU_FRAME_COUNT |
| vocadito_35_p00 | +30 | measured | False | FINE_MEDIAN_ERROR |
| vocadito_36_p00 | -30 | measured | False | PLATEAU_FRAME_COUNT; FINE_P90_ERROR |
| vocadito_36_p00 | +30 | measured | True |  |
| vocadito_36_p01 | -30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_36_p01 | +30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_37_p00 | -30 | not_attempted_identity_or_source_refused | False | Exact original identity failed; no shifted measurements or tolerance change |
| vocadito_37_p00 | +30 | not_attempted_identity_or_source_refused | False | Exact original identity failed; no shifted measurements or tolerance change |
| vocadito_37_p01 | -30 | measured | True |  |
| vocadito_37_p01 | +30 | measured | True |  |
| vocadito_38_p00 | -30 | measured | False | FINE_MEDIAN_ERROR; FINE_P90_ERROR |
| vocadito_38_p00 | +30 | measured | True |  |
| vocadito_38_p01 | -30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_38_p01 | +30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_39_p00 | -30 | measured | True |  |
| vocadito_39_p00 | +30 | measured | True |  |
| vocadito_39_p01 | -30 | measured | True |  |
| vocadito_39_p01 | +30 | measured | True |  |
| vocadito_3_p00 | -30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_3_p00 | +30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_3_p01 | -30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_3_p01 | +30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_6_p00 | -30 | measured | True |  |
| vocadito_6_p00 | +30 | measured | True |  |
| vocadito_6_p01 | -30 | measured | True |  |
| vocadito_6_p01 | +30 | measured | True |  |
| vocadito_8_p00 | -30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |
| vocadito_8_p00 | +30 | not_attempted_geometry_refused | False | No support under unchanged final reference masks and geometry rules |

## Active next step

Gate E paired-training preparation: authenticate the 28 passed training shifts and freeze a candidate addition list before packaging. Prepare the three existing timing-only boundary corruptions (split, merge, jitter), at most 84 prospective variants, with unchanged acoustic arrays and private audited targets. Check every float32 boundary feature for a real observable change and retain refused variants without fallback, as in development packaging. Keep original source/singer partitions, passed plateau/outside anchors, unlabelled ramps and gaps, and pending review status explicit. Produce separately identified candidate training artifacts; preserve the current 936-example pool, normalizer, models, 304 development variants and 88 original controls. No new audio extraction/rendering, optimizer updates or checkpoint reselection in packaging. Before any later fit, separately resolve the vocadito_37_p00 exact feature-compatibility refusal and declare any training-pool/normalization change. This refusal does not authorize deleting historical artifacts or silently adopting a numerical tolerance. Gate E useful learned repair remains open, D/J4 review deferred, F open and research-ready examples zero.

## Evidence

- [declaration.json](../runs/training_support_audit/c19afe4d0d024d55b63cadf4bbc1b1c8/declaration.json)
- [evidence.json](../runs/training_support_audit/c19afe4d0d024d55b63cadf4bbc1b1c8/evidence.json)
- [independent_verification.json](../runs/training_support_audit/c19afe4d0d024d55b63cadf4bbc1b1c8/independent_verification.json)
- [identity_refusal_review.json](../runs/training_support_audit/c19afe4d0d024d55b63cadf4bbc1b1c8/identity_refusal_review.json)
- [plateau_frame_trace_NOT_MODEL_INPUT.json](../runs/training_support_audit/c19afe4d0d024d55b63cadf4bbc1b1c8/plateau_frame_trace_NOT_MODEL_INPUT.json)
- [panel_evidence.json](../runs/training_support_audit/c19afe4d0d024d55b63cadf4bbc1b1c8/panel_evidence.json)
