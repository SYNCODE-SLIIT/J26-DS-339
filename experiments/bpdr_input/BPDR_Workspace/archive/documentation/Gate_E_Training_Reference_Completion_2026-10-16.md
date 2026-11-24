# Gate E original-training reference completion - 16 October 2026

**Original reference preparation is now complete for all thirty frozen Vocadito training phrases**, covering twenty recordings and fourteen singers. All eleven previously missing original-audio packages pass the existing technical acceptance checks. Their decoded accepted-audio features and final human-reference masks are now available; the nineteen earlier rows and saved arrays remain unchanged. There are **no missing accepted caches and no technical refusals** in this thirty-phrase inventory. Technical acceptance does not establish musical suitability.

**Twenty-three phrases have final multi-interior support candidates**, covering **seventeen recordings and thirteen singers**. **Seven phrases fail support geometry**: vocadito_1_p00, vocadito_1_p01, vocadito_36_p01, vocadito_38_p01, vocadito_3_p00, vocadito_3_p01 and vocadito_8_p00. All seven are retained with their actual measured masks; they are technically accepted recordings without enough trusted intervention support. The existing continuous-span selector still has three candidates. The earlier 29/30 annotation-only screening count is not final acoustic eligibility.

This completion adds **eleven accepted working WAVs and eleven byte-identical retained source copies**, not shifted or corrected recordings. The unchanged Phase-1 path measures eleven pre-encoding feature sets; final masks separately use eleven decoded-WAV feature sets and eleven original YIN tracks. These are twenty-two PESTO extraction calls and eleven YIN measurements across the interrupted and resumed attempts. The nineteen earlier decoded baselines/YIN tracks are not recalculated. All human-reference guards, confidence at least 0.8, natural unvoiced gaps and the existing 50-cent tracker agreement rule remain unchanged.

An implementation check caught a floating-point duration rounding issue on vocadito_32_p01: 226,409 original samples at 44,100 Hz correctly become 113,204 samples at 22,050 Hz under the unchanged pipeline. Multiplying its stored floating-point duration back by the target rate incorrectly rounded to 113,205. The new completion validator derives the expected count once from exact source sample counts, preserving the existing round-to-even behavior. It does not trim audio, loosen timeline checks or change the shared or component technical pipeline. The interrupted attempt, its three completed measurements and four accepted packages are preserved; the resumed run reuses all of them.

**351 regression tests pass**, including the exact half-sample rounding case and source/partition/refusal guards. Independent verification replays the eleven new masks over **6,217 frame positions**, verifies all nineteen earlier rows exactly and authenticates their 57 saved artifacts. Together with the prior nineteen-mask replay, this accounts for **20,365 frame positions** across all thirty originals. All 389 frozen inputs and twenty-eight common-pipeline files remain unchanged. No development, PP1-evaluation or final-holdout acoustic/reference arrays are processed in this completion.

**The next training acoustic audit is declared but not executed:** every one of the twenty-three candidates has a frozen support and two keys, -30 and +30 cents, for forty-six keys total. One exact zero-shift identity per phrase and the existing realized-shift audit rules remain mandatory. Geometry does not predict that all or any of these shifts will pass. Failed keys remain visible and cannot supply training targets.

**No shifted WAVs, new paired examples, optimizer updates or model adoption are added.** The engineering pool remains 936 training variants, 304 development variants and 88 original controls, with historical 900 comparison updates and 800 diagnostic updates. Research-ready examples remain zero. The previous frozen-model result still demonstrates no useful repair benefit. Completing broader reference coverage resolves a data-preparation gap; it does not prove a cause or remedy for near-zero learned response. Gate E remains open, D/J4 musical review stays owner-deferred and Gate F delivered repair remains open.

## Complete frozen training inventory

| Phrase | Final reliable frames | Multi-interior support | Original result source |
| --- | ---: | --- | --- |
| vocadito_10_p00 | 82 | candidate | earlier result reused |
| vocadito_13_p00 | 159 | candidate | earlier result reused |
| vocadito_13_p01 | 57 | candidate | earlier result reused |
| vocadito_14_p00 | 134 | candidate | earlier result reused |
| vocadito_18_p00 | 179 | candidate | earlier result reused |
| vocadito_18_p01 | 81 | candidate | completed original-only preparation |
| vocadito_1_p00 | 25 | refused | earlier result reused |
| vocadito_1_p01 | 16 | refused | completed original-only preparation |
| vocadito_20_p00 | 39 | candidate | earlier result reused |
| vocadito_25_p00 | 143 | candidate | earlier result reused |
| vocadito_26_p00 | 28 | candidate | earlier result reused |
| vocadito_27_p00 | 135 | candidate | earlier result reused |
| vocadito_28_p00 | 37 | candidate | completed original-only preparation |
| vocadito_29_p00 | 247 | candidate | earlier result reused |
| vocadito_32_p00 | 191 | candidate | earlier result reused |
| vocadito_32_p01 | 41 | candidate | completed original-only preparation |
| vocadito_35_p00 | 153 | candidate | earlier result reused |
| vocadito_36_p00 | 211 | candidate | earlier result reused |
| vocadito_36_p01 | 25 | refused | completed original-only preparation |
| vocadito_37_p00 | 513 | candidate | earlier result reused |
| vocadito_37_p01 | 507 | candidate | earlier result reused |
| vocadito_38_p00 | 121 | candidate | completed original-only preparation |
| vocadito_38_p01 | 10 | refused | completed original-only preparation |
| vocadito_39_p00 | 243 | candidate | earlier result reused |
| vocadito_39_p01 | 133 | candidate | completed original-only preparation |
| vocadito_3_p00 | 2 | refused | completed original-only preparation |
| vocadito_3_p01 | 0 | refused | earlier result reused |
| vocadito_6_p00 | 151 | candidate | earlier result reused |
| vocadito_6_p01 | 34 | candidate | completed original-only preparation |
| vocadito_8_p00 | 0 | refused | completed original-only preparation |

## Active next step

Gate C acoustic prerequisite for Gate E: execute the saved fixed-support training audit protocol on all 23 final-geometry candidates, in phrase-ID order, at -30 and +30 cents (46 keys). Run one exact zero-shift identity per phrase, reuse authenticated original audio/features/YIN/masks, and apply the unchanged renderer and both-tracker plateau/outside audit rules. Keep the frozen support for both directions; retain failed or unattempted keys without usable targets, fallback rendering or threshold relaxation. This audit creates no paired-training catalog and performs no model fitting; package only passed cases in a later declared step. Preserve current 936/304/88 training/development/original-control artifacts, singer partitions and all model weights. No shared-pipeline execution or development/holdout processing. Gate E useful learned repair remains open, D/J4 review deferred, F open and research-ready examples zero.

## Evidence

- [declaration.json](../runs/training_reference_completion/e638b73ea869486ea7dd7d53bfd8455c/declaration.json)
- [evidence.json](../runs/training_reference_completion/e638b73ea869486ea7dd7d53bfd8455c/evidence.json)
- [independent_verification.json](../runs/training_reference_completion/e638b73ea869486ea7dd7d53bfd8455c/independent_verification.json)
- [next_training_audit_protocol.json](../runs/training_reference_completion/e638b73ea869486ea7dd7d53bfd8455c/next_training_audit_protocol.json)
- [panel_evidence.json](../runs/training_reference_completion/e638b73ea869486ea7dd7d53bfd8455c/panel_evidence.json)
