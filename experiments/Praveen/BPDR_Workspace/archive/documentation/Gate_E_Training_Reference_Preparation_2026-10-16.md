# Gate E training-reference cache preparation - 16 October 2026

Cache-first preparation has accounted for **all thirty frozen Vocadito training phrases** (twenty recordings / fourteen singers). **Nineteen have authenticated accepted-audio caches; eleven have no matching accepted cache and remain unavailable for final acoustic-mask analysis.** This step deliberately produces no accepted or shifted WAVs and does not substitute raw crops for missing accepted packages. It does not complete final acoustic preparation for all thirty phrases.

On the nineteen available phrases, **seventeen have final multi-interior support candidates**, spanning **fifteen recordings and thirteen singers**. The two measured geometry refusals are vocadito_1_p00 (25 final reliable frames) and vocadito_3_p01 (zero final reliable frames). Their raw human-reference masks contained 224 and 67 frames, respectively; original runtime voicing and PESTO/YIN agreement reduce usable support. These are support refusals after technical acceptance, not technical rejections of their original audio. The old continuous-span selector finds three candidates among the nineteen measured phrases. The earlier 29/30 annotation-only candidates are an upper-bound screening result, not final acoustic eligibility. Eleven unmeasured phrases have unavailable final counts, not zero trusted frames or failed acoustic audits.

**Cache handling matters:** Phase-1 features are measured before PCM24 WAV encoding. They are retained but are not silently reused as measurements of decoded accepted audio. The one historical decoded baseline for vocadito_37_p00 is reused after exact waveform equality and equality with its historically pinned A-input features; its original private reference mask replays exactly. The other eighteen decoded baselines and all nineteen missing original YIN tracks are measured with the unchanged component CPU analyzers. The nineteen final masks preserve both nonpartial human annotator interiors, the existing reference pitch tolerances, confidence at least 0.8, natural unvoiced gaps and the existing 50-cent tracker-agreement limit. No note gaps, partial notes or disagreement frames become labels.

Independent Boolean replay verifies all nineteen masks across **14,148 frame positions**, including exact selected support geometry and the eleven unavailable states. **341 regression tests pass.** All 289 pinned inputs and twenty-eight common-pipeline files remain unchanged. The diagnostic inventory only reads metadata for nontraining files; no development, PP1-evaluation or final-holdout acoustic/reference arrays are opened for this preparation. The original YIN estimates remain measurements, not independent musical ground truth.

**No shifted audio, new paired examples, optimizer updates or model adoption are added.** The engineering pool stays at **936 training variants, 304 development variants and 88 original controls**; the previous 900 comparison updates and 800 diagnostic updates remain historical. Research-ready examples remain zero. Geometry only identifies where an intervention could be attempted; it does not demonstrate that rendered shifts will meet acoustic tolerances, that a model can learn repair, or that musical quality is approved. The earlier frozen-model result remains negative. Wider training coverage is a prerequisite being investigated, not a proven explanation or cure for near-zero model response. Gate E remains open for useful preservation-safe correction, D/J4 review is owner-deferred, and Gate F remains open.

## Every frozen training phrase

| Phrase | State | Human reference frames | Final reliable frames | Multi-interior geometry |
| --- | --- | ---: | ---: | --- |
| vocadito_10_p00 | prepared | 84 | 82 | candidate |
| vocadito_13_p00 | prepared | 164 | 159 | candidate |
| vocadito_13_p01 | prepared | 57 | 57 | candidate |
| vocadito_14_p00 | prepared | 147 | 134 | candidate |
| vocadito_18_p00 | prepared | 180 | 179 | candidate |
| vocadito_18_p01 | unavailable | unavailable | unavailable | unavailable |
| vocadito_1_p00 | prepared | 224 | 25 | refused |
| vocadito_1_p01 | unavailable | unavailable | unavailable | unavailable |
| vocadito_20_p00 | prepared | 39 | 39 | candidate |
| vocadito_25_p00 | prepared | 184 | 143 | candidate |
| vocadito_26_p00 | prepared | 35 | 28 | candidate |
| vocadito_27_p00 | prepared | 138 | 135 | candidate |
| vocadito_28_p00 | unavailable | unavailable | unavailable | unavailable |
| vocadito_29_p00 | prepared | 247 | 247 | candidate |
| vocadito_32_p00 | prepared | 192 | 191 | candidate |
| vocadito_32_p01 | unavailable | unavailable | unavailable | unavailable |
| vocadito_35_p00 | prepared | 185 | 153 | candidate |
| vocadito_36_p00 | prepared | 221 | 211 | candidate |
| vocadito_36_p01 | unavailable | unavailable | unavailable | unavailable |
| vocadito_37_p00 | prepared | 513 | 513 | candidate |
| vocadito_37_p01 | prepared | 507 | 507 | candidate |
| vocadito_38_p00 | unavailable | unavailable | unavailable | unavailable |
| vocadito_38_p01 | unavailable | unavailable | unavailable | unavailable |
| vocadito_39_p00 | prepared | 245 | 243 | candidate |
| vocadito_39_p01 | unavailable | unavailable | unavailable | unavailable |
| vocadito_3_p00 | unavailable | unavailable | unavailable | unavailable |
| vocadito_3_p01 | prepared | 67 | 0 | refused |
| vocadito_6_p00 | prepared | 159 | 151 | candidate |
| vocadito_6_p01 | unavailable | unavailable | unavailable | unavailable |
| vocadito_8_p00 | unavailable | unavailable | unavailable | unavailable |

## Active next step

Gate E training-data prerequisite, continuing Gate B/C preparation: declare original-only technical preparation for the eleven frozen Vocadito training phrases without accepted caches. Use the existing component-owned technical acceptance and frozen CPU PESTO/YIN workflow; preserve the raw crops, all refusals, current reference guards and singer partitions. Reuse the nineteen authenticated results from this step without recalculation. Complete the thirty-phrase final-mask inventory, then predeclare a bounded fixed-support training shift audit before creating any new pairs or fitting. No shared-pipeline execution, development/holdout processing, threshold relaxation or research approval. The 936/304/88 engineering artifacts and existing models stay frozen. Gate E useful learned correction remains open; D/J4 review deferred, F open, research-ready examples zero.

## Evidence

- [declaration.json](../runs/training_reference_preparation/76d30a4d90bf462698019ed7182a2f20/declaration.json)
- [evidence.json](../runs/training_reference_preparation/76d30a4d90bf462698019ed7182a2f20/evidence.json)
- [independent_verification.json](../runs/training_reference_preparation/76d30a4d90bf462698019ed7182a2f20/independent_verification.json)
- [panel_evidence.json](../runs/training_reference_preparation/76d30a4d90bf462698019ed7182a2f20/panel_evidence.json)
