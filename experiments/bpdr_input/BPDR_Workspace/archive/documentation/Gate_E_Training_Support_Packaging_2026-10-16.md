# Gate E candidate training-support packaging - 16 October 2026

**Candidate training packaging is complete: 77 valid variants from the 28 passed acoustic shift cases**, covering **eighteen phrases, fourteen recordings and eleven singers**. The fixed recipe proposed eighty-four variants: split, merge and jitter for each passed shift. **Seven merge variants are refused because the stored float32 Gaussian boundary feature does not change.** There are no boundary-construction refusals or replacement corruptions. Actual candidate counts are **28 split, 21 merge and 28 jitter**, with thirty negative-shift and forty-seven positive-shift variants. Variants are repeated controlled views, not independent recordings.

The refused merges are vocadito_13_p01 at +30 cents and vocadito_28_p00, vocadito_6_p00 and vocadito_6_p01 at both -30 and +30 cents. Removing the nominal central edge changes timing metadata but not the sampled max-Gaussian feature in these cases; metadata differences alone are insufficient. All eighty-four packaging outcomes are recorded, and all sixty original acoustic-audit keys remain in the full inventory, including the thirty-two nonpassing or unattempted keys. The fifteen measured acoustic failures, two identity-blocked cases, one renderer-blocked case and fourteen geometry-refused keys supply no new training candidates.

**Four-view contracts are verified exactly.** A/B share the accepted original WAV and identical original seventy-column acoustics; C/D share the saved shifted WAV and identical shifted acoustics. A/C use the original timing-only Gaussian boundary channel; B/D use the corresponding observable corruption. All source timestamps and acoustic columns are unchanged, and no human pitch label, private target or reference mask enters model input archives. Boundary metadata uses controlled human A1 timings in this experiment; it is not a runtime note-inference or component-integration claim.

Private targets retain the exact passed-audit anchor mask, negative injected cents on the audited plateau and zero preservation anchors outside support. The adapter adds required loader field names without changing any source mask, timestamp or correction value. Ramps, unvoiced gaps and other unaudited frames remain unlabelled. All seventy-seven candidate records remain **pending review, exploratory and training_eligible=False**. The engineering loader can read all seventy-seven; each is refused by the research gate. Normalized 70/71-column batches are finite using the unchanged original training normalizer; no normalizer is fitted or replaced.

**The active engineering pool remains 936 examples.** A separate **candidate-only combined catalog contains 1,013 records: the exact earlier 936 plus 77 additions**. Its Vocadito portion would contain ninety-five variants across nineteen phrases, fourteen recordings and eleven singers; the 918 existing Jamendo variants are retained unchanged. This file is explicitly candidate_not_adopted with fit_authorized=False. No active training/comparison pointer, model checkpoint or selected update changes. The 304 current development variants and all 88 original controls remain unchanged.

**365 regression tests pass.** Independent verification reproduces all seventy-seven complete four-view packages and private targets, checks **54,462 per-variant source frame positions**, replays all eighty-four boundary decisions and confirms the first 936 proposed combined records equal the saved legacy catalog exactly. It confirms no training/development/held-out recording or singer overlap for the new human-reference candidates. All 2,252 pinned inputs and twenty-eight common-pipeline files remain unchanged. This step performs **zero audio writes, acoustic extractions, audit attempts, model forwards or optimizer updates**.

The earlier vocadito_37_p00 identity refusal remains unresolved: its waveform, pitch and YIN are exact, while five feature arrays differ slightly. None of its blocked new shift cases is packaged, and its eighteen historical training variants are preserved without a fresh compatibility approval. Before a later fit, a separate saved-array diagnostic and versioned reuse/quarantine decision must resolve that legacy issue, followed by an explicit training-pool/normalization/fit declaration. The proposed combined catalog must not silently become the active fit input.

**No model improvement or musical accuracy is claimed.** The earlier frozen-model evaluation remains negative, and the historical comparison/diagnostic update counts stay 900/800. Research-ready examples remain zero. Gate E candidate packaging is completed; useful preservation-safe learned correction remains unproven. D/J4 musical review remains owner-deferred and Gate F delivered repair remains open.

## All planned boundary variants

| Phrase | Cents | Corruption | State | Changed boundary frames |
| --- | ---: | --- | --- | ---: |
| vocadito_13_p01 | +30 | split | packaged | 8 |
| vocadito_13_p01 | +30 | merge | refused_unobservable_boundary | 0 |
| vocadito_13_p01 | +30 | jitter | packaged | 169 |
| vocadito_14_p00 | -30 | split | packaged | 15 |
| vocadito_14_p00 | -30 | merge | packaged | 4 |
| vocadito_14_p00 | -30 | jitter | packaged | 598 |
| vocadito_18_p00 | -30 | split | packaged | 9 |
| vocadito_18_p00 | -30 | merge | packaged | 9 |
| vocadito_18_p00 | -30 | jitter | packaged | 708 |
| vocadito_18_p00 | +30 | split | packaged | 9 |
| vocadito_18_p00 | +30 | merge | packaged | 9 |
| vocadito_18_p00 | +30 | jitter | packaged | 708 |
| vocadito_18_p01 | -30 | split | packaged | 12 |
| vocadito_18_p01 | -30 | merge | packaged | 4 |
| vocadito_18_p01 | -30 | jitter | packaged | 312 |
| vocadito_18_p01 | +30 | split | packaged | 12 |
| vocadito_18_p01 | +30 | merge | packaged | 4 |
| vocadito_18_p01 | +30 | jitter | packaged | 312 |
| vocadito_20_p00 | -30 | split | packaged | 9 |
| vocadito_20_p00 | -30 | merge | packaged | 8 |
| vocadito_20_p00 | -30 | jitter | packaged | 414 |
| vocadito_20_p00 | +30 | split | packaged | 9 |
| vocadito_20_p00 | +30 | merge | packaged | 8 |
| vocadito_20_p00 | +30 | jitter | packaged | 414 |
| vocadito_26_p00 | +30 | split | packaged | 8 |
| vocadito_26_p00 | +30 | merge | packaged | 1 |
| vocadito_26_p00 | +30 | jitter | packaged | 632 |
| vocadito_27_p00 | -30 | split | packaged | 10 |
| vocadito_27_p00 | -30 | merge | packaged | 10 |
| vocadito_27_p00 | -30 | jitter | packaged | 675 |
| vocadito_27_p00 | +30 | split | packaged | 10 |
| vocadito_27_p00 | +30 | merge | packaged | 10 |
| vocadito_27_p00 | +30 | jitter | packaged | 675 |
| vocadito_28_p00 | -30 | split | packaged | 10 |
| vocadito_28_p00 | -30 | merge | refused_unobservable_boundary | 0 |
| vocadito_28_p00 | -30 | jitter | packaged | 641 |
| vocadito_28_p00 | +30 | split | packaged | 10 |
| vocadito_28_p00 | +30 | merge | refused_unobservable_boundary | 0 |
| vocadito_28_p00 | +30 | jitter | packaged | 641 |
| vocadito_29_p00 | +30 | split | packaged | 13 |
| vocadito_29_p00 | +30 | merge | packaged | 4 |
| vocadito_29_p00 | +30 | jitter | packaged | 745 |
| vocadito_32_p00 | +30 | split | packaged | 11 |
| vocadito_32_p00 | +30 | merge | packaged | 14 |
| vocadito_32_p00 | +30 | jitter | packaged | 995 |
| vocadito_32_p01 | +30 | split | packaged | 11 |
| vocadito_32_p01 | +30 | merge | packaged | 14 |
| vocadito_32_p01 | +30 | jitter | packaged | 179 |
| vocadito_36_p00 | +30 | split | packaged | 8 |
| vocadito_36_p00 | +30 | merge | packaged | 7 |
| vocadito_36_p00 | +30 | jitter | packaged | 998 |
| vocadito_37_p01 | -30 | split | packaged | 22 |
| vocadito_37_p01 | -30 | merge | packaged | 28 |
| vocadito_37_p01 | -30 | jitter | packaged | 906 |
| vocadito_37_p01 | +30 | split | packaged | 22 |
| vocadito_37_p01 | +30 | merge | packaged | 28 |
| vocadito_37_p01 | +30 | jitter | packaged | 906 |
| vocadito_38_p00 | +30 | split | packaged | 4 |
| vocadito_38_p00 | +30 | merge | packaged | 5 |
| vocadito_38_p00 | +30 | jitter | packaged | 954 |
| vocadito_39_p00 | -30 | split | packaged | 11 |
| vocadito_39_p00 | -30 | merge | packaged | 43 |
| vocadito_39_p00 | -30 | jitter | packaged | 952 |
| vocadito_39_p00 | +30 | split | packaged | 11 |
| vocadito_39_p00 | +30 | merge | packaged | 43 |
| vocadito_39_p00 | +30 | jitter | packaged | 952 |
| vocadito_39_p01 | -30 | split | packaged | 9 |
| vocadito_39_p01 | -30 | merge | packaged | 8 |
| vocadito_39_p01 | -30 | jitter | packaged | 550 |
| vocadito_39_p01 | +30 | split | packaged | 9 |
| vocadito_39_p01 | +30 | merge | packaged | 8 |
| vocadito_39_p01 | +30 | jitter | packaged | 550 |
| vocadito_6_p00 | -30 | split | packaged | 9 |
| vocadito_6_p00 | -30 | merge | refused_unobservable_boundary | 0 |
| vocadito_6_p00 | -30 | jitter | packaged | 951 |
| vocadito_6_p00 | +30 | split | packaged | 9 |
| vocadito_6_p00 | +30 | merge | refused_unobservable_boundary | 0 |
| vocadito_6_p00 | +30 | jitter | packaged | 951 |
| vocadito_6_p01 | -30 | split | packaged | 13 |
| vocadito_6_p01 | -30 | merge | refused_unobservable_boundary | 0 |
| vocadito_6_p01 | -30 | jitter | packaged | 225 |
| vocadito_6_p01 | +30 | split | packaged | 13 |
| vocadito_6_p01 | +30 | merge | refused_unobservable_boundary | 0 |
| vocadito_6_p01 | +30 | jitter | packaged | 225 |

## Active next step

Gate E experiment preparation: complete a bounded saved-array compatibility diagnostic for vocadito_37_p00 using its recorded baseline and identity arrays, unchanged feature contracts and the existing normalizer. Quantify numerical differences without re-extraction, rewriting old arrays, changing acoustic thresholds or retroactively promoting the two blocked shift keys. Document a versioned reuse/quarantine decision for its eighteen legacy training variants, then lock the separate candidate training catalog and training-only normalization/sampling/matched-fit/development protocol before another fit. Keep the active 936-example pool, current normalizer, checkpoints, 304 development variants and 88 original controls unchanged during this preparation; no optimizer updates or shared-pipeline execution. Candidate packaging is complete, but its proposed 1,013-example combined catalog is not adopted. Gate E useful preservation-safe learned correction remains open; D/J4 review deferred, F open and research-ready examples zero.

## Evidence

- [declaration.json](../runs/training_support_packaging/e0044ddc5a3d414096a1bfc0e671a918/declaration.json)
- [evidence.json](../runs/training_support_packaging/e0044ddc5a3d414096a1bfc0e671a918/evidence.json)
- [independent_verification.json](../runs/training_support_packaging/e0044ddc5a3d414096a1bfc0e671a918/independent_verification.json)
- [packaging_outcomes.json](../runs/training_support_packaging/e0044ddc5a3d414096a1bfc0e671a918/packaging_outcomes.json)
- [candidate_combined_catalog.json](../runs/training_support_packaging/e0044ddc5a3d414096a1bfc0e671a918/candidate_combined_catalog.json)
- [loader_checks.json](../runs/training_support_packaging/e0044ddc5a3d414096a1bfc0e671a918/loader_checks.json)
- [panel_evidence.json](../runs/training_support_packaging/e0044ddc5a3d414096a1bfc0e671a918/panel_evidence.json)
