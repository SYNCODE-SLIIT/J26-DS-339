# Gate E human-reference development packaging - 16 October 2026

Gate E development packaging is complete with **19 usable Vocadito variants, not the planned 21**. Seven previously passed shift cases were each considered for split, merge and jitter boundary corruptions. All seven split and seven jitter variants are observable; five merge variants are observable. **The two merge variants for vocadito_5_p01 (-30 and +30 cents) are refused** because removing the predeclared central timing edge leaves the sampled float32 Gaussian boundary feature exactly unchanged. The existing four-view validator requires an observable contrast; its rule, Gaussian sigma 0.04 seconds, seed 339 and corruption functions are unchanged. No alternative edge or tolerance was searched.

The first attempt stopped at this existing safeguard after writing ten valid partial pair manifests. That interrupted run, its declaration and log remain preserved and are excluded from the current catalog. A new declaration explicitly retains all 21 candidate packaging outcomes: 19 packaged and two refused. Both original acoustic directions for vocadito_5_p01 remain available through split and jitter variants; the merge refusal does not rewrite its passed acoustic audits.

The combined development catalog now has **304 variants: 19 human-reference Vocadito + 285 unchanged model-estimated Jamendo variants**. The new Vocadito data represent seven shifted cases, six phrases, three recordings and three singers, not nineteen independent recordings. Only two of the seven shift cases are negative injections; five are positive. The 19 variants are five negative and fourteen positive, with split/merge/jitter counts 7/5/7. All 28 acoustic audit keys remain recorded, including seventeen measured failures and four unattempted geometry keys. Those excluded keys do not enter paired targets.

Every A/B pair reuses one original waveform; every C/D pair reuses one saved synthetic waveform. The first 70 acoustic feature columns and timestamps match their saved sources exactly. Only timing-derived column 71 differs between nominal and corrupted views; pitch references never enter the input archives. Private anchor masks and correction values match the passed-audit files exactly; the adapter adds loader field names without filling unlabelled frames or changing target signs. The 70-feature control retains exact A=B and C=D inputs.

**326 regression tests pass**. Independent checks validate all 76 new input archives, nineteen boundary/target/normalized-input replays and both zero-change merge refusals. Each new record is refused by the engineering training selector and research training gate; whole-package training loaders and a development normalizer fit are also refused. The 936 training examples and saved training-only normalizer are unchanged. All 88 original no-change development controls are reused exactly (14 Vocadito, 74 Jamendo); their original source-status metadata remains historical.

This step performs **zero audio generation, acoustic extraction, new shift audits, model forward passes or optimizer updates**. The shared pipeline remains unchanged. The original 900-update comparison and 800 diagnostic updates remain historical; their previous 285-example Jamendo-only evaluation is not rewritten. The new combined catalog has not been evaluated by a model yet. These are exploratory development examples with pending musical review, training_eligible=False and research-ready examples zero. Gate E remains open, D/J4 review owner-deferred, Gate F open. The next step is a separately declared frozen-model development evaluation.

## All 21 planned variants

| Phrase | Injected cents | Corruption | Changed boundary frames | Outcome |
| --- | ---: | --- | ---: | --- |
| vocadito_2_p00 | -30 | split | 30 | packaged |
| vocadito_2_p00 | -30 | merge | 5 | packaged |
| vocadito_2_p00 | -30 | jitter | 758 | packaged |
| vocadito_2_p01 | +30 | split | 17 | packaged |
| vocadito_2_p01 | +30 | merge | 8 | packaged |
| vocadito_2_p01 | +30 | jitter | 870 | packaged |
| vocadito_5_p00 | +30 | split | 5 | packaged |
| vocadito_5_p00 | +30 | merge | 7 | packaged |
| vocadito_5_p00 | +30 | jitter | 873 | packaged |
| vocadito_5_p01 | -30 | split | 11 | packaged |
| vocadito_5_p01 | -30 | merge | 0 | refused_unobservable_boundary |
| vocadito_5_p01 | -30 | jitter | 290 | packaged |
| vocadito_5_p01 | +30 | split | 11 | packaged |
| vocadito_5_p01 | +30 | merge | 0 | refused_unobservable_boundary |
| vocadito_5_p01 | +30 | jitter | 290 | packaged |
| vocadito_19_p00 | +30 | split | 11 | packaged |
| vocadito_19_p00 | +30 | merge | 14 | packaged |
| vocadito_19_p00 | +30 | jitter | 935 | packaged |
| vocadito_19_p01 | +30 | split | 16 | packaged |
| vocadito_19_p01 | +30 | merge | 16 | packaged |
| vocadito_19_p01 | +30 | jitter | 510 | packaged |

## Validation assessment: share with caveats

The actual result is nineteen usable variants, not twenty-one. The two refusals are feature observability failures, distinct from the earlier seventeen failed acoustic audits and four unattempted geometry keys. All three refusal classes remain visible and cannot be combined as model errors. The exact float32 boundary-change test matches the established model-input representation; it is not a new perceptual-quality threshold. The five new observability tests include a distinct-edge Gaussian example that produces identical sampled features.

The seven passed targets are adapted to the existing private target archive schema, retaining every anchor/time/value exactly. No target or human F0 enters the three-key model input archives. The first seventy features come from saved original/detuned acoustics; Gaussian timing information is the only boundary augmentation. Original human annotations are reference evidence, not proof of subjective musical acceptability. Jamendo references remain model estimates and artist IDs remain singer proxies. The new development subset has only three singer groups and uneven direction/corruption coverage.

The complete combined catalog reuses all 285 original Jamendo record dictionaries without edits and adds nineteen development-only records. The original no-change catalog is copied byte-for-byte with all 88 existing controls. Its paired_source_status values describe the original preparation run; they are not current packaging decisions. The saved training normalizer is loaded only for finite/alignment checks and never refitted. Training loader checks refer to the existing guarded entry points; low-level record reading is used only after development validation for preparation checks. No claim is made that arbitrary direct tensor use is a safe training interface.

The future 304-example evaluation is a post-selection development diagnostic: the three models were historically selected using the old Jamendo-only development pool. New human-reference coverage permits additional diagnosis, not an independent held-out accuracy claim or checkpoint reselection. PP1 and final holdouts remain reserved; model adoption, musical review and delivered-waveform validation are still open.

Run: `runs/support_development_packaging/779ea64608384daa9014865187c365b4`. `declaration.json` locks inputs, all 21 planned outcomes and the interruption receipt; `summary.json` contains nineteen new records; `combined_catalog.json` contains 304 records; `packaging_outcomes.json` retains the two refused merges; `full_audit_inventory.json` preserves all 28 prior acoustic keys; `loader_checks.json` and `independent_verification.json` verify features, private targets, source preservation and loader gates. The interrupted ten-record partial run is preserved outside the published catalog.

## Next — Gate E frozen-model evaluation

Gate E frozen-model development evaluation: separately declare evaluation of the three historically selected model checkpoints (augmentation-only update 25, frame-only update 50, BPDR update 300), using their verified Windows CPU exports and the unchanged training-only normalizer. Compare each with zero correction on the explicit 304-pair development catalog (19 Vocadito / 285 Jamendo) and all 88 retained no-change originals. Report both datasets separately and recording/singer-or-artist-proxy-aware macros, direction/corruption coverage, preservation, boundary sensitivity and actual counter-detuning response. This is post-selection exploratory development diagnosis: no checkpoint reselection, fitting, gain/loss search, rendering or production adoption. Keep all audit and boundary refusals visible; reserve PP1/final holdouts. Human review remains deferred (D/J4), Gate E and F open, research-ready examples zero.
