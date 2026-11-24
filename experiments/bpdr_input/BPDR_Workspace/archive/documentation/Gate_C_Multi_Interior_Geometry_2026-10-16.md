# Gate C multi-interior support geometry for Gate E - 16 October 2026

The read-only support-design check is complete. **Development: 12/14 phrases have candidate geometry using their unchanged final reliable masks**, compared with 0/14 under the old continuous-reliable-span selector. These 12 phrases cover seven recordings and six singers. All 14 cached final masks replay exactly from human annotations, saved original voiced frames and saved YIN/PESTO agreement; 28 original cache files match their producing-run hashes. The old refusals are retained and no paired development example is added.

**Training: 29/30 phrases have annotation-only candidate geometry**, compared with 3/30 under the old selector on the same annotation grid. These are 20 recordings and 14 singers, using the original laboratory 20 ms annotation-ranking grid. They are not acoustic eligibility results: original voiced/tracker filtering still has to be established for these candidates. The two evidence levels are reported separately; no combined success rate or accuracy is calculated.

The declared alternative places one continuous waveform intervention across multiple trusted annotated interiors. Its plateau must contain at least two distinct A1/A2 note-pair identities and at least 10 trusted frames; at least 10 trusted outside frames remain. Existing 80 ms ramps, 100 ms analysis guards, 300 ms minimum physical plateau and crop-edge guard are preserved. Endpoints are existing frame timestamps. Selection is shortest duration, then largest minimum inside/outside frame count, then earliest start/end. The design was declared before measuring outcomes. Untrusted gaps remain unlabelled; the waveform may cross them, but they never enter the audit denominator or targets. The existing 75% post-intervention coverage requirement and all independent shift/drift limits remain unchanged and unmeasured here.

**All three candidate refusals are retained:** development vocadito_9_p00 has one reliable frame, vocadito_9_p01 has zero; training vocadito_38_p01 has only 10 annotation-trusted frames, insufficient for both 10-frame groups. Every selected development plateau has exactly the minimum 10 trusted frames, so losing even one during the later acoustic audit would fail the minimum-frame condition. No geometry threshold is adjusted after seeing this. This protocol may still fail acoustically.

**297 regression tests pass**, including nine new geometry tests and a separate exhaustive boolean oracle. Independent boolean replay verifies all 41 selected candidate frame counts. The common pipeline remains unchanged (28 pinned files). PP1 evaluation/final holdout arrays were not opened. No new extraction, audio, pair, optimizer update, learned waveform or model adoption occurs. The original model comparison remains 900 updates and prior diagnostics 800; research-ready examples remain zero. Gate C support/audit work remains a prerequisite to Gate E validation; D/J4 review is owner-deferred and Gate F remains open.

## Every phrase retained

| Partition and evidence | Phrase | Old selector geometry | Candidate interval seconds | Trusted plateau / outside frames | Distinct note pairs |
| --- | --- | --- | --- | --- | --- |
| train (annotation only) | vocadito_1_p00 | refused | 16.48–17.22 | 10 / 195 | 2 |
| train (annotation only) | vocadito_1_p01 | refused | 0.86–1.60 | 10 / 169 | 2 |
| development (final-mask replay) | vocadito_2_p00 | refused | 2.78–3.52 | 10 / 234 | 2 |
| development (final-mask replay) | vocadito_2_p01 | refused | 3.26–4.00 | 10 / 266 | 2 |
| train (annotation only) | vocadito_3_p00 | refused | 6.06–6.80 | 10 / 123 | 2 |
| train (annotation only) | vocadito_3_p01 | refused | 0.60–1.36 | 10 / 43 | 2 |
| development (final-mask replay) | vocadito_5_p00 | refused | 4.26–5.22 | 10 / 89 | 2 |
| development (final-mask replay) | vocadito_5_p01 | refused | 4.08–4.82 | 10 / 37 | 2 |
| train (annotation only) | vocadito_6_p00 | refused | 7.92–8.66 | 10 / 148 | 2 |
| train (annotation only) | vocadito_6_p01 | refused | 0.68–1.44 | 10 / 20 | 2 |
| development (final-mask replay) | vocadito_7_p00 | refused | 14.74–15.50 | 10 / 42 | 2 |
| train (annotation only) | vocadito_8_p00 | refused | 0.60–1.34 | 10 / 98 | 2 |
| development (final-mask replay) | vocadito_9_p00 | refused | refused | — | — |
| development (final-mask replay) | vocadito_9_p01 | refused | refused | — | — |
| train (annotation only) | vocadito_10_p00 | refused | 0.62–1.38 | 10 / 73 | 2 |
| train (annotation only) | vocadito_13_p00 | refused | 8.92–9.66 | 10 / 153 | 2 |
| train (annotation only) | vocadito_13_p01 | refused | 0.30–1.08 | 10 / 38 | 2 |
| train (annotation only) | vocadito_14_p00 | refused | 11.22–11.98 | 10 / 127 | 2 |
| development (final-mask replay) | vocadito_16_p00 | refused | 15.62–16.66 | 10 / 27 | 2 |
| train (annotation only) | vocadito_18_p00 | refused | 2.96–3.70 | 10 / 163 | 2 |
| train (annotation only) | vocadito_18_p01 | refused | 0.70–1.46 | 10 / 62 | 2 |
| development (final-mask replay) | vocadito_19_p00 | refused | 12.64–13.40 | 10 / 173 | 2 |
| development (final-mask replay) | vocadito_19_p01 | refused | 0.30–1.06 | 10 / 119 | 2 |
| train (annotation only) | vocadito_20_p00 | refused | 7.38–8.14 | 10 / 26 | 2 |
| development (final-mask replay) | vocadito_21_p00 | refused | 9.84–10.58 | 10 / 153 | 2 |
| development (final-mask replay) | vocadito_21_p01 | refused | 0.10–0.86 | 10 / 15 | 2 |
| train (annotation only) | vocadito_25_p00 | refused | 6.90–7.64 | 10 / 166 | 2 |
| train (annotation only) | vocadito_26_p00 | refused | 2.98–3.86 | 10 / 24 | 2 |
| train (annotation only) | vocadito_27_p00 | refused | 5.34–6.10 | 10 / 126 | 2 |
| train (annotation only) | vocadito_28_p00 | refused | 2.38–3.14 | 10 / 43 | 2 |
| train (annotation only) | vocadito_29_p00 | refused | 11.46–12.20 | 10 / 226 | 2 |
| train (annotation only) | vocadito_32_p00 | refused | 4.54–5.34 | 10 / 179 | 2 |
| train (annotation only) | vocadito_32_p01 | refused | 1.52–2.42 | 10 / 30 | 2 |
| development (final-mask replay) | vocadito_34_p00 | refused | 4.84–5.58 | 10 / 80 | 2 |
| development (final-mask replay) | vocadito_34_p01 | refused | 1.92–2.66 | 10 / 40 | 2 |
| train (annotation only) | vocadito_35_p00 | refused | 5.42–6.16 | 10 / 170 | 2 |
| train (annotation only) | vocadito_36_p00 | refused | 4.78–5.52 | 10 / 197 | 2 |
| train (annotation only) | vocadito_36_p01 | refused | 0.38–2.66 | 10 / 13 | 2 |
| train (annotation only) | vocadito_37_p00 | pass | 4.44–5.18 | 10 / 490 | 2 |
| train (annotation only) | vocadito_37_p01 | pass | 7.32–8.06 | 10 / 490 | 2 |
| train (annotation only) | vocadito_38_p00 | refused | 14.28–15.18 | 10 / 105 | 2 |
| train (annotation only) | vocadito_38_p01 | refused | refused | — | — |
| train (annotation only) | vocadito_39_p00 | pass | 3.86–4.60 | 10 / 231 | 2 |
| train (annotation only) | vocadito_39_p01 | refused | 5.10–5.84 | 10 / 120 | 2 |

## Validation assessment: share with caveats

This is complete enumeration of the declared endpoint grid, not proof over every possible continuous-time interval. Old-selector pass counts measure support selection, not the full acoustic audit. Candidate counts are necessary geometry only; no after-shift measurements exist. Both annotator identities define a note pair: two identities do not imply two long continuous reliable spans. Fragmentation within the same note pair does not satisfy the multi-interior requirement. The full original annotation/tracker masks are copied only into clearly private diagnostic archives and do not become runtime inputs.

No training source is relabelled or promoted. All 30 training phrases are consistently evaluated at annotation level even though one prior successful phrase has a saved acoustic cache. No unavailable tracker results are inferred. The two development refusals belong to the same recording; retaining them keeps all eight development recordings visible. A candidate development support only demonstrates that the old continuous-span refusal can sometimes be addressed without relaxing trusted-frame definitions; it cannot explain or solve the separate model learning failure.

Run: `runs/support_geometry/84240d57a663460ea7f99903888fc00d`. `declaration.json` locks the design and input hashes; `evidence.json` records all 44 phrases; `independent_verification.json` replays counts; `historical_cache_verification.json` checks producing-run cache hashes; `panel_evidence.json` pins the figure and 297-test log. Original pipeline, annotations, masks, reviews, saved fits and failed Jamendo crop remain intact.

## Next — Gate C audit prerequisite, Gate E open

Gate C support/audit prerequisite for Gate E: declare a bounded acoustic pilot on the locked multi-interior supports before any generation. Use two development phrases from distinct recordings/singers, deterministically first by phrase ID (vocadito_2_p00 and vocadito_5_p00), with -30/+30-cent injections: four attempted shifts, no fallback support search. Reuse the existing renderer and independent YIN/PESTO audits with every threshold, original mask and refusal retained. Measure plateau coverage, actual shift error, outside drift and exact outside samples; geometry does not establish these. This is a support-protocol engineering pilot, not model validation or fitting. Training annotation candidates require their own final acoustic masks before interventions. No new training, loss/gain search or holdout access. D/J4 review deferred, Gate E and F open.
