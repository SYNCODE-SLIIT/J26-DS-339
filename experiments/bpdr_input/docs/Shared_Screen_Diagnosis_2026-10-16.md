# Training-crop screen diagnosis and export handoff — 16 October 2026

**Current gates: B/C — source support and training readiness, open.** The next local preparation step is complete. The common pipeline remains unchanged; no new audio, intervention pair, fit, threshold change or musical approval is produced. Gate F remains open and D/J4 review remains deferred.

Run: `runs/shared_screen_diagnostics/4223bd9c1b2548dca882373bfd58087d`. The declaration precedes computation. Only the 13 frozen-training crops are passed to YIN; all three final-holdout crops are excluded before any tracker call. Cached YIN, stage masks and cents differences are retained for replay. All original intake records/reviews remain unchanged.

| Fixed screen stage | Retained frames | First loss at this stage |
| --- | ---: | ---: |
| Predicted GAME note interiors | 6,771 | — |
| Single predicted note | 6,771 | 0 |
| PESTO voiced support | 5,052 | 1,719 |
| Finite positive pitch estimates | 5,052 | 0 |
| YIN/PESTO agreement within 50 cents | 1,330 | 3,722 |
| PESTO/unshifted GAME agreement within 100 cents | 779 | 551 |

All 13 saved reliable masks replay exactly. Counts are summed frames from one artist proxy, not independent observations or musical accuracy. The first-failure accounting follows the declared order; losses are not independent causal attributions. Tracker disagreement is the largest measured screen loss, but it does not identify which prediction is correct. There is no evidence here to justify relaxing the screen or applying an octave fix.

The longest continuous reliable endpoint span is **0.68 seconds** (track 384); none reaches the existing v0.3 intervention procedure's 0.86-second raw support requirement. This supports keeping these crops withheld for pair construction. Keep them for diagnosis rather than delete them. No rendering or intervention audit was attempted and no alternative geometry was adopted.

## Ready-to-forward data request

`bpdr_training_export_request.zip` contains:

- `REQUEST.md`: exact artifacts, unchanged record/hash/offset/pitch conventions, original manifest and producing revision, failed-track logs and archive layout.
- `priority_track_ids.txt`: 64 tracks, one first missing track per frozen training artist.
- `all_missing_training_track_ids.txt`: all 407 missing eligible training tracks.
- `tracks.csv`: actual official annotation paths, artist proxies, frozen training assignments, priority flags and recorded license URLs.

The priority selection exactly matches the previously declared metadata-only list; evaluation artists are excluded. The ZIP's contents are verified against the separate handoff manifest. This request uses the existing common pipeline's supported export/resume operation; no unverified operator command or pipeline change is supplied. The request has **not been sent**. More exports improve source coverage but do not guarantee screen passes or create supervised correction targets. Vocadito remains the human-reference dataset and Jamendo remains supplementary predicted-source data.

**Next — Gates B/C (open):** Send the prepared 64-artist training-export request to the common-pipeline operator, receive existing-pipeline BPDR packages plus original manifest/producing revision, and import/screen them without changing partitions or thresholds. Construct known-target audited pairs before a new fit. No local heavy extraction is required now.

Do not add another diagnostic branch or start model training while awaiting this batch. After import, make one bounded readiness decision: measured source diversity/support and valid known-target pairs, or a recorded data limitation. Gate F delivery and future experienced review remain separate obligations.

## Completed verification

All **222 component regression tests pass**, including six new screen/split guards. Independent saved-array replay reproduces all 13 stage masks, exclusive loss counts, longest spans and refusal of current paired-intervention support. The 407-track CSV exactly matches the frozen eligible training catalog and official artist/path entries; the 64-track priority list is unchanged. All four ZIP members match their manifest hashes. Original reviews, intake inputs and 28 common-pipeline files remain unchanged. No held-out pitch inference, fit, new pair, threshold change or policy adoption.

Both maintained source notebooks freshly execute with zero cell errors and verified source/executed/HTML/index hashes, yielding **28 plots and 41 audio widgets**. The new two-panel screen/support plot was visually inspected. Panel: `runs/notebooks/19aee705027a49d39727d7124ff0179d/02_data_review_and_paired_experiments.html`. Full checks: `final_verification.json` in this diagnostic run. These are engineering checks, not musical accuracy or research validation.
