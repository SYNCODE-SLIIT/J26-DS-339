# Gate C: controlled shared-data pairs — 16 October 2026

**Gate C engineering pair construction is complete. Next: Gate E fair models and measurements in exploratory mode.** Research-ready examples remain zero; actual musical review is still deferred and usable delivered repair (Gate F) remains open. No learning model is trained in this step. No shared-pipeline code/settings, original data, source partitions or reviews are changed.

| Measured result | Count |
| --- | ---: |
| Declared training crops / artist proxies | 71 / 23 |
| Declared / recorded shift requests | 426 / 426 |
| Shifts passing all checks and pair packing | 306 |
| Failed or withheld shift requests | 120 |
| Crops / artist proxies with at least one pass | 70 / 23 |
| Crops passing all six shifts | 27 |
| Validated four-view engineering examples | 918 |
| Research-ready examples / model-training updates | 0 / 0 |

## Fixed scope and target meaning

All 71 crops from the declared second-batch queue are used in their frozen training partition. Every crop receives the unchanged six requested offsets (-30, -20, -10, +10, +20, +30 cents), with no outcome-dependent stopping or source replacement. Artist IDs remain proxies, not verified singer identities. No development, PP1-evaluation or final-holdout audio receives feature extraction or intervention here.

The existing v0.3 renderer, masks, supports, tolerances and audit code are unchanged. Source YIN arrays are reused after hash verification. Local frozen PESTO analyzes both original and detuned audio on the same grid/backend; this avoids measuring a pitch difference across two different extraction devices. The zero-shift FLOAT-WAV round trip preserves original decoded samples and repeated acoustics exactly. Every nonzero rendered WAV is saved/reloaded before its fresh acoustic/YIN audit. Sample counts and samples outside edit support are checked exactly. All failures and generated artifacts are retained.

The targets are **known injected relative offsets**, not judgments that an original singer was sharp or flat. Correction targets equal the negative injected cents only on independently audited plateaus, with zero anchors on audited outside frames. Ramps, unreliable frames and voicing gaps are excluded from supervised targets. The supplied GAME notes remain model estimates and never become intended-note correction labels.

For each successful shift, A/B share original audio and C/D share detuned audio. A/C use original GAME timing boundaries; B/D use the same controlled split, merge or jitter. The timing-only boundary feature preserves the shared export's Gaussian sigma 0.04-second representation; it does not use note pitch. A/B and C/D acoustic features are identical; only their boundary column changes. The 70-feature control never receives that column. Private references/targets are stored separately from model-input archives and checked for leakage.

## Counts and limitations

Three boundary corruptions produce three four-view examples per passing shifted waveform. They are dependent examples from the same source; they do not multiply independent recording or singer counts. The reported denominator includes all 426 requested shifts, not only successes. Multiple failure reasons can apply to one request; reason counts are not additive independent failures. No threshold, source mask, support geometry, model-estimate reference type or musical-review decision is relaxed to increase yield.

`summary.json` is compatible with the existing engineering pair reader. Every pair is typed `exploratory_not_research`, `reference_type=model_estimate`, `review_status=pending`, `training_eligible=false`; the research loader refuses it. Vocadito remains the human-reference dataset and Jamendo remains supplementary. The new paired set allows engineering learning comparisons; it does not demonstrate real-input correction accuracy, improved music, runtime repair permission or downstream benefit.

Before a combined fit, align the timing-only boundary encoding in **new derived input archives**. The older Vocadito pilot uses a triangular 0.04-second score; these new pairs preserve the shared export's Gaussian sigma 0.04-second score. Keep original audio, targets, audits and old view archives unchanged. This encoding difference must not become an unintended dataset cue in the model comparison. It does not invalidate the numerical audio audits or authorize shared-pipeline changes.

## Evidence and next action

Run: `runs/shared_pair_audit/5ef0c7b39c8e4d6493c26aca1820528e`. `declaration.json` and copied source hashes precede measurement; `locked_audit_policy.json` fixes the rules. `evidence.json` retains all attempts, sources, measured reasons and artifact hashes. `summary.json` lists only successfully audited/packed engineering pairs. Source WAVs and imported reviews remain unchanged. No shared extraction pipeline, separator or GAME rerun occurs; local frozen PESTO/YIN remeasurement is the pair audit itself, not learner fitting.

Gate E engineering: align timing-only boundary encoding across Vocadito/Jamendo in new derived inputs while preserving historical pairs, then lock the matched BPDR, augmentation-only and frame-only comparison protocol (same frozen training sources, audited targets, normalization, model budget and seeds) and the development preparation/evaluation rule before development tuning. Then run explicitly exploratory fits. Keep PP1 evaluation and final holdout reserved; own-training fit is not independent accuracy. Gate D/J4 review remains deferred and Gate F delivered repair remains open.

Do not return to the old harmonic-diagnostic branch automatically or infer research readiness from this larger engineering dataset. Gate B independent source suitability and actual singer identity remain limitations; Gate D/J4 and F are separate future obligations.

## Completed verification

All **237 component regression tests pass**, including nine new selection, Gaussian-boundary, target-leakage, input-integrity and research-refusal checks. The complete private artifact inventory contains 7,430 verified hashes. Saved-array replay reproduces 426 measured audit dictionaries and their target masks/values. Separate decoded-WAV checks verify 71 exact source identities and 426 exact outside-support waveforms. All 426 declared source/shift keys are retained without replacement.

All **918 four-view examples** pass waveform/input/target hashes, source/audit/policy provenance, acoustic identity within A/B and C/D, identical boundary metadata across A/C and B/D, deterministic corruptions and private-target exclusions. Every pair is read individually by the existing engineering reader; one real-pair `load_pairs` probe confirms integration without loading the whole collection into memory. The research loader refuses the collection. Legacy `singer_id` fields hold artist-proxy group IDs here and must not be reported as verified singers.

Original source and imported-review hashes, prior review sheets and all 28 common-pipeline files remain unchanged. No nontraining audio processing, review approval, learner fitting, production renderer adoption or accepted repaired output occurs. All passing/failing requests, originals, private references and generated augmentation WAVs remain evidence.

`boundary_encoding_note.json` verifies all 72 historical Vocadito view archives against their triangular boundary formula. They are preserved exactly. This pair run uses Gaussian timing scores to match the shared exports; a common derived encoding remains a Gate E preparation requirement before mixing both datasets.

Both maintained notebooks freshly execute with zero cell errors, **30 plots and 44 audio widgets**, with source/executed/HTML/index hashes verified. The new four-panel figure was visually inspected. Latest panel: `runs/notebooks/1888b91fcfd8429a8343135fe3d5cd3d/02_data_review_and_paired_experiments.html`. Full checks are in `final_verification.json` in this run. These are engineering validity checks, not human musical accuracy or independent BPDR benefit.
