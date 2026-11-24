# Fixed source-only Vocadito/Jamendo transfer — 15 October 2026

**Current gate: Gate F — renderer validation, open.** Gate D/J4 experienced musical review remains deferred. The fixed candidate does **not** transfer successfully to these ten recordings: all ten estimator results are refused, with no decomposition or corrected vocal output. This is evidence about this bounded method, not a finding that either entire dataset is unusable.

## Unchanged prospective protocol and inputs

Completed run: `runs/source_transfer/2b89e4c6e31d4adf8a82eca82673fb2f/evidence.json`. The [prospective declaration](../runs/estimated_harmonic_correction/a2eea865ced64ec19aafe310e9527d61/transfer_screen_protocol.json), hash `1236a98112eeda11cff1c9150db96fe73a8497a2d4e1be13ab39c66442ae6906`, is unchanged. Include all ten saved preparation records, train only, including old screen exclusions: three Vocadito crops from three verified singer groups and seven Jamendo crops from four artist proxies. Seven Jamendo crops are not seven independent verified singers; the bounded sample is not representative dataset yield.

Use the already saved technically accepted unchanged mono 22050-Hz audio, not new corpus extraction or separation. Dataset/group labels, notes and human F0 annotations are not passed to the estimator/fitter. Source samples and all ten hashes remain unchanged. Frozen estimator/decomposition code and PESTO/checkpoint provenance are verified. The same global nonpositive-frequency refusal, tracker agreement/confidence rules, guards, band center method and amplitude floor apply; no support-only relaxation, fallback or outcome tuning.

## All ten outcomes retained

| Record | Dataset | Estimator refusal | Predicted retained / decoded samples | Nonpositive frequency samples total / on retained support | Longest retained run, s |
| --- | --- | --- | --- | --- | --- |
| vocadito_vocadito_10_p00 | vocadito | NONPOSITIVE_ESTIMATED_FREQUENCY | 46763/200607 | 388 / 0 | 0.540000 |
| vocadito_vocadito_13_p00 | vocadito | NONPOSITIVE_ESTIMATED_FREQUENCY | 79380/413120 | 1073 / 0 | 0.400000 |
| vocadito_vocadito_14_p00 | vocadito | NONPOSITIVE_ESTIMATED_FREQUENCY | 67875/268962 | 482 / 0 | 0.820000 |
| jamendo_1071484_s4189500 | jamendo | EMPTY_GUARDED_SUPPORT | 0/441000 | 79009 / 0 | 0.000000 |
| jamendo_1086669_s5953500 | jamendo | NONPOSITIVE_ESTIMATED_FREQUENCY | 78498/441000 | 593 / 0 | 0.800000 |
| jamendo_1086669_s7056000 | jamendo | NONPOSITIVE_ESTIMATED_FREQUENCY | 70560/441000 | 322 / 0 | 1.220000 |
| jamendo_1101810_s1984500 | jamendo | NONPOSITIVE_ESTIMATED_FREQUENCY | 119381/441000 | 522 / 0 | 1.760000 |
| jamendo_1101810_s11466000 | jamendo | NONPOSITIVE_ESTIMATED_FREQUENCY | 153027/441000 | 24325 / 0 | 2.360000 |
| jamendo_1113639_s1323000 | jamendo | NONPOSITIVE_ESTIMATED_FREQUENCY | 110219/441000 | 1087 / 30 | 1.170476 |
| jamendo_1113639_s8820000 | jamendo | NONPOSITIVE_ESTIMATED_FREQUENCY | 91212/441000 | 8204 / 24 | 0.540000 |

There are **nine `NONPOSITIVE_ESTIMATED_FREQUENCY` refusals** and **one `EMPTY_GUARDED_SUPPORT` refusal**. The empty-support source is `jamendo_1071484_s4189500`. Its frequency array is still recorded diagnostically, but the estimator's returned reason is not overwritten. No fitter is invoked on a refused result; all ten decomposition statuses are `WITHHELD_ESTIMATOR_REFUSAL`. There are no decomposition arrays or generated/candidate WAVs. Existing technical unchanged acceptance is not musical or transfer approval.

The synthetic five rendered/full-pass results remain valid for their controls; they do not establish applicability on these recordings. No current renderer is adopted, repaired output delivered or research training anchor created.

## Coverage and support diagnosis

Intermediate retained predicted samples total **194,018/882,689 for Vocadito** and **622,897/3,087,000 for Jamendo**, with all crop rows retained. The denominator is every decoded sample, not human-voiced samples. Predicted support does not override a refusal. Tracker agreement is a source-only diagnostic; no true-voiced coverage, intended-note correctness or musical accuracy is inferred from it. Jamendo lacks verified human F0 ground truth in this workflow.

Seven crops have all nonpositive values outside retained support; two Jamendo crops also have **30 and 24 inside retained support**, respectively `1113639_s1323000` and `1113639_s8820000`. The remaining crop has no guarded/retained support. Thus changing the global check to ignore only outside-support problems would not address every local phase issue; no such change is made here. Nonpositive analytic-phase derivatives are estimator failures/limitations, not physically negative vocal pitch.

A separate [post-screen geometry analysis](../runs/source_transfer/2b89e4c6e31d4adf8a82eca82673fb2f/support_geometry.json) reads the saved masks without changing them. Vocadito longest retained segments are **0.54, 0.40 and 0.82 s**, all shorter than the existing experimental renderer minimum **0.86 s**. Four Jamendo crops contain seven diagnostic positive-frequency retained runs at least 0.86 s; this is geometry only, not eligibility, harmonic fidelity, strict phase closure or edit permission. All ten original refusals remain. No interval is merged, gap-filled, promoted or corrected.

The method therefore needs a revised source/segment strategy before natural-vocal correction. Simply using the earlier synthetic pass count, the technical loader result, or the presence of intermediate masks would hide this limitation. These results do not settle the suitability of Vocadito or Jamendo for all BPDR roles; earlier annotation, source/license/split, manual review and research gates persist.

## Validation and maintained notebooks

**175 tests pass** in `runs/source_transfer_regression_2026-10-15.txt`. Five new tests verify unavailable versus measured-zero support, predicted coverage versus invalid retained frequencies, withholding refused crops without invoking a fitter or losing arrays, source-only fitter arguments, and preserving the exact full training selection. The completed run verifies **434 frozen hashes** and saves ten prediction NPZs plus declaration/code snapshots. Geometry is a separate saved-array diagnosis; it is not a new parameterized experiment or a changed refusal.

Both maintained notebooks are updated with every crop, dataset/group distinctions, all ten refusals, decoded-sample coverage and the separate geometry limitation. Both maintained notebooks executed from fresh kernels with **zero cell errors**, **22 plots and 40 audio widgets** total: notebook 1 has two plots/two audio; notebook 2 has twenty plots/38 audio. [Current panel notebook](../runs/notebooks/a6f39c9a20344e9f81e112b2a4e91a9b/02_data_review_and_paired_experiments.html). Source/executed/export hashes, all 434 frozen hashes and ten prediction artifacts verify. All ten fixed-filter phase/frequency arrays recompute exactly from the original audio and saved estimated centers, with source-only tracker/support masks, refusal reasons, coverage and geometry rechecked. All ten source-audio hashes and both review-file hashes are unchanged. [Final verification](../runs/source_transfer/2b89e4c6e31d4adf8a82eca82673fb2f/final_verification.json). The new four-panel plot was visually checked. Maintained sources retain no stale outputs; historical executed copies remain evidence. The original preparation manifest, train/development/test partitions, review files, research eligibility and historical results remain unchanged. No learner training, pair generation, new source separation or vocal correction occurs.

## Next — Gate F (open)

Diagnose frozen tracker-agreement/support fragmentation and predeclare a revised segment-level estimator comparison on the same ten recordings plus the eight synthetic controls; preserve all v0.1 refusals and do not render vocal corrections.

This next work is planned, not implemented here. Study source-only agreement and the guards that fragment support, then specify a separately versioned segment-level candidate and comparison before measuring it. Do not silently weaken the current rule, edit the old refusals or rerun v0.1 with selected crops until it appears to pass. No vocal correction is credited before its source support and reconstruction are tested. Gate D/J4 remains deferred; research-ready examples and accepted repaired packages stay zero. Windows/macOS portable integration remains planned.
