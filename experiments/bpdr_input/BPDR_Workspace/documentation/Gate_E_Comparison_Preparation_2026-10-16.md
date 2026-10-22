# Gate E preparation - 16 October 2026

Input alignment and the comparison protocol are complete. Gate E remains open: no new learner fitting, development processing or independent accuracy measurement has occurred. Research-ready examples remain zero; Gate D/J4 review is deferred and Gate F delivered repair remains open.

## What was completed

The component now has 936 aligned four-view engineering examples: 18 from the original Vocadito pilot and 918 from Jamendo. Vocadito remains the primary human-reference source and covers one recording / one annotated singer ID. Supplementary Jamendo covers 70 source crops / 23 artist proxies, whose actual singers remain unknown. Repeated variants are dependent and are not extra independent singers or songs.

All 3,744 derived views use a timing-only maximum Gaussian score with sigma 0.04 seconds. Every saved time grid and first 70 acoustic features is identical to its source. Only Vocadito's 72 boundary columns change from the historical triangular score; Jamendo's boundary columns remain identical. Original archives, WAVs, targets, audit masks/policy/results, partitions and reviews are preserved. Private targets stay separate. Human annotations and model estimates remain distinct reference types, with their original pending review states.

The training-only normalizer is computed in bounded memory. Each dataset receives half the probability mass; groups, recordings, phrases, shifts and corruptions divide that mass equally in sequence. Within each pair, four views share its weight and real frames share each view's weight. Thus longer phrases and many variants do not automatically dominate. Unvoiced real context remains and padding is excluded. This is an explicit exploratory mixture choice, not population weighting or a solution to Vocadito's single-singer coverage. The frame-only control will use exactly the first 70 saved normalization columns.

A common 300-by-8 weighted sampling schedule is saved for the three future conditions. It is a planned example schedule; no updates have run. Independent verification replays the schedule and normalizer moments, authenticates original artifacts, compares all derived views with their originals, checks engineering loader integration for both datasets and confirms research-loader refusal. No source expansion, waveform rendering or acoustic extraction is performed.

## Locked comparison and development rule

The first exploratory fit uses seed 339, 300 updates, 100 warmup updates, batches of 8, AdamW at 0.001 and gradient clipping at 1. The common encoder has hidden size 128, four layers/heads, feed-forward size 512 and dropout 0.1. Existing provisional loss weights are boundary 1, detuning 1 and smoothness 0.1. Conditions are augmentation-only (71 inputs), frame-only (70 inputs) and BPDR (71 inputs). Full phrase context is retained. Common-shaped initial tensors must match; frame-only's projection uses the first 70 columns of the common 71-column initialization.

The new runner must enforce the same saved batches and normalization in all conditions, preserve the existing loss's per-example reduction before the batch mean, and evaluate one example at a time. The old trainer samples examples uniformly, fits concatenated frame-weighted normalization and evaluates one enormous training-fit batch; those paths need adapters for this experiment. Existing model/loss source modules are preserved for historical reproducibility.

Prepare development using the existing frozen source partitions, all screen-eligible development exports and Vocadito development phrases, unchanged suitability/support/audit rules, all six fixed shifts, all three corruptions, the same boundary encoding and all failures retained. Do not inspect PP1 evaluation or final-holdout acoustic outcomes. No new target is an intended musical note: targets remain known relative injected offsets. Do not approve model-estimated GAME pitches as original-intonation labels.

Save checkpoints every 25 updates. Select by minimum development macro correction MAE, with an equal mean over A/B original zero targets and C/D injected-correction targets, then equal corruption/shift/phrase/recording/group/dataset aggregation. Ties select the earliest update. All conditions use the exact same prepared development examples and candidate updates. If no usable development pairs exist, stop selection and report unavailable; training fit and reserved holdouts cannot be substituted. Report results per dataset and reference type, with any equal-dataset macro explicitly labeled. A single declared seed is an engineering pilot, not a stability analysis or novelty claim.

Report original-input correction magnitude on voiced real A/B frames independently of the injected-detuning target mask. This is a no-change diagnostic; calling original singing musically acceptable requires future review. Research eligibility, musical improvement, renderer safety and downstream integration are separate obligations.

## Next - Gate E

Prepare the declared development evidence, then implement and run the three matched exploratory fits with this contract and reloadable checkpoints. Gate E does not close merely because inputs and a protocol exist. Gate D/J4 remains deferred and Gate F remains open.

Run: `runs/comparison_preparation/5c135b2dca4e4da68422d86d0caa3681`. See `declaration.json`, `locked_protocol.json`, `summary.json`, `normalizer.npz`, `sampling_schedule.npz`, `evidence.json` and `independent_verification.json`.

## Completed verification

All 245 component tests pass, including eight new checks for hierarchical mixture weights, heldout/research guards, duplicate identities, real-frame normalization, invalid weights and a reproducible common schedule. The initial sandbox-only temporary-directory failure was resolved by running the same tests with the required temporary-file access; it was not an algorithm failure.

Independent verification checks all 936 pairs / 3,744 views against original feature grids and acoustic columns. It authenticates saved WAV/target/audit/source hashes, replays weighted normalization moments independently and reproduces the saved sampling schedule. Both dataset engineering-loader probes pass and the research loader refuses pending review data. Only 72 Vocadito boundary columns change; all Jamendo boundary columns remain exact. Original artifact and derived artifact inventories are recorded in the verification.

Both maintained panel notebooks freshly execute with zero errors, 31 plots and 44 audio controls. Source, executed-copy, HTML and evidence-index hashes verify; the new three-panel figure was visually inspected. All 28 shared-pipeline files and frozen source/review inputs remain unchanged. No real-data learner is fitted on the new collection and no development, PP1-evaluation or final-holdout acoustic processing occurs in this step.

Latest panel: `runs/notebooks/d84a4ce120e0429d9b6adba42a33e406/02_data_review_and_paired_experiments.html`. Final proof: `runs/comparison_preparation/5c135b2dca4e4da68422d86d0caa3681/final_verification.json`.
