# Next experiment: qualify useful preparation without changing the research question

Protocol draft, 7 November 2026. No experiment described below is represented as completed. Numeric criteria are proposed engineering choices, not literature standards; freeze their final values and source manifests before new held evaluation. This protocol does not change any historical gate or result.

## E01. Replicate and test cleanup for the actual handoff extractor

**Hypothesis.** On supported noisy solo vocal inputs, a fixed preparation policy improves a frozen downstream extractor while avoiding meaningful damage to clean or expressive recordings. The prior positive pYIN result motivates this test; it does not answer it.

**First freeze the integration target.** The shared PP1 Vishwa adapter currently uses the pYIN baseline in `src/melody_understanding`. The standalone BPDR demo separately supports GAME. Record which is primary, its code/checkpoint/config hashes, preprocessing and time mapping. A new extractor later requires a new compatibility evaluation. BPDR consumes the extractor for testing; its notes/key outputs remain owned elsewhere.

### Stage E01a: replication and measurement hygiene

Use only the already opened 21 training-role Vocadito sources and the four root ESC-50 clips from the existing experiment. Keep reserved people, source descendants and validation noises closed.

1. Replay all saved-record aggregates and check declaration bindings. This step has now been performed by `tools/audit_continuation_evidence.py`; it is not full inference replay.
2. Independently rerun raw and processed pitch extraction. The fixed three-case diagnostic in `tools/replay_noise_audit_subset.py` now reproduces all 18 estimates exactly with the historical later-timestamp tie convention. This is a bounded initial check, not the full replay. Pin timestamp alignment and exact-tie handling; an alternative tie convention changed a few pYIN scores by up to 0.4315 percentage points.
3. Verify physical input mixtures and noise offsets, audio hashes, model weights, 48 kHz enhancer processing/resampling, documented delay compensation and final encoding. Do not repair a length mismatch using cyclic `numpy.resize`; fail or use explicitly documented padding/cropping/alignment. The existing 189 outputs had equal sizes, so this latent code issue was not an observed size mismatch in that set.
4. Add a separately declared peak-safe delivery policy for the seven outputs above full scale. Apply documented gain/headroom handling, then re-extract and rescore the actual decoded delivered files. Preserve the old floating-point results as historical evidence. A gain change can alter an estimator's output, so do not assume old scores remain valid.
5. Report clean controls explicitly. The saved neural gate accepts four clean sources and one is harmful under the PESTO score definition. Do not omit clean harm because the headline table contains noisy cases only.

**Stop condition.** If scores cannot be reproduced, alignment is unresolved, or encoding invalidates the result, fix this stage before training or model selection. No “corrected” result replaces the historical failed gate.

#### E01a delivery encoding declaration

Export all 189 saved neural candidates to new mono 22,050 Hz PCM24 files.
Keep sample count fixed. Apply one constant gain per clip,
`min(1, (1 - 2^-20) / peak)`, and preserve the original FLOAT files.
Verify the decoded output against the exact quantized scaled samples.
This changes export only. It introduces no delay compensation.

First re-extract pYIN, PESTO and Praat on every candidate whose original
peak exceeded full scale, using the seven-case subset identified by that
physical condition. Reproduce the original FLOAT scores before comparing
the PCM24 scores. Use the historical later-frame exact-tie convention.
Report clean-control membership and each before/after score. Stop on
any failed integrity or original-score replay. This initial subset
cannot qualify the full gate. Full export rescoring and delay validation
remain prerequisites for E01b. Budget is 189 deterministic exports and
42 pitch extractions, with no fit or threshold tuning.

### Stage E01b: a new declared development contrast

Keep data, mixtures, acoustic analysis, sample timing, output encoding and downstream extractor identical. Compare:

- B0: format-only original.
- B1: existing conservative spectral preparation, with its own declared noise-reference requirement.
- B2: fixed DeepFilterNet3 output.
- B3: original-or-B1/B2 chosen by a fixed input/output-only decision policy, with the original always available.

Report B2 always-on and B3 separately. Freeze a single baseline policy before scores; record which variable changes from the historical three-tracker-consensus gate. Use prospective development data for any new selector/thresholds. The previously observed 18 PESTO harms cannot be reused as an untouched final test of a gate designed to reject them.

**Runtime information allowed:** raw/candidate acoustics, noise-only evidence where available, clipping/headroom, voiced coverage, measured input-output pitch changes on reliable frames, discontinuities, analyzer disagreement, candidate type and computational cost.

**Runtime information forbidden:** clean original, human F0/note labels, injected support or amount, source/singer identity shortcuts, file names encoding the corruption, and held-reference scores. Confidence and tracker agreement are diagnostics, not proof of correctness.

An optional lightweight supervised selector should only follow sufficient independently grouped development examples. Train on measured benefit/harm labels from training data. Compare with the fixed policy and a simple linear classifier before a boosted-tree/ExtraTrees option. Do not add a new neural architecture to this experiment.

### Stage E01c: frozen confirmation

Record singer, original recording, session/device, melody where applicable and noise-file roles before augmentation. Keep all derivatives together. At least one confirmation slice must contain unseen noise recordings; at least one must contain unfamiliar singers/recording sessions. Use the already planned consented 5–15-second hum pilot to measure deployment feasibility, then determine the independent confirmation sample size. Vocadito singing and digitally mixed noise alone cannot establish natural humming performance.

Do not open fresh confirmation until B3, preprocessing, renderer, thresholds, primary metric and exclusions are frozen. The existing four ESC-50 validation clips may be part of a declared future confirmation, not repeated tuning. Check existing locks before access. Preserve every attempted recording and reason for exclusion/refusal.

**Measurements:**

- Primary usefulness: paired note precision/recall/F1 from the frozen downstream extractor against independent annotations, with onset-and-pitch and offset-aware matching reported separately.
- Pin the `mir_eval` version and matching arguments. A standard starting definition is 50-cent pitch and 50-ms onset tolerance; offset-aware scoring also uses the larger of 20% of reference duration or 50 ms, with `strict=False`. Report the offset-free variant separately. These are matching tolerances, not permissible waveform-edit magnitudes. [Official transcription metric documentation](https://mir-eval.readthedocs.io/latest/api/transcription.html).
- Acoustic mechanism: RPA50 against human performed-F0 labels where available, voicing errors and pitch preservation on clean controls. These do not measure intended-note accuracy.
- Preservation: harmful-output counts both over all attempts and over accepted outputs; separate numerical harm, audible artifact and participant-reported idea loss.
- Coverage: repair/cleanup acceptance, beneficial delivered cases per all attempted cases, unchanged acceptances, refusals and technical failures.
- Runtime: cold-start and warm median/p95, file lengths and documented hardware. No latency estimate from only a few favorable clips.
- Uncertainty: paired singer-level summaries and bootstrap intervals; report noise/session slices. Frames and repeated noisy versions are not independent participants.

**Proposed advancement criteria, to freeze before held testing:** no broken handoff/encoding/time mapping; positive singer-level primary note-F1 gain with a 95% lower bound above zero; no observed material clean-control loss under the predeclared preservation margin; harmful-output rate reported with uncertainty and an agreed upper limit; useful coverage above a stated practical floor. As an initial development target, consider at least +3 percentage points mean note-F1 on the declared noisy subset and at least 25% beneficial coverage across all in-scope noisy attempts. These targets are proposals, not predictions. Pilot variation may justify different pre-test values, with reasons recorded. Existing stricter pitch-repair gates stay unchanged.

The scientific preservation comparison should use matched acceptance coverage or report the complete risk/coverage relationship. An empirical 0% harm in a small, correlated sample does not establish a population harm rate below 5%.

**Adoption rule:** a result can pass for the frozen primary extractor and fail to generalize to another. Report both; never relabel the historical all-tracker gate as passed. Adopt only the configuration actually tested, and verify final user-reviewed WAV integration. If no candidate qualifies, keep original/rerecord behavior and report the failed cleanup qualification.

## E02. Finish a scientifically interpretable BPDR comparison

**Current state:** six of twelve saved fit receipts exist in `2026-11-06_short_objective_comparison`; all six checkpoint/declaration bindings pass the 7 November audit. Four augmentation-only fits are complete, with nearly zero output. Two frame-only fits are complete. Full BPDR and full replay are absent. Earlier records of this run being live are historical; no training process was found at initial inspection in this review.

**Do first:** inspect the remaining training-only positive-fit/head diagnostic, real batch labels, per-view effective positive/zero weights, bounded head saturation and gradients. Confirm whether the arm can fit a small diagnostic sample while preserving explicit no-change examples. The current balanced query classes do not necessarily imply balanced nonzero supervised targets after constructing four views and adjacent frames. Treat this as a hypothesis to measure, not a diagnosed implementation bug.

Use fold 1 and its first frozen 36-query batch for a bounded diagnostic.
Count the actual masked target mass across A/B/C/D. Inspect the saved
augmentation-only head and gradients on this training batch. From seed
339, fit that same batch for 200 updates with the frozen augmentation-only
objective and optimizer, then measure positive signed response, MAE and
original/known-zero requests. Log at 0, 50, 100 and 200. This is one
training-only memorization check with no hyperparameter sweep, no held
evaluation and no replacement of historical fits. A failure is a
diagnosis prerequisite, not permission to change the comparison silently.

**Resume rule:** if the frozen implementation is valid, resume its missing fits from matching receipts and complete its independent replay once; retain the nearly-zero control. If a substantive implementation or target issue is found, stop, document it and declare a corrected matched experiment in a new folder. Do not mix corrected fits with old control fits. Do not cherry-pick a successful seed.

**Research requirements:**

- Same source partitions, four-view audio, supervised anchors, budget, selection rule and cleanup for all three learned arms.
- Boundary-aware arms receive genuinely changed provisional boundary input. Verify false split, false merge/missing boundary and jitter coverage; a shifted-boundary-only result is narrower than the proposal.
- Keep accepted originals, identity resynthesis, vibrato/glide/transposition preservation cases and outside-support labels visible.
- Evaluate prediction error and real delivered-WAV error, all-attempt useful coverage, false edits, known-zero damage, per-singer/domain/direction results and downstream benefit independently.
- Do not use a supplied guide, original reference or label-derived support at inference in the core comparison. Any reference-conditioned extension is a distinct experiment and scope decision.
- Keep the original four-view BPDR objective intact in the main contrast. An alternative backbone, loss or regularizer is a secondary hypothesis, not a replacement for the required matched controls.

**Stopping rule:** one diagnostically justified matched comparison, with budget and seeds declared before held scores. By 11 December, choose a candidate only if delivered-output qualification passes. If BPDR has no advantage, record the negative result. A valid null result completes the scientific test but does not qualify a harmful repair feature.

## E03. Integration evidence, after qualification

Run the same selected recording through upload, actual processing, original/prepared playback, accept/reject, export and the downstream consumer. Save source/working/candidate/accepted/consumer hashes. Exercise refusal, rollback, protected intervals, changed user selection, multiple accepted edits and corrupted package cases. The exact final accepted WAV must be the consumer input; a preview beside an unchanged handoff is insufficient.

Do not add final transcription, key, beat, tempo or harmonization features inside BPDR. Use the existing public handoff statuses and retain source-time semantics. This is the end-to-end evidence needed for component completion, separate from research superiority.
