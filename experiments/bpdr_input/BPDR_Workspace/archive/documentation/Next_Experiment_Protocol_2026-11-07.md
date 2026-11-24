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

After that subset replay, score all 189 PCM24 exports with the three
frozen estimators (567 estimates). Keep each historical gate decision
fixed to isolate encoding in this diagnostic. Report always-enhanced
and historically routed scores, including all 21 clean controls, with
singer-level means and harm counts. Cache each completed case against
source, annotation and estimator hashes. This is export sensitivity on
opened development sources. Gate recomputation, delay validation and
held qualification remain separate prerequisites.

For alignment diagnostics, compare all 21 clean saved enhancer inputs
with their FLOAT outputs and new PCM24 exports. Use the full clip and
four uniformly spaced one-second windows. Measure the peak normalized
waveform cross-correlation within ±100 ms, its value at zero lag and
the strongest competing peak outside ±2 ms of the winner. Calibrate
the lag sign on a known 100-sample synthetic delay. Apply no time shift.
This no-fit check can expose a consistent residual offset. Periodicity
and enhancement artifacts can make lag ambiguous, so observed waveform
lag alone cannot prove the enhancer's internal delay configuration.

### Stage E01b: a new declared development contrast

#### Recovered enhancer runtime and delay provenance (8 November)

The historical scratch wrapper/environment and cached DeepFilterNet3
configuration/checkpoint have been located. Before any renewed neural use,
freeze their hashes, package versions, resampling source and Python source
closure in `tools/audit_neural_runtime_replay.py --mode declare`, running
inside that isolated environment. Use a workspace-local model copy with
logging disabled so the recovered cache remains unchanged. The wrapper
uses the enhancement function's default `pad=True`; inspect the executable
STFT compensation, not just file lengths. Report FFT/hop, compensation,
lookahead settings and the limitation of recovering provenance after a run.

Budget: the same fixed track-1 clean/rain-10/rain-0 subset, three enhancement
calls in create and three in fresh-process verify. No new source, fitting,
post-filter, attenuation-limit change, thread sweep, alignment shift or
consumer rescoring. Preserve historical FLOAT files; write new outputs.
Require exact new-run waveform replay and separately report exact historical
waveform/file matches and all differences. A mismatch is retained, never
replaced or treated as successful historical replication. Configuration
verification or waveform equality does not qualify the failed cleanup gate.

First run the actual-consumer compatibility check on the same fixed
track-1 clean/rain-10/rain-0 cases used in the earlier audit. Compare
format-only PCM24 from the saved enhancer input with its new neural
PCM24 output. Call the frozen owning `analyze_vocal(path, trim=False)`
for both. Use mir_eval 0.8.2 from the isolated evaluation dependency
directory, with the note tolerances specified below, and score both
annotator sets separately. Intersect reference intervals with the
first-eight-second crop and report truncated-note counts. Score RPA50
with the historical later-timestamp exact-tie rule as a secondary metric.
Keep all three cases and both baselines. This six-call development
compatibility check selects no gate and opens no reserved data.

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

The actual-consumer development extension evaluates all 189 already
opened cases using format-only and neural PCM24, for 378 calls.
Use the same frozen consumer, mir_eval version, two annotators,
eight-second reference intersection and RPA tie rule as the three-case
compatibility check. Cache by waveform, references and consumer hashes.
Record per-source/noise/SNR metrics, clean controls and singer-level
paired means/intervals. Report per-case F1 drops greater than 0.02 as
a descriptive development harm count, with both denominators. This
does not qualify a runtime selector and does not evaluate B1 without
its required declared noise-only reference. No parameter selection,
fresh source/noise access or model fitting occurs in this extension.

Before interpreting that extension, verify all 189 mixed and FLOAT
waveforms against their 7 November audit hashes and all 189 encoded
outputs against their export receipt. Run a fresh-process replay of
all 378 owning-consumer calls using
`tools/verify_primary_cleanup_development.py --mode verify` after the
first result completes. Freeze the verifier by its recorded source
hash; retain raw consumer note events and pitch tracks in the replay
cache. Require exact per-case metric and preprocessing equality and
independently reconstructed singer summaries within 1e-12. Budget:
378 additional calls, no fitting, selection, threshold changes or
reserved-data access. A mismatch stops interpretation; retain both
outputs and investigate rather than overwrite the first result.

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

The augmentation-only diagnostic replay matched exactly. Next use the
same fold-1 batch and seed for 200 shared warm-up updates followed by
200 updates with the frozen full BPDR objective enabled. Record all
loss terms and the same positive/preservation/head metrics. This one
400-update training-only diagnostic asks whether the original response
terms recover from warm-up collapse. Keep the head, anchor weights,
optimizer and samples fixed. No held evaluation or loss sweep.

The original full-objective diagnostic failed the same training-batch
fit prerequisite. The current compact model multiplies its bounded
regression output by a sigmoid activity head, while this comparison
supplies no activity supervision. The proposal encoder has a direct
regression head. Test one focused variable by bypassing the activity
multiplier using the existing `gated=False` configuration. Keep the
same seed, batch, initialization, 200-update augmentation objective,
anchor weighting, optimizer and ±30-cent bound. Retain the activity
parameters so capacity and initialization order stay fixed. Log head
gradients and positive/zero behavior. Stop after this one training-only
probe. No held score or selection threshold changes.

Inspect gradients without fitting another model. On the same batch,
decompose the official anchor loss into nonzero-target and zero-target
contributions using its exact per-example/view normalization. Compare
their parameter gradient norms, cosine and dot products with the total
objective at initialization and the saved augmentation checkpoint.
Also check for identical input tokens with conflicting supervised
targets. Bind the report to the existing checkpoint, trace and data.
This audit chooses no new weights or learning rates.

The audit finds a saved-model zero-target gradient norm of 0.005488
versus 0.00003175 on nonzero targets, with a negative positive/total
gradient dot product. No exact-input target conflicts occur in the
examined batch. Test one separately declared common-anchor correction:
half the anchor mass on nonzero targets and half on zero targets,
normalizing per example within each class and then over examples with
that class. Keep the original smoothness coefficient, gated model,
batch, initialization, optimizer and 200-update budget. This is a
training-only prerequisite, not a change to any historical result or
acceptance gate. Stop after one run, with no class-weight sweep.

Before the corrected comparison, audit split, merge and jitter views
on all existing V64 arrays. Recover provisional edges from each input's
Gaussian boundary track using peaks of at least 0.5, 100 ms apart.
Reuse the existing corruption helper with seed 339. The split inserts
the midpoint of the largest edge gap, merge removes the middle
internal edge and jitter uses the helper's ±60 ms range. Count
unsupported cases and effective changes after voicing/padding masks.
Keep all acoustic columns fixed. Count affected contexts on original,
positive and known-zero masks; none of these scoring masks enters
the corruption procedure. This audit fits no models.

**Resume rule:** if the frozen implementation is valid, resume its missing fits from matching receipts and complete its independent replay once; retain the nearly-zero control. If a substantive implementation or target issue is found, stop, document it and declare a corrected matched experiment in a new folder. Do not mix corrected fits with old control fits. Do not cherry-pick a successful seed.

**Research requirements:**

The corrected comparison uses the verified equal-class anchor in all
three freshly initialized arms, with the original BPDR-specific terms
and weights exclusive to full BPDR after 200 shared warm-up updates.
Retain four V68 source-disjoint folds, seed 339, optimizer, model
configuration, 1,200 updates and the exact V68 query traces. Cycle
split, merge and jitter equally, for 400 updates per corruption in
each fit, and score all three held perturbations. Each input supplies
its own acoustic provisional boundaries. Keep frame-only capacity
matched with a zero boundary channel.

Budget is twelve fresh fits followed by exact checkpoint replay,
with no seed, weight or checkpoint sweep. Smoke-test the three
corruption types in all arms before training. Require the existing
raw gates on every perturbation, lower mean held counter-detuning
error than both controls, and lower boundary sensitivity than
augmentation-only on every perturbation before WAV testing.
Keep all 1,152 attempts, 707 exclusions, 53 original controls and
unassessed groups. The existing identity-resynthesis controls and
expressive originals still require held-checkpoint assessment before
research or repair qualification. The six earlier fits remain
historical stopped evidence.

- Same source partitions, four-view audio, supervised anchors, budget, selection rule and cleanup for all three learned arms.
- Boundary-aware arms receive genuinely changed provisional boundary input. Verify false split, false merge/missing boundary and jitter coverage; a shifted-boundary-only result is narrower than the proposal.
- Keep accepted originals, identity resynthesis, vibrato/glide/transposition preservation cases and outside-support labels visible.
- Evaluate prediction error and real delivered-WAV error, all-attempt useful coverage, false edits, known-zero damage, per-singer/domain/direction results and downstream benefit independently.
- Do not use a supplied guide, original reference or label-derived support at inference in the core comparison. Any reference-conditioned extension is a distinct experiment and scope decision.
- Keep the original four-view BPDR objective intact in the main contrast. An alternative backbone, loss or regularizer is a secondary hypothesis, not a replacement for the required matched controls.

**Stopping rule:** one diagnostically justified matched comparison, with budget and seeds declared before held scores. By 11 December, choose a candidate only if delivered-output qualification passes. If BPDR has no advantage, record the negative result. A valid null result completes the scientific test but does not qualify a harmful repair feature.

**8 November numerical execution amendment.** A frame-only checkpoint
produces a two-float32-ULP within-batch difference on one endpoint frame,
despite exact duplicate tokens; repeated batch outputs reproduce exactly.
The no-fit diagnosis is in `2026-11-08_frame_duplicate_diagnosis`.
`tools/resume_balanced_short_objectives.py` records a separate declaration
under the original study's `numerical_resume/`, replaces precisely that
construction assertion with a maximum four-ULP check, and verifies exact
duplicate inputs. It retains the actual predictions, all scientific gates,
the frozen fitter and bitwise checkpoint replay. Five verified fits are
reused; only the missing seven are fitted. Existing prediction files must
match exactly before reuse. No source files, historical locks, fit targets,
learning budgets or selection rules are changed. A larger residual stops
the run for investigation. This amendment is numerical execution handling,
not a relaxed scientific preservation/accuracy gate.

### E02 identity controls after checkpoint replay

After the balanced comparison's twelve-checkpoint replay, evaluate all eight
existing cases from the 6 November identity-resynthesis declaration using
each source's held checkpoint in every arm. Recreate clean, forced identity
resynthesis and registered detuned inputs with the frozen renderer and
preprocessing; use the same split/merge/jitter generator on each input's own
acoustics. Keep unsupported boundary views in the accounting. Budget: no
fits, 24 waveform feature extractions and up to 216 two-view predictions per
create/verify process. Score registered physical masks and common reliable
frames separately; masks enter scoring only. Report clean/identity requests,
identity sensitivity, physical-detune response error, boundary sensitivity
and voiced coverage. Exact prediction/summary replay is required. Frame-only
identical inputs are a construction check; retain numerical residuals.
This is inspected development control evidence, not fresh confirmation or
intended-note correctness. Tool/declaration:
`tools/check_balanced_identity_controls.py`, parent study `identity_controls/`.
No model promotion follows from this extension.

## E03. Integration evidence, after qualification

The real-hum workflow engineering check uses only the already opened
training-role eight-second F01_0001_0001_1_D phrase, its existing controlled
rain derivative and constructed silence. Freeze UI/operator/input hashes
before execution. Budget: four routes in each create/verify process and
three real pYIN calls per process. Upload/recording bytes are injected at the
widget boundary; no microphone claim. Real processing, acoustic review,
protection/undo, export and consumption remain active. Force explicit
automated-fixture confirmation metadata in the harness, never human approval.
Require exact consequential replay, timeline integrity, positive paired SNR
on the fixed rain example under existing preservation limits, exact selected
consumer hashes and no accepted path on silence refusal. No retuning, fitting
or source selection after scores. Tool/source snapshots are recorded under
`2026-11-08_real_hum_workflow`; no general qualification follows from one phrase.

The local-edit combination engineering check declares one generated
ten-second harmonic source and two manual +/-20-cent requests on [2,3)
and [6,7), with unchanged permission, renderer and acoustic audit limits.
Each single edit and their combination must pass final encoded/decoded
auditing and fresh-process replay. Budget: four audits per process, eight
total, plus two real frozen-pYIN calls on the combined exported fixture.
No fitting, new data, label selection or threshold sweep. This verifies
transport, support and numerical-audit behavior; it supplies no independent
participant, trained-model accuracy or useful real-hum qualification.
Tool: `tools/check_local_combination_audio.py`; corrected declaration and
source snapshots: `2026-11-08_local_combination_audio_v2` under demo sessions.

Before candidate qualification, an explicitly synthetic engineering check
may exercise the selected-WAV transport alone. The 8 November fixture
declaration under `results/demo_sessions/2026-11-08_primary_selected_handoff/`
binds two five-second synthetic original/gain-prepared exports, four real
frozen-pYIN calls including repeats, no fitting and no human approval.
This does not activate a failed candidate, complete shared PP1 orchestration,
or satisfy the beneficial-output/qualified-repair demonstrations below.
`repair/primary_handoff.py` verifies reviewed package status, selection,
hashes, frozen consumer, copied input and post-consumer integrity. The
interactive demo uses explicit user confirmation; fixture mode requires
a separate explicit argument and is never accepted as human judgment.

Run the same selected recording through upload, actual processing, original/prepared playback, accept/reject, export and the downstream consumer. Save source/working/candidate/accepted/consumer hashes. Exercise refusal, rollback, protected intervals, changed user selection, multiple accepted edits and corrupted package cases. The exact final accepted WAV must be the consumer input; a preview beside an unchanged handoff is insufficient.

Do not add final transcription, key, beat, tempo or harmonization features inside BPDR. Use the existing public handoff statuses and retain source-time semantics. This is the end-to-end evidence needed for component completion, separate from research superiority.
