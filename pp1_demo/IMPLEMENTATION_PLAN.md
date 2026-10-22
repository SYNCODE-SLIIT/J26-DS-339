# PP1 visual demonstration — implementation plan

Prepared 22 October 2026 for the requested 23 October demonstration. This is a plan, not a claim that the application is implemented.

## Goal and boundaries

Build a local visual application showing uploaded audio flowing through Praveen → Vishwa → Pasan → Rahul. Implement Vishwa's explanatory presentation content; provide clearly empty editable presentation sections for the other owners. Pipeline adapters may call existing teammate implementations, but must not edit them. Distinguish an empty owner narrative from an unavailable runtime model: actual runtime results can still be shown in a generic result card.

Keep all application code/configuration inside `pp1_demo/`. Read component source/checkpoints from their existing folders. Store uploads/results under an application-local ignored directory. Do not retrain models, modify original datasets, edit teammate implementations or present temporary inputs as learned predictions.

## Current checkout and prerequisite findings

Remote check completed 22 October 2026: `git fetch origin` succeeded. Branch `feature/vishwa-pp1-melody-understanding` and its upstream both point to `b27dba2`, with zero commits ahead and zero behind. No pull was necessary. Working tree was clean before adding this plan. This commit contains the recent remote merge and Vishwa's integration review.

- Rahul's selected `best_multitask.pt`, quality mapping and matching vocabularies now exist in the merged tree.
- Pasan's selected checkpoint is now `experiments/pasan/checkpoints/phase3_temporal_transformer_weighted_best.pt` (confirmed present locally). Its training cell in `03_phase3_transformer_development.ipynb` uses `HarmonyTemporalTransformer`, with root and chord-family predictions and no genre conditioning. Recover its exact architecture, label order, temporal settings and preprocessing; require strict weight loading. Earlier inspection of the function3 checkpoint does not establish this checkpoint's metrics or compatibility.
- Praveen's requested `BPDR_Workspace/results/model_recovery/notebook_training/2759404560ab4b559a633659d316bec0/selected_model.joblib` is absent at the checked local path. His recovery loader also requires evidence/declaration assets and validation. Obtain the complete owner-exported inference bundle; do not bypass these checks or substitute the older Torch runtime for this joblib model.
- Vishwa has a working pYIN baseline and a limited 13-class CNN/hybrid feasibility model. The latter does not generate complete absolute-pitch notes and is not a demonstrated improvement.

## Chosen implementation approach

Use Streamlit for the local frontend, Plotly for linked interactive timelines, and existing Matplotlib output where appropriate. Native audio players and downloadable JSON/MIDI/WAV keep the interface simple. Use the current Python/uv installation with application-specific dependency declarations; isolate model runners where dependency/import conflicts exist. Inspect compatibility before installing. Do not blindly sync the root environment or overwrite its lock.

Avoid a React/API build for this deadline. The orchestration layer should be callable independently of Streamlit so inference can be checked first and the frontend remains replaceable.

## Planned folder layout

```text
pp1_demo/
  IMPLEMENTATION_PLAN.md          # This plan
  README.md                      # Exact setup/start/rehearsal instructions
  pyproject.toml                  # App dependencies and selected compatible versions
  .gitignore                     # runtime/, caches, temporary artifacts
  app.py                         # UI entry point
  orchestrator.py                # Ordered stage execution and failure propagation
  contracts.py                   # Validated canonical input/output structures
  config/                        # Checkpoint paths, supported options, presets
  adapters/
    bpdr.py
    vishwa.py
    pasan.py
    chord_label_bridge.py
    rahul.py
  workers/                       # Isolated interpreter entry points as needed
  components/
    shared_ui.py
    timeline.py
    audio.py
    status.py
  panels/
    vishwa/                      # Implemented explanations and graphs
    praveen/                     # Owner content templates, explicitly unfilled
    pasan/                       # Owner content templates, explicitly unfilled
    rahul/                       # Owner content templates, explicitly unfilled
  rendering.py                   # Synthesized melody/chord/combined playback
  tests/                         # Contract, adapter and timeline checks
  runtime/                       # Ignored uploads, manifests and generated media
```

Separate presentation content from executable adapters. A teammate should be able to edit their text/figures without changing another person's inference. Avoid embedding machine-specific Windows paths as defaults; resolve repository-relative paths with optional external overrides.

## Screen design

### A. Input and run controls

- Upload MP3/WAV or select a verified bundled/local demo example.
- Original audio player, duration, waveform and file details.
- Genre from supported saved vocabularies; key, BPM, meter and first-beat offset controls where needed. Rahul’s mode is fixed to `major` for every demo run, displayed as “Major — temporary demo assumption”. Keep any upstream mode setting separate.
- Explicit provenance chips: User supplied, Estimated, Rule based, Model predicted, Saved demonstration.
- Run pipeline, stage progress, reset, run identifier and concise error messages.
- Reject unsupported duration or let the user explicitly select a crop; preserve crop offset and show it. Begin with short solo vocals within the selected BPDR policy.

### B. Pipeline overview

Show four component cards with input/output counts, elapsed time and status: Not run, Running, Complete, Bypassed, Failed, Saved result. Use connecting arrows and a shared time axis. A separate small bridge card explains conversion of Pasan’s predicted root/family into Rahul-compatible chord labels.

### C. Member panels

Each member gets the same tabs:
1. Input & output — generic actual runtime artifacts.
2. Processing steps — owner-authored narrative/diagrams.
3. Visual evidence — owner graphs linked to run artifacts or historical experiments.
4. Decisions & experiments — why choices were made and what alternatives showed.
5. Completed / missing / next — explicit capability checklist.

Vishwa tabs are filled. Other narrative/evidence/checklist sections show “Owner content pending” and concise editable prompts. Do not invent teammate achievements. Generic runtime panels may show their actual model results even while narrative tabs are empty.

### D. Final output

Original audio, extracted melody, basic chord timeline, refined chord timeline and synthesized accompaniment. Offer original-plus-accompaniment playback and downloadable JSON/MIDI/audio. Render all streams on one timeline. Clearly identify synthetic instrument sounds; these are not recovered accompaniment stems.

## Vishwa panel: content to implement

1. Purpose: turn a solo vocal recording into structured melody information.
2. Audio preparation: original versus prepared waveform; original/processed sample rate and channel count; peak normalization; trimming state and offset.
3. Features: Mel spectrogram and pYIN F0/voicing, with sensible silence masking and labelled units.
4. Notes: piano-roll timeline, note table (pitch/onset/offset/duration), synthesized melody playback and pitch-class-duration distribution.
5. Model comparison: default working pYIN output used downstream. Optional actual CNN/CNN–Transformer pilot inference shown separately as rest/pitch-class probabilities or labels; never turn those into claimed complete note transcription. If local checkpoint loading fails, show availability status rather than simulated predictions.
6. Historical research: fixed 20 TRAIN/10 VALID cohort, majority/pYIN/CNN/hybrid score-class accuracies 12.10/48.55/42.53/39.45%; show metric meaning and cohort. Keep separate from the current upload. Strict note-F1 results use a different task and metric.
7. Label review: eight user delay observations and their preliminary status, with first-case octave disagreement unresolved. Do not apply these values to arbitrary uploads.
8. Completed: data audit, preprocessing, baseline, evaluation, initial learned prototypes and listening tools. Missing: verified performed-pitch labels, demonstrated learned improvement, CFP, validated beat/key/phrase heads, learned harmony embedding. Next: targeted timing verification and justified model development.
9. Show actual stage/checkpoint/provenance; unsupported JSON fields remain null.

## Data contracts and bridges

Use seconds from the original selected recording as the common time reference. Maintain explicit source/crop offset and resampling metadata. Audio identity is a hash; cache identity includes audio hash, checkpoint hashes, settings and adapter version.

Canonical note: pitch_midi, onset_seconds, offset_seconds, confidence_kind/value, source.
Canonical harmonic segment: start_seconds, end_seconds, function, score provenance.
Canonical chord: start_seconds, end_seconds, label, source, optional candidates.

- BPDR: consume only the owner's accepted handoff. Keep repaired previews separate if the component does not authorize them for downstream delivery. A bypass is explicit and uses the original working WAV.
- Vishwa: map onset_s/duration_s/pitch_midi into canonical notes. Keep model probability semantics visible.
- Pasan: recover exact notebook input features, relative pitch normalization, beat/meter fields, genre/mode handling and label ordering for the named checkpoint. Do not reuse incompatible phase4 code merely because its output classes have similar names.
- Chord-label bridge: decode the selected temporal model’s root and family predictions using its original label ordering and key-relative conventions. Convert these into the exact chord vocabulary expected by Rahul. Document unsupported labels and explicit mapping rules. This replaces the previous T/PD/D-to-chord proposal; do not fabricate a function-only output or choose fresh chords where the model already predicts them.
- Rahul: use his existing model loader and matching vocabularies, build valid full melody/context/temporal inputs, and preserve dynamic padding masks. Call his inference method without reference chord targets. Always supply mode `major`, resolved through his saved vocabulary rather than a guessed numeric ID. Record `mode_source=demo_constant` and show “Major — temporary demo assumption” in the UI and exported run metadata. Do not claim mode detection or silently overwrite upstream mode information. Label raw scores correctly and do not invent unsupported vocabulary IDs.
- Missing key/tempo/downbeats: prefer operator-supplied settings over hidden defaults. A regular grid derived from supplied BPM/meter is labelled “User-configured grid”, not detected beats. Validate whether the chosen recording yields sufficient model windows.

## Milestones in execution order

### M0 — update and inventory

Fetch/check the user's feature branch and merge newer remote commits while retaining local commits. Check a clean working tree; no unrelated branch changes. Inventory the named checkpoints and companion assets. Record dependency versions and missing assets.

Done when: current branch status is known, the three model bundles are either located or explicitly reported missing, and the app plan is agreed by implementation scope.

### M1 — adapter feasibility before frontend polish

Load each selected model in its compatible runtime, using strict architecture/vocabulary checks. Build a valid small synthetic contract fixture to check tensor shapes and decoding; distinguish this from a real research inference. Then run a real short audio example where possible. BPDR remains explicitly unavailable/bypassed until the required valid bundle is present.

Done when: available models load and infer; exact input fields, expected outputs and failure modes are recorded. No training dataset shards are required merely for inference if the complete preprocessing specification is available.

### M2 — Vishwa vertical slice

Implement upload/decode → preprocessing → features → notes → plots/playback/downloads. Add the populated Vishwa tabs and research summaries. Run existing baseline without retraining. Add actual pilot-model comparison only after verifying its checkpoint/features.

Done when: one selected WAV and one MP3 decode path produce real outputs with synchronized visualizations and a complete manifest.

### M3 — downstream integration

Build key/beat inputs, execute the selected Pasan temporal checkpoint, decode root/family predictions into basic chords, construct Rahul inputs with fixed demo mode `major`, and run refinement. Support common timeline conversion, nonfinite/empty-note rejection and explicit partial-window handling.

Done when: a suitable real clip reaches refined chord predictions without feeding reference chord targets into inference. Every input has provenance.

### M4 — member-panel shell and final rendering

Create four consistent panels; finish Vishwa's content, leave owner-editable placeholders for the other narratives. Show actual generic stage results. Render basic/refined chord previews and mixed playback with conservative peak protection.

Done when: each stage has a visible input/output, meaningful status and inspectable artifacts; placeholder sections cannot be confused with completed work.

### M5 — rehearsal and fallback

Choose one short solo vocal primary example and a backup. Execute full runs, record actual latency, verify replay from exact saved run metadata, and test startup/failures. Keep live and saved-result modes clearly distinct. If a stage cannot run, display its reason and require an explicit bypass/demo mode rather than fabricating output.

Done when: one documented start command launches the app; the primary demonstration is rehearsed; cached fallback is labelled and tied to its original recording/settings; teammate source diffs are empty.

## Verification scope

Check model loading, feature/vocabulary compatibility, timestamp and crop conversion, mask lengths, positive durations, missing asset handling and declared source preservation. Use meaningful small tests for these high-impact seams. Verify frontend upload/playback/controls with a browser and inspect the visible state. Do not rerun every teammate's training notebook or claim their old scores were reproduced.

## Deadline priorities

Must: actual Vishwa processing/visuals, expandable panel structure, explicit metadata controls, named-model loaders where assets permit, downstream chain, clear fallbacks and one rehearsed example.
Should: actual Vishwa learned-model comparison, top-k chord display, shared zoom and JSON/MIDI downloads.
Defer: new training, new datasets, CFP development, automatic source separation for arbitrary commercial songs, production deployment, authentication, sophisticated instrument rendering and exhaustive benchmark evaluation.

## Remaining external dependency

Obtain Praveen's complete selected recovery-model export including the companion evidence/declaration files required by its loader. A Windows path in a message does not supply those files to this Mac. If unavailable before PP1, show BPDR unavailable/bypassed and demonstrate the rest with a valid solo recording; do not label that as successful BPDR inference.

## Decision log — 22 October 2026 update

- User selected `phase3_temporal_transformer_weighted_best.pt` in place of the function3 latest checkpoint. File presence and associated notebook training configuration were inspected; no inference or new metric evaluation was performed.
- Revise the downstream adapter to decode chord roots/families, then translate to Rahul’s vocabulary. Exact compatibility remains an implementation prerequisite.
- User requested a dummy `major` mode for Rahul on every demo run. Keep this explicit in the interface and provenance; it is not a model prediction and may be inconsistent with minor-mode recordings. Prefer a major-mode rehearsal example.
- Changes are limited to this application plan. Teammate code and checkpoints remain unchanged.
