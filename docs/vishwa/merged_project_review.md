# Merged project review — 22 October 2026

Scope: inspect the merge at `67c84e0` against Vishwa's previous `37f5d54`. Source, component READMEs, saved notebook outputs and installed package metadata were inspected. This is a component/integration assessment, not a line-by-line correctness review of every merged file. No teammate experiments were rerun or checkpoints loaded. Saved results below are historical evidence, not fresh reproduction.

## Repository map for VS Code

| Folder/file | Responsibility | Start here |
|---|---|---|
| `src/melody_understanding`, `scripts`, `notebooks/vishwa` | Vishwa: vocal audio to melody, evaluation and label review | `docs/vishwa/README.md` |
| `experiments/bpdr_input` | Liyanage L. N. P., IT23141506: technical vocal-input checks and experimental Boundary-Pitch Disentangled Repair | Component README, then `src/bpdr_input/pipeline.py` and `contracts.py` |
| `experiments/pasan/melody_extraction` | Audio extraction/comparison infrastructure for multiple existing melody tools | README, `models.py`, `pipelines.py`, `artifacts.py` |
| `experiments/pasan/notebooks` | Melody-to-chord and harmonic-function model development | Read numbered notebooks chronologically; inspect reusable `phase2_harmony` and `phase4_harmony` modules |
| `experiments/rahul` | Chord-recognition references and refinement training-event preparation | README, `config.py`, `src/build_training_events.py` |
| `datasets/mtg-jamendo` | Metadata, extraction code and supporting artifacts | Distinguish metadata/extraction outputs from original audio and human labels |
| `papers`, `docs/music-theory` | Literature and learning resources | Supporting references |
| `system.md` | Proposed system narrative | Architecture proposal; not an inventory of verified capabilities |

The comparison contains 1,869 changed paths, including large generated artifacts. Counts do not measure research completion. No differences were found in Vishwa's source/notebook/script/test/document paths between the pre-merge commit and merge HEAD. There is no tracked `.vscode` configuration or `.code-workspace` found in the inspected repository. VS Code uses this same checkout; it is not a separate project copy.

## BPDR input component

Implemented source includes original preservation, WAV/FLAC checks, duration limits, channel/sample-rate handling, frozen PESTO analysis, accepted-package validation and diagnostic reports. The active technical path accepts 0.25–20-second clips and exports checked mono 22050 Hz PCM24 audio. Output status controls whether accepted audio exists. `validate_package()` checks checksum, sample metadata and finite/range validity before downstream use.

The README distinguishes technical acceptance from musical repair. It documents training scaffolds, controlled renderer/controller experiments and a historical total of 147 regression checks. It explicitly reports zero research-ready examples and no accepted repaired package; reviewer approval and renderer validity remain open. Those reported runs are not reproduced here, and the component's local `runs` folder is absent in this checkout.

Integration issue: some HumTrans recordings exceed the 20-second input policy. Do not silently crop them to satisfy it. Your own audio preparation also resamples/normalizes; define ownership of transformations and preserve source timing when consuming accepted BPDR output.

## Pasan: extraction infrastructure

Common structures include note onset/offset/pitch/confidence/source, frame F0 and voicing, key estimates and provenance. Outputs include MIDI, Parquet/CSV notes, NPZ pitch tracks, preview audio and metadata. Adapters exist for pYIN, MLX separation, Basic Pitch, GAME, RMVPE, ROSVOT, Essentia and SheetSage. Adapter presence does not prove a model is installed or evaluated on this machine.

Useful reuse: artifact schemas, listening previews, cache/provenance handling and optional independent transcription comparisons. These automatically extracted outputs are pseudo-labels; they do not resolve your HumTrans octave/timing uncertainty by themselves. Global-key estimation from accompanied mixes also does not directly validate key estimation on solo humming.

## Pasan: harmony experiments

Saved notebooks cover initial melody-to-chord training, dataset audits, key/label normalization, beat alignment, autoregressive and direct Transformers, class weighting, temporal/change-aware models, an RL comparison and a BiLSTM harmonic-function baseline. Checkpoints exist locally, but their compatibility and inference have not been verified in this review.

Examples of saved evidence:
- Phase 2 dataset audit prints a snapshot of 373 songs, 4,476 windows and 133 artists. The referenced generated snapshot path is machine-specific; it is not proof the same snapshot exists locally.
- `04_phase3_improved_transformer_development.ipynb` reports validation joint accuracy 0.3395 for supervised Flat 84 versus 0.3418 after RL, while change F1 falls from 0.1280 to 0.1241. This is a small mixed metric result, not a general superiority claim. Its output states TEST was not accessed.
- Different notebook stages use different targets and datasets; their scores cannot be combined into a single project accuracy.

The maintained `phase4_harmony/model.py` exposes a Transformer or local MLP option, optional genre conditioning (off by default), mode embeddings and three harmonic-function logits plus change predictions. `phase4_harmony/predict.py` emits TONIC/PREDOMINANT/DOMINANT segments, not unique Roman-numeral chords or refined accompaniment. Older stages predict other chord vocabularies. Confirm the selected stage/checkpoint before integration.

## Rahul: refinement data preparation

Source implements LV-Chordia extraction from mixed audio, chord normalization/canonicalization, pYIN melody extraction, detected beats, global key and genre metadata. A heuristic basic harmonizer selects diatonic triads on two-beat windows; basic chords are then associated with one-beat model events. Reference chords are introduced after input-side features are generated, reducing target leakage in that construction.

The README describes a ten-track production pilot builder and explicitly states that the refinement Transformer is not implemented/trained. Detected chord references are model outputs, not manually certified labels. Bar/downbeat fields remain null where downbeats have not been estimated. The generated `outputs/aligned` folder is absent locally, so historical pipeline descriptions are not evidence of a currently runnable prepared dataset here.

Some early README wording says basic chords will come from another component; the later workflow and source include the heuristic builder. Distinguish early milestone text from the later implementation.

## The actual interface gap

Pasan's phase4 predictor expects:

| Required field | Vishwa currently has | Work needed |
|---|---|---|
| `duration_sec` | `source_duration_s` | Explicit schema mapping |
| `notes[].onset_sec` | `onset_s` | Rename while preserving original timeline |
| `notes[].offset_sec` | `onset_s` + `duration_s` | Compute end time |
| `notes[].midi_pitch_rounded` | `pitch_midi` | Explicit mapping; retain provenance |
| Key `tonic_pc` and major/minor `mode` | `key: null` | Verified estimator or user-supplied key |
| Beat `time_sec`, bar index, beat-in-bar and meter | Beats/tempo absent | Establish beat/downbeat/meter information; never invent it |
| Compatible checkpoint + snapshot metadata | Some checkpoints present | Identify matching assets and validate loading/inference |
| Genre list | No current genre input in baseline output | Pass user-selected supported genre with explicit semantics |

The extraction NoteEvent class itself uses `midi_pitch`; phase4 cleaning expects `midi_pitch_rounded`. Even teammate representations need stage-specific mapping. `complete_windows()` accepts complete bars in supported meters and requires a real following boundary. A list of beat timestamps alone is insufficient. Partial windows are not emitted. Short humming clips may yield no output under the chosen window size.

## Environment after the merge

Root pyproject adds MLX separator, SheetSage, pandas, PyArrow, mir_eval, pretty_midi and tqdm; dev adds ipykernel and pytest. Torch was removed as a direct requirement but remains transitive in the lock, where the resolved version is 2.13.0. Installed torch is currently 2.14.1. A sync could therefore change the research environment.

Installed root metadata checked: torch 2.14.1, numpy 2.5.4, librosa 1.0.0, mido 1.3.3, pretty-midi 0.2.11.post0, praat-parselmouth 0.4.7. Not installed in this root environment: mlx-audio-separator, sheetsage-infer, lv-chordia, pesto-pitch. Component-local environments may differ and were not exhaustively audited. BPDR owns a separate pyproject/lock and Windows launchers. Some optional extraction tools need separate Python environments/checkpoints.

The repository references a root README.md that is absent. There is no verified fresh-environment install from this review. Do not treat fetching Git files as installing dependencies. No environment sync or downloads were performed.

Checked absent local assets include `datasets/mtg-jamendo/vocal-tagged-audio`, `experiments/rahul/outputs/aligned`, `experiments/bpdr_input/runs`, `phase2_dataset_v2` and one conventional phase4 run path. These checks are specific paths, not a claim that every possible alternate asset directory is absent. Some historical notebook outputs name another user's `/Users/praise/...` checkout.

## Architecture discrepancy to settle

`system.md` proposes melody understanding → functional harmonic planning → chord refinement → post-generation harmony validation. The merged BPDR implementation instead prepares/repairs vocal input before melody understanding. Both cannot be silently credited as the same fourth component. Reconcile the intended team architecture with the actual component ownership before a final integrated demo. The inspected code does not establish a completed post-generation validator.

## Recommended next actions

1. Preserve the present working Vishwa environment; plan compatible component environments before syncing.
2. Select the intended Pasan predictor/checkpoint and its matching metadata, and confirm complete-bar/key requirements.
3. Build a small explicit schema adapter from Vishwa notes to the chosen downstream interface; reject missing required musical fields with clear diagnostics.
4. Continue the eight-case TRAIN timing experiment with original and user-delayed references. Reuse teammate tooling where appropriate, but retain unresolved octave labels.
5. Establish reliable beat/key/meter inputs and the actual team architecture, then run one end-to-end example with provenance.
6. Assess additional datasets only for specific missing supervision. Jamendo metadata is useful, but extracted labels are not ground truth and raw audio is not present at the checked canonical path.

This review changes no teammate implementation and does not claim their historical experiments have been reproduced on this Mac.
