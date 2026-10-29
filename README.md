# Genre-aware Vocal Melody Harmonization

**J26-DS-339 · Data Science · SLIIT Faculty of Computing**

A research project that investigates how solo singing or humming can be transformed into musically coherent, genre-aware chord accompaniment. The system separates vocal validation, melody understanding, harmonic progression planning and detailed chord refinement so that each stage can be developed and evaluated independently.

The repository includes component experiments, selected trained checkpoints and **Harmony Lab**, the local PP1 demonstration that connects all four components. The current demo combines live inference with explicitly identified saved musical context. It is a research prototype rather than a fully validated, unrestricted vocal-to-music product.

## Start here

- **Run and interpret the demonstration:** [Harmony Lab demo guide](pp1_demo/README.md).
- **Understand the research:** [Team](#team-and-supervision), [components](#research-components) and [PP1 progress](#research-progress-at-pp1).
- **Continue component work:** follow the [repository map](#repository-map) and the component-specific documentation linked below.

## Problem and intended users

A melody can support many valid harmonizations. Producing a useful accompaniment requires more than selecting locally compatible chords: the progression should remain coherent over time and suit the intended genre. Beginners may lack the theory needed to explore these choices, while experienced songwriters and producers still spend time trying alternatives.

The project targets singers, songwriters, composers, arrangers, producers and beginner musicians. Its research focuses on preserving the performed melody, representing musical context, and separating high-level harmonic planning from detailed chord realization. Multiple alternative, genre-specific progressions are part of the overall research direction; the current demo shows one connected prediction path with per-beat refinement alternatives.

## Team and supervision

| Member | Student ID | Research responsibility |
| --- | --- | --- |
| Vishwa Perera | IT23439078 | Harmony-oriented melody understanding |
| Pasan Perera | IT23148086 | Genre-adaptive functional chord progression generation from melody |
| Praveen Liyanage | IT23141506 | Vocal input validation and minimal musical repair using Boundary-Pitch Disentangled Learning |
| Rahul Jayakody | IT23157132 | Context-aware chord refinement for genre-aware melody harmonization |

**Supervisor:** Prof. Dilshan De Silva

**Co-supervisor:** Prof. Samantha Thellijagoda

Team details and research summaries follow *Progress Presentation 1*, dated **23 October 2026**. The team slide and Rahul's component footer identify his student ID as IT23157132; a conflicting header on his component title slide is treated as a presentation typo.

## Research components

### 1. Vocal input validation and minimal musical repair — Praveen

**Objective:** detect and repair small local pitch errors despite uncertain note boundaries, while preserving acceptable singing, timing, natural pauses and expression.

Boundary-Pitch Disentangled Learning (BPDR) investigates training views that distinguish boundary perturbations from actual pitch changes. The workspace includes audio validation, diagnostic reporting, repair experiments, comparison playback and an accepted-audio handoff.

**Current demo:** the selected notebook-trained ExtraTrees model runs through its guarded exploratory runtime. Technical acceptance can forward the original recording unchanged; a separate repair preview is available only when the runtime produces one. The selected model has not passed its research/generalization gates, so acceptance must not be read as proof of accurate pitch correction.

[BPDR workspace](experiments/bpdr_input/BPDR_Workspace/README.md) · [selected inference export](experiments/bpdr_input/BPDR_Workspace/exports/selected_run_inference/README.md)

### 2. Harmony-oriented melody understanding — Vishwa

**Objective:** transform monophonic vocal audio into a structured musical representation for harmonization, investigating melody, pitch, beat, tempo, key, phrase boundaries, harmony features and embeddings.

The implemented work includes a HumTrans quality audit, a pYIN note-extraction baseline, Mel features, and CNN and CNN–Transformer weak-score classifiers. Portable checkpoints and preprocessing provenance support reproducible inference.

**Current demo:** pYIN extracts octave-aware note events from BPDR's accepted audio. Both trained classifiers execute in parallel and predict 13 rest/pitch-class categories. They do not predict octaves or note boundaries and do not supply the downstream notes. Validated learned beat/key/phrase heads, CFP features and harmony embeddings remain research goals.

[Research workflow](docs/vishwa/README.md) · [experiment record](docs/vishwa/research_progress.md) · [portable checkpoint bundle](experiments/vishwa/checkpoints/weak_score_pilot/README.md)

### 3. Genre-adaptive functional chord progression generation — Pasan

**Objective:** predict musically meaningful progressions from melody, comparing detailed chord prediction with higher-level harmonic-function planning and investigating genre adaptation.

The research includes melody–chord alignment, key normalization, quality filtering, Transformer variants, temporal attention and compatibility-based experiments. Harmonic-change timing remains a central challenge.

**Current demo:** the selected Phase 3 Temporal Transformer, epoch 9, consumes sixteen complete bars of melody and beat context and predicts a relative root and coarse chord family for each beat. Its model definitions are loaded from the matching notebook without executing training cells. **This particular checkpoint is not genre-conditioned**, even though genre-aware modeling is part of the wider research programme.

[Development notebook](experiments/pasan/notebooks/03_phase3_transformer_development.ipynb) · [melody extraction experiments](experiments/pasan/melody_extraction/README.md)

### 4. Context-aware chord refinement — Rahul

**Objective:** refine basic progressions into detailed chords using melody, harmonic context, genre and temporal information.

The research compares baseline, class-weighted and multi-task Transformers over beat-level musical sequences. The custom architecture described at PP1 uses two encoder blocks with four attention heads.

**Current demo:** a selected multi-task checkpoint consumes Pasan's predicted progression and matching musical context in groups of up to four bars. It predicts full chord labels over 402 classes and a separate auxiliary quality head over 13 classes. Displayed top-three candidates use the full-chord softmax directly. The auxiliary head does not re-score them. Refinement mode is temporarily fixed to major.

[Chord-refinement workflow and checkpoint loader](experiments/rahul/README.md)

## Pipeline and current implementation

```text
Solo vocal excerpt
  -> Praveen: validate audio and assess local pitch repair
  -> Vishwa: extract live pYIN notes; run CNN and CNN–Transformer experiments
  -> Context bridge: combine live crop notes with saved surrounding notes/beats/key
  -> Pasan: predict relative roots and chord families over sixteen bars
  -> Rahul: refine predicted chords using musical context and genre
  -> Render: synthetic melody, chord MIDI and accepted vocal + accompaniment WAV
```

The context bridge and audio renderer are integration utilities, not additional member research components. The demo uses the predicted harmonic plan rather than reference chord labels as refinement input. Output timelines are clipped back to the playable vocal excerpt.

| Demonstrated now | Research or product work still to establish |
| --- | --- |
| Four connected components with inspectable handoffs | Generalization to unrestricted recordings and unfamiliar singers |
| Live melody extraction and parallel learned pitch-class inference | Learned-model improvement and reliable performed-note boundaries |
| Saved beat/key context and predicted chord progressions | Validated joint beat, key, phrase and harmony representation learning |
| Genre input to the refinement model | End-to-end genre control across the selected planning/refinement pair |
| Synthetic chord accompaniment, MIDI and provenance downloads | Validated musical preference, diversity and downstream benefit |

## Research progress at PP1

These are **historical component results reported for PP1**, not accuracy scores for a new demo run. Different tasks, cohorts and metrics must not be ranked as if they measured the same outcome.

| Component | Data and evidence | Reported result and interpretation |
| --- | --- | --- |
| Vocal validation / BPDR | Audio validation, diagnostic reports and experimental pitch-repair previews implemented | Reliable correction on unfamiliar singers, boundary robustness, independent listening evaluation and downstream benefit remain pending. |
| Melody understanding | HumTrans audit; fixed feasibility cohort of 20 TRAIN and 10 VALID recordings | Recording-macro weak score-class accuracy: majority 12.10%, pYIN 48.55%, CNN 42.53%, CNN–Transformer 39.45%. These are pitch-class results, not note-transcription F1; pYIN remains the downstream baseline. |
| Harmonic progression | 1,603 songs processed; 1,069 retained songs and 8,909 sixteen-bar windows; 11+ Transformer variants | 31.60% exact chord accuracy on the full Phase 3 validation set. Separately, 63.06% function accuracy and 54.43% Macro-F1 on a compatibility-selected subset. The subset result is not a full-validation result. |
| Chord refinement | 525 tracks, 14,522 sequences and 212,169 beat events | 80.38% exact chord accuracy for the multi-task model; 94.17% top-three accuracy for the baseline. These belong to different model configurations. Major-seventh refinement remains difficult. |

The presentation reports two recordings completing the connected pipeline and 12 tests passing at that milestone. Those observations demonstrate integration, not model accuracy or a formal user study. The presentation also lists progression-to-realization integration as pending in Pasan's component table; the checked-in demo now contains that adapter handoff. Its existence does not settle the research evaluation of the interface.

## Run and use Harmony Lab

Run commands from the **repository root**. On the current local machine, that folder is `J26-DS-339` inside the supplied `Project` directory.

With the existing configured environment and local dataset assets:

```sh
./pp1_demo/start.sh
```

Open [Harmony Lab](http://127.0.0.1:8765), select a vocal from the library, listen to the original, choose a refinement genre and click **Run pipeline**. Inspect each component, compare the extracted melody and harmonic plan with the combined output, then download **Run JSON**, **Refined MIDI** or **Combined WAV**. Stop the server with **Ctrl+C**.

For a fresh environment, install Python 3.12 and `uv`, then use the demo-specific setup:

```sh
uv venv --python 3.12 .venv
uv run --no-sync python pp1_demo/tools/setup_demo.py --install
uv run --no-sync python -B -m pp1_demo.app
```

The setup installs pinned inference dependencies and an isolated BPDR runtime. It does not download the Jamendo recordings or regenerate extracted context. A fresh clone therefore needs the local assets described in the [demo guide](pp1_demo/README.md#required-local-assets) before the full library application can start. Use this demo-specific route to preserve its tested dependencies; ordinary `uv sync` can remove its extra inference packages.

**How to interpret it:** listen for melodic alignment and musical compatibility, inspect where predictions change, and compare alternatives. A completed run means the pipeline executed. Softmax percentages are uncalibrated model scores, technical acceptance is not a singing-quality rating, and saved key/tempo fields are not live predictions. The [demo guide](pp1_demo/README.md#interpreting-the-results) explains every view and the fixed-major/mixed-context limitations.

## Repository map

| Location | Purpose |
| --- | --- |
| [`pp1_demo/`](pp1_demo/README.md) | Local Harmony Lab website, adapters, orchestration, rendering and verification tools |
| [`experiments/bpdr_input/`](experiments/bpdr_input/README.md) | Praveen's BPDR workspace and selected inference export |
| [`src/melody_understanding/`](src/melody_understanding/) | Shared melody-understanding implementation, including portable score-model architecture |
| [`experiments/vishwa/`](experiments/vishwa/) | Melody experiments, evaluation outputs and selected checkpoints |
| [`experiments/pasan/`](experiments/pasan/) | Harmonic-planning notebooks, checkpoints and extraction experiments |
| [`experiments/rahul/`](experiments/rahul/README.md) | Context preparation, refinement models and matching vocabularies |
| [`docs/vishwa/`](docs/vishwa/README.md) | Melody research decisions, target policy and experiment history |
| [`docs/music-theory/`](docs/music-theory/README.md) | Music-theory reference material |
| [`scripts/`](scripts/) / [`tests/`](tests/) | Research preparation, training, diagnostics and checks |
| [`datasets/`](datasets/) / [`notebooks/`](notebooks/) / [`papers/`](papers/) | Local data organization, research notebooks and literature |
| [`UV.md`](UV.md) | General project dependency-management guidance; use the demo-specific instructions above for Harmony Lab |

Selected inference checkpoints and matching vocabularies are tracked in the repository. Dataset audio, training shards and most generated artifacts remain local. Do not assume a notebook can be rerun solely because its checkpoint is present. Preserve matching preprocessing, model definitions and evaluation scope when comparing experiments.

## Verification

With demo assets installed:

```sh
.venv/bin/python -B -m unittest discover -s pp1_demo/tests -v
.venv/bin/python -B pp1_demo/tools/rehearse.py --track track_0000382
```

The second command requires that particular recording to be available. The [demo guide](pp1_demo/README.md#verification-and-rehearsal) also covers an HTTP integration check and a standalone melody check for machines without the full Jamendo library. These are functional checks; independent musical evaluation remains separate.

## Documentation basis

Project scope, member details and PP1 milestones come from *PP1 - Presentation.pdf*, Progress Presentation 1, 23 October 2026: team (slide 2), system design (slide 6), component summaries (slides 8–20). Runtime behavior is documented from the checked-in `pp1_demo` implementation. When the presentation describes an objective that is not implemented by the selected checkpoint, the distinction is stated above. The presentation is not bundled by this documentation change.
