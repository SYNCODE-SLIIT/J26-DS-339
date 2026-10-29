# Harmony Lab — PP1 Demo Guide

Harmony Lab is the local demonstration for **J26-DS-339: Genre-aware Vocal Melody Harmonization**. It runs the four research components on compatible local MTG-Jamendo vocal excerpts and makes their intermediate outputs inspectable.

Use the [project README](../README.md) for member details, research objectives and PP1 progress. This guide covers setup, presentation, interpretation, outputs and troubleshooting.

## What this demo demonstrates

```text
Vocal assessment       Melody understanding        Harmonic planning     Chord refinement
Praveen / BPDR    ->   Vishwa / pYIN + CNNs    ->   Pasan / Transformer -> Rahul / Transformer
                              |
                 live notes inside the vocal crop
                 + saved surrounding notes, beats and key
                              |
                 sixteen-bar context for downstream inference
```

All four components execute. The context bridge and final renderer connect them. The website uses Python's standard-library HTTP server and plain HTML/CSS/JavaScript, so no Node.js, frontend build, Streamlit or hosting account is required for this demo.

The library selects existing dataset vocals; **this interface does not record a microphone or accept arbitrary uploads**. Saved melody outside the audible crop, saved beats and saved key complete the context needed by the downstream models. Learned CNN outputs are shown as experiments while live pYIN notes drive the harmonic models. Generated accompaniment is synthesized instrument audio, not generated harmony vocals or recovered accompaniment stems.

## Quick start with the existing local environment

Open a terminal in the repository root, `J26-DS-339`:

```sh
./pp1_demo/start.sh
```

Open **[http://127.0.0.1:8765](http://127.0.0.1:8765)**. Keep the terminal running. Stop the server with **Ctrl+C**.

To choose another port:

```sh
./pp1_demo/start.sh --port 8766
```

Then open [http://127.0.0.1:8766](http://127.0.0.1:8766). The server binds to `127.0.0.1` only. Once dependencies, recordings and model weights are available, inference and generated artifacts remain local. First use of BPDR's upstream PESTO pitch estimator may require a weights download.

## Set up another machine

Use **Python 3.12** and **uv**, from the repository root:

```sh
uv venv --python 3.12 .venv
uv run --no-sync python pp1_demo/tools/setup_demo.py --install
uv run --no-sync python -B -m pp1_demo.app
```

If `.venv` already exists, omit the first command. The Python module entry point is also suitable for Windows; `start.sh` is the macOS/Linux convenience launcher. The setup script selects the appropriate Windows interpreter paths.

The setup installs `requirements-inference.txt` into the main `.venv`, and the serialization-compatible BPDR dependencies from `requirements-bpdr.txt` into `pp1_demo/runtime/bpdr_env`. It does not retrain models, sync or change the root lockfile, download recordings, or produce their saved extraction packages. Prefer `uv run --no-sync` for these commands so the root project dependency set does not replace the demo's inference environment.

### Required local assets

The full library UI needs both model bundles and dataset context. The dataset layout expected by `config.py` and `dataset.py` is:

```text
datasets/mtg-jamendo/
├── annotations/
│   └── autotagging_genre.tsv
└── melody-chord-extraction-pipeline/outputs/
    └── track_<recording-id>/
        ├── bpdr/
        │   ├── vocal.wav
        │   └── record.json
        ├── melody/
        │   └── combined.json
        ├── beats/
        │   └── beats.json
        └── key.json
```

Transfer compatible existing extraction packages or regenerate them through the existing extraction workflow. Empty files or a copied WAV alone are insufficient: schemas, source offsets, audio checksums and musical context must match. Raw audio is not bundled in Git.

The catalogue requires usable notes and a saved key, sixteen complete covering bars with valid beat structure, and an audio duration between 0.25 and 20 seconds. The familiar presentation examples are approximately 17–20 seconds. Missing or incompatible packages are skipped, so the library count depends on local assets. A discovered recording is not proof that it has been rehearsed end to end.

Keep these model assets and their matching source/mappings together:

| Component | Required bundle / source |
| --- | --- |
| BPDR | `experiments/bpdr_input/BPDR_Workspace/exports/selected_run_inference/` and the BPDR workspace source |
| Melody | `experiments/vishwa/checkpoints/weak_score_pilot/`: `cnn.pt`, `cnn_transformer.pt`, `train_normalization.npz`, `manifest.json`; architecture in `src/melody_understanding/score_model.py` |
| Planning | `experiments/pasan/checkpoints/phase3_temporal_transformer_weighted_latest.pt` and `experiments/pasan/notebooks/03_phase3_transformer_development.ipynb` |
| Refinement | `experiments/rahul/outputs/multitask_transformer/multitask_20261020_164527_051309/best_multitask.pt` with `chord_quality_mapping.json`; vocabularies in `experiments/rahul/outputs/model_preprocessing/final_all_tracks_quality_v6_multibar/` |

Paths are centralized in [`config.py`](config.py). Do not exchange a checkpoint's vocabularies or normalization for those from another experiment.

### Existing BPDR runtime and override

The demo prefers the compatible legacy interpreter when it exists:

```text
datasets/mtg-jamendo/melody-chord-extraction-pipeline/.venv-bpdr/bin/python
```

Otherwise it uses the app-local `pp1_demo/runtime/bpdr_env` environment. To explicitly select a compatible interpreter on macOS/Linux:

```sh
PP1_BPDR_PYTHON=/absolute/path/to/python ./pp1_demo/start.sh
```

In PowerShell, set `$env:PP1_BPDR_PYTHON` to that environment's Python executable before launching the Python module. Match BPDR's NumPy, scikit-learn and joblib versions to the exported manifest; an arbitrary Python interpreter is insufficient.

## Presenting the demo

1. **Choose a recording.** Search the vocal library by recording ID, genre or saved source key. Select an entry and play the original vocal before running anything.
2. **Inspect the source.** Click **About this source** to view crop timing, original key, sixteen-bar context and provenance. The reference major key may differ from the source minor key.
3. **Choose the refinement genre.** Use the genre selector. Its initial value comes from compatible saved tags, with a fallback when necessary. Changing it affects refinement input; the selected planning checkpoint is not genre-conditioned.
4. **Run pipeline.** One run executes at a time. Follow the four component cards. Short pauses display completed handoffs so the audience can follow them; they are presentation holds, not additional model computation.
5. **Look inside each stage.** Select **Vocal assessment**, **Melody understanding**, **Harmonic planning** or **Chord refinement** in the sidebar/cards/tabs. Expand the input and provenance details when explaining what is live and what is saved.
6. **Compare the listening outputs.** Play the extracted melody, harmonic plan and finished demo. Compare the finished mix with the original vocal, particularly at note starts, rests and harmonic changes. Only one audio player plays at a time.
7. **Export the evidence.** Download **Run JSON**, **Refined MIDI** and **Combined WAV**. Keep the run identifier with any observations so the result can be traced to its source, settings and checkpoints.

A useful presentation sequence is one rehearsed recording with its default genre, followed by the same recording with another supported genre. Describe any audible difference as an observation from those runs, not a measured improvement. The demo does not evaluate accuracy against reference chords during inference.

## Interpreting the results

### Vocal assessment — Praveen

- **Measured pitch contour:** PESTO estimates fundamental frequency (F0) in hertz; silence is masked. It is an acoustic estimate rather than a score annotation.
- **Learned correction estimates:** the ExtraTrees model proposes local changes in cents. One semitone is 100 cents. Raw estimates are not automatically authorized edits; the operating policy and review status matter.
- **Accepted audio:** technical acceptance means processing can continue. An `ACCEPT_UNCHANGED` handoff preserves the original pitch and timing. It is not a rating of good singing and does not prove an error was repaired.
- **Repair preview:** when present, this is a separate experimental output. The selected model's research/generalization gates have not passed. Check `forwarded_audio`, `handoff_status` and the accepted-audio artifact to establish what actually reached melody extraction.

### Melody understanding — Vishwa

- **Live melody notes / piano roll:** horizontal position is time, vertical position is MIDI pitch and rectangle width is note duration. MIDI pitch includes octave. Listen for octave jumps, missed notes, over-segmentation and timing mismatches.
- **Voiced fundamental frequency:** pYIN's F0 estimate on accepted audio. Gaps represent masked unvoiced estimates, not automatically a musical phrase boundary.
- **Mel spectrogram:** an 80-band time–frequency representation of prepared mono 22,050-Hz audio. It shows energy patterns, not chord labels.
- **CNN / CNN–Transformer heatmaps:** actual checkpoint probabilities for rest plus twelve pitch classes. A pitch class such as C does not distinguish C3 from C4. These models have no octave or note-boundary head; they are not converted into the downstream note sequence.
- **Historical validation:** the displayed 12.10% majority, 48.55% pYIN, 42.53% CNN and 39.45% CNN–Transformer results concern recording-macro weak score-class accuracy on the development pilot. They are neither current-recording accuracy nor note-transcription F1. The learned models did not beat the matched pYIN baseline.

Before inference, the library preview uses saved notes. After the live stage, pYIN notes replace the entire audible interval, including rests. Saved notes only survive outside that interval. Beats and key remain saved inputs.

### Harmonic planning — Pasan

- **Input melody:** the shared piano roll shows the audible excerpt. Dashed beat lines come from saved extraction. The model receives the full covering sixteen-bar context, not just the visible crop.
- **Roman plan:** each beat receives one of twelve relative roots and one of seven coarse families: `MAJ`, `MIN`, `DOM`, `DIM`, `HDIM`, `AUG`, `SUS`. For example, `♭III · MIN` means a minor-family chord rooted three semitones above the reference tonic. These are relative-root/family predictions, not a separate validated tonic/subdominant/dominant classifier.
- **Root prediction scores:** raw root softmax probabilities. A high percentage means preference among this model's classes, not calibrated correctness or musical approval. Family scores are separate.
- **Canonical handoff:** coarse families map to representative chord symbols before refinement; for example, `DOM` maps to a seventh chord, `HDIM` to half-diminished seventh and `SUS` to suspended fourth. A family does not specify every extension or inversion.

The demo uses the epoch-9 **latest** Temporal checkpoint, not a claim that it is the best checkpoint across all research variants. Genre is not an input to this selected model.

### Chord refinement — Rahul

- **Plan versus refined progression:** compare the Roman plan with the named refined chords at the same times. The model can add detail or select a different label; a changed chord is not automatically an improved chord.
- **Top score and alternatives:** the table shows the first twelve beat predictions. The main label and alternatives come from the 402-class full-chord head. Percentages are raw, uncalibrated softmax scores; top-three candidates are local alternatives rather than three complete alternative progressions.
- **Auxiliary quality head:** thirteen output categories support multi-task learning. This head is not fused into displayed full-chord scores.
- **Unknown inputs:** `UNK` mappings mean a context category was absent from the saved vocabulary. Inspect the JSON details before interpreting those predictions.
- **Major-mode constraint:** mode is always major in this demo. For a saved minor key, the reference tonic is shifted to its relative major. This normalization is not evidence that minor-mode refinement has been validated.

### Listening room

| Player / artifact | Meaning | What to listen for |
| --- | --- | --- |
| The extracted melody | Synthetic playback of live pYIN notes | Note identity, octave, onsets, durations and pauses relative to the original vocal |
| The harmonic plan | Instrumental rendering of predicted roots and coarse families | Progression continuity and harmonic-change timing |
| The finished demo | BPDR accepted vocal mixed with synthesized refined accompaniment | Melody–chord compatibility, timing and audible clashes |
| Separate repair preview, when present | Experimental BPDR processing | Whether a genuine local error changed and whether expression/timing were preserved |

Synthetic timbre, voicing and mixing affect what you hear. The finished mix is an audition aid, not a production master or a formal listening-study result. Some unsupported chord labels may remain visible but cannot be rendered; inspect run warnings before judging silence as a model prediction.

## Live predictions, saved inputs and assumptions

| Item | Source in the connected demo |
| --- | --- |
| Vocal excerpt | Existing local dataset crop; checksum verified before the run |
| Vocal assessment and correction estimates | Live BPDR model execution |
| Notes inside the audible crop | Live pYIN on accepted audio |
| CNN and CNN–Transformer class maps | Live inference using packaged feasibility checkpoints and TRAIN normalization |
| Melody outside the audible crop | Saved extraction from the same track, clipped to the context window |
| Beat grid, saved tempo and source key | Saved extraction metadata; not newly estimated by this demo |
| Genre | Compatible dataset metadata, a default or explicit user selection; provenance records the choice |
| Reference major key | Derived from the saved key, using relative-major normalization for minor sources |
| Basic and refined chord labels | Live planning/refinement inference; reference chord targets are not inputs |
| Refinement mode | Fixed major demo setting |
| Audio/MIDI previews | Synthesized from this run's notes and chord predictions |

This mixed-context design preserves the input structure used by the trained models. It does not demonstrate end-to-end understanding of an arbitrary unseen microphone recording without external musical context.

## Outputs and run provenance

Each attempt is stored under the ignored local directory:

```text
pp1_demo/runtime/runs/<run-id>/
```

| File | Contents |
| --- | --- |
| `manifest.json` | Status, stage results, settings, warnings, model identities, source provenance and artifact hashes on successful completion |
| `input.wav` / `accepted.wav` | Original excerpt and actual audio forwarded by BPDR |
| `repair_preview.wav` | Optional experimental repair preview, only if available |
| `vishwa_prepared.wav` / `vishwa_features.npz` | Prepared audio and extracted melody features |
| `melody.wav` / `melody.mid` | Live pYIN note rendering and MIDI |
| `basic.wav` / `basic.mid` | Predicted coarse chord rendering and MIDI |
| `refined.wav` / `refined.mid` | Refined chord rendering and MIDI |
| `combined.wav` | Accepted vocal plus refined instrumental accompaniment |

The folder also contains BPDR logs/results. Failed runs may contain only the artifacts created before the failure. The UI exposes Run JSON, Refined MIDI and Combined WAV downloads; other registered artifacts remain available in the local run folder and artifact API.

In the JSON, inspect `status`, `error`, `warnings`, `stages`, `settings`, `provenance` and checkpoint hashes first. `context.inference_notes` distinguishes live accepted-audio notes from saved surrounding context. Visible timestamps are relative to the crop; full context coordinates and crop offsets are retained for reconstruction. Stage `seconds` excludes display holds, while total elapsed time can include those holds.

### BPDR runtime isolation

The demo creates a byte-preserving private source snapshot under `pp1_demo/runtime/bpdr_sandbox/`. The unmodified export installer and guarded loader operate on that snapshot, with relocated runtime paths and original hash-linked evidence. Generated files and caches stay under `pp1_demo/runtime/`; teammate source and checkpoint/evidence bytes are not rewritten. The experimental model's failed generalization status remains visible.

## Verification and rehearsal

Run from the repository root with the configured demo environment. These commands create local ignored runtime outputs and do not train models.

```sh
# Unit and integration-contract checks; requires local library/model assets.
.venv/bin/python -B -m unittest discover -s pp1_demo/tests -v

# Rehearse one recording without the website, if that ID is in your library.
.venv/bin/python -B pp1_demo/tools/rehearse.py --track track_0000382

# Optional: run every currently discoverable recording; runtime scales with the library.
.venv/bin/python -B pp1_demo/tools/rehearse.py --all
```

With the website running, open another terminal and run:

```sh
.venv/bin/python -B pp1_demo/tools/check_http.py
```

For an alternate server port, pass `--port 8766`. The HTTP check starts a real run on the first available recording and verifies API responses, audio byte ranges, stage handoffs, downloads and artifact boundaries. It will conflict with an already active run, so wait for the UI to finish first.

For Windows, use `uv run --no-sync python -B` in place of `.venv/bin/python -B`.

### Standalone melody check without the Jamendo library

After installing the inference dependencies, use your own solo vocal file:

```sh
uv run --no-sync python -B -m unittest discover -s pp1_demo/tests -p test_vishwa.py -v
uv run --no-sync python -B pp1_demo/tools/check_vishwa.py --audio /path/to/solo_vocal.wav
```

The result is written to `pp1_demo/runtime/vishwa_standalone/result.json` alongside generated melody-stage artifacts. This path checks pYIN and both packaged melody classifiers. It does **not** run BPDR, harmonic planning or refinement, or add an upload workflow to the full demo UI. HumTrans data and training shards are not needed for this inference path.

## Troubleshooting

| Symptom | Check / action |
| --- | --- |
| `start.sh` cannot find `.venv/bin/python` | Run the fresh-environment setup from the repository root. On Windows, launch the Python module. |
| Missing scientific package after `uv sync` | Re-run `setup_demo.py --install`, then use `uv run --no-sync` for the demo. |
| `No aligned local vocal examples are available` | Check the extraction layout, genre annotations, key, notes, valid sixteen-bar context and duration limit. Dependency installation does not supply dataset assets. |
| `Some model files are missing` / library fails to load | Restore the selected checkpoint bundles and matching vocabularies at the configured paths. The initial presence indicator is not a full integrity check. |
| BPDR interpreter or serialization error | Check `PP1_BPDR_PYTHON`, the selected environment and the versions recorded in BPDR's manifest. The legacy interpreter takes priority if present. |
| First BPDR use cannot load pitch weights | Allow the upstream PESTO download or provide its weights in the runtime's configured cache before an offline rehearsal. |
| Audio checksum mismatch | Restore the WAV and matching `record.json` from the same extraction, or regenerate that package. Do not replace the checksum merely to suppress the error. |
| Checkpoint/hash mismatch | Restore the intact bundle with its matching preprocessing and manifest. Missing/corrupt melody assets stop the stage. |
| Address already in use | Start on another port and use the matching URL and `check_http.py --port` value. |
| A run stops or is rejected | Read the displayed error and run manifest/BPDR log. Downstream stages do not complete after a failed prerequisite. |
| Output seems repetitive or poorly aligned | Inspect the melody first, then saved beats/key, the plan and refinement. A functional run does not establish musical quality or harmonic-change accuracy. |

