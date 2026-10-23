# Harmony Lab — local PP1 demo

A local website that runs **Praveen → Vishwa → context bridge → Pasan → Rahul** on
all locally available compatible Jamendo vocal excerpts (the count depends on local assets). Vishwa extracts melody live and runs both packaged score-class checkpoints. All four
components execute; saved outside-crop features and the fixed major mode are
labelled explicitly.

## Start

From the repository root:

```sh
./pp1_demo/start.sh
```

Open **http://127.0.0.1:8765**. Choose a recording, listen, then press **Run pipeline**.
Search the library by recording ID, genre or key. Click a stage to inspect its inputs, outputs, plots and provenance. Download the
run JSON, refined MIDI or combined WAV from the listening room.

Use `./pp1_demo/start.sh --port 8766` to choose another local port. Stop with Ctrl+C.
The website uses standard-library HTTP plus plain HTML/CSS/JavaScript: no Node,
Streamlit, frontend installation, third-party CDN or hosting account is required.
It binds to loopback only. Audio and checkpoints stay local.

## Existing runtimes

The launcher uses the repository's existing `.venv/bin/python`, with its PyTorch,
NumPy, soundfile, pretty-midi, mir-eval and Rahul preprocessing dependencies.
BPDR runs in the existing compatible interpreter at:

`datasets/mtg-jamendo/melody-chord-extraction-pipeline/.venv-bpdr/bin/python`

Override it with `PP1_BPDR_PYTHON=/absolute/path/to/python ./pp1_demo/start.sh` if
necessary. The BPDR environment's own sklearn/joblib versions take priority;
optional plotting packages can be read from the main project's site-packages.
For a new machine, use the setup below. An app-local BPDR environment is used when the legacy interpreter is absent.

## What runs

1. **Praveen:** verify the complete exported notebook-run bundle; load the owner’s
   guarded experimental ExtraTrees model; run the owner’s audio analysis,
   prediction, operating policy and accepted-audio path. Plot actual F0 and raw
   correction estimates. Technical acceptance is not a musical-quality score.
   A separate repair preview is exposed only if the owner runtime produces one.
2. **Vishwa:** read BPDR accepted audio; prepare mono 22,050-Hz audio; extract
   Mel features, voiced pYIN F0 and note events. Strictly load the packaged CNN and
   CNN–Transformer checkpoints with TRAIN normalization and run both on the live
   Mel frames. Their 13 outputs are rest/pitch class, not octave-aware notes.
   pYIN supplies the downstream note events. Replace all saved notes inside the
   audible crop (including rests), retain clipped outside-crop notes, and preserve
   saved beats/key/genre metadata. Both downstream models consume this mixed-context
   melody. Checkpoints are hash-verified; missing/corrupt assets stop the stage.
3. **Pasan:** strictly load `phase3_temporal_transformer_weighted_latest.pt`, epoch
   9, using the notebook’s original Temporal architecture. Normalize melody to the
   reference major collection; normalize time by context duration; infer root and
   family over **sixteen complete bars**.
4. **Rahul:** map the predicted coarse families into canonical chord inputs and
   build the full musical context with the matching saved vocabularies. Run the
   exact multi-task checkpoint in groups of up to four bars. Mode is fixed to
   **major**; genre defaults to the recording’s saved metadata and can be changed.
   Full-chord top-3 outputs are raw softmax probabilities; the quality head is not
   fused into them.
5. Render live pYIN melody and predicted chords as synthetic instruments, and mix the
   refined accompaniment with BPDR’s accepted vocal. These are not generated
   harmony vocals or recovered accompaniment stems.

### Short audio, full model context

The playable vocals are 17–20 second crops. To preserve the trained Temporal
input format, the demo retains sixteen complete bars of musical context from
around each crop, replacing the audible interval with Vishwa’s live notes. Both Transformers run on that context; plots and playback are
then clipped to the vocal excerpt. Crop offsets are retained in run metadata.
This is a dataset-context demonstration, not a live microphone transcriber.

### BPDR isolation

The export's original installer writes into its component workspace. Instead of
writing there, this demo makes a byte-preserving private source snapshot under
`pp1_demo/runtime/bpdr_sandbox/`. The **unmodified** export installer and guarded
loader run against that snapshot, with relocated model-card paths and the original
hash-linked evidence files. Checkpoint bytes, evidence and teammate source remain
unchanged. All subprocess outputs, model caches and plotting caches are app-local.
The model is still an experimental preview with failed generalization gates;
that status is retained in the UI and run manifest.

## Presentation and outputs

Fast stages briefly hold their **completed** handoff on screen so the audience
can follow the pipeline. Actual model timings exclude this presentation pause.
Saved features and placeholders are never labelled as model predictions.

The website labels stages by component rather than team member. Both Transformer
views include the shared input-melody piano roll. Harmonic planning displays the
notebook's 12 relative Roman roots and seven chord families (for example,
`♭III · MIN`); chord refinement shows that Roman plan alongside its predicted
chord names. This is a presentation change: model inference, canonical chord
handoffs, synthesized audio and downloaded run data keep their original formats.

Each run is stored at `pp1_demo/runtime/runs/<run-id>/` with its manifest, input and
accepted audio, BPDR log/results, synthesized WAVs and MIDI files. These files are
ignored by Git. The UI serves only selected dataset audio and registered artifacts.
One run executes at a time; rejected or failed audio stops the chain.

## Verification

```sh
.venv/bin/python -B -m unittest discover -s pp1_demo/tests -v
.venv/bin/python -B pp1_demo/tools/rehearse.py --track track_0000382
# With the website running:
.venv/bin/python -B pp1_demo/tools/check_http.py
```

The original six library examples were rehearsed with real BPDR, Pasan and Rahul inference.
The initial backend rehearsals completed in approximately 6–7 seconds per recording
on this machine, excluding presentation holds. All were accepted unchanged by
BPDR and produced excerpt-aligned chord predictions and audio/MIDI artifacts.
This is a functional integration check, not model-accuracy validation.

### Expanded vocal library

The library discovers all local vocal crops with a usable saved key, melody and
sixteen-bar beat context within BPDR’s duration limit. It lists the compatible locally available
recordings, including major and minor source keys; Rahul’s demo mode remains major.
Search by ID, genre or key. Detailed note/beat arrays and waveforms are fetched only
for the selected recording. Missing or incompatible extraction packages are skipped.
The expanded catalogue is not a claim that every recording has been rehearsed.

## Reproduce Vishwa on another machine

The small inference bundle is tracked at
`experiments/vishwa/checkpoints/weak_score_pilot/` (about 211 KB).
It includes **cnn.pt**, **cnn_transformer.pt**, **train_normalization.npz**, a hash
manifest, training configuration and evaluation provenance. The portable architecture
is `src/melody_understanding/score_model.py`. No HumTrans data or training shards
are needed for inference. These are existing five-epoch feasibility checkpoints,
not newly trained or production-validated models.

Use Python 3.12 and uv. From the repository root (macOS, Linux or Windows):

```sh
uv run --no-sync python pp1_demo/tools/setup_demo.py --install
uv run --no-sync python -B -m unittest discover -s pp1_demo/tests -p test_vishwa.py -v
uv run --no-sync python pp1_demo/tools/check_vishwa.py --audio path/to/solo_vocal.wav
uv run --no-sync python -B -m pp1_demo.app
```

If no environment exists yet, run `uv venv --python 3.12 .venv`, then
`uv run --no-sync python pp1_demo/tools/setup_demo.py --install`.
The setup uses uv to install the app's pinned scientific requirements into `.venv`
and creates a separate BPDR runtime under ignored `pp1_demo/runtime/bpdr_env`.
It does not sync or edit the root lockfile, download recordings, or retrain models.
The first BPDR pitch-estimator use may download its upstream PESTO weights.

**Dataset requirement for the existing library UI:** each recording still needs
`bpdr/vocal.wav`, `bpdr/record.json`, `melody/combined.json`, `beats/beats.json`,
`key.json`, and matching genre annotations in the existing Jamendo extraction
layout. Transfer or regenerate those locally through the existing extraction
pipeline. They are not bundled with model code and raw audio is not committed.
The standalone Vishwa check accepts your own audio without that dataset.

Vishwa appears in the sidebar, stage timeline and inspection panel. It shows the
live piano roll, F0, Mel spectrogram, both trained-classifier probability maps,
checkpoint hashes, preprocessing, decisions, historical results and limitations.
Edit the existing `pp1_demo/panels/vishwa/*.md` files to update its research notes.
