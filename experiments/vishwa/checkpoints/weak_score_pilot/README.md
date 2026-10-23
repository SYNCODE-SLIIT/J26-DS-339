# Vishwa weak-score inference bundle

These are byte-for-byte exports of the existing seed-339, five-epoch feasibility
models. Both checkpoint SHA-256 digests match `experiments/vishwa/results/score_model_pilot.json`.
No retraining or new accuracy evaluation was performed to create this export.

- `cnn.pt`: 18,413 parameters; historical VALID recording-macro score-class accuracy 42.53%.
- `cnn_transformer.pt`: 31,117 parameters; historical accuracy 39.45%.
- `train_normalization.npz`: TRAIN-only per-Mel-band mean/std, required for inference.
- `manifest.json`: hashes, feature settings, class ordering, training policy and scope.
- Model definition: `src/melody_understanding/score_model.py`, copied from the training class without architecture changes.

Inputs: mono 22,050-Hz audio, peak 0.95, 80 Mel bands, FFT 4096, hop 256,
power dB relative to the recording maximum, TRAIN normalization, 256-frame windows.
Outputs: 13 frame classes (rest, then C through B), with padding excluded.
These are intended-score pitch-class models. They do not predict octave, note
boundaries, key, beats, chords or a validated harmony embedding. The current
pipeline uses live pYIN note events downstream and presents both learned models
as genuine parallel experiments. No accuracy is inferred from an unlabelled upload.

See `pp1_demo/README.md` for uv installation, portable inference and the live UI.
These files are small Git binary assets; no Git LFS or raw dataset is required.
