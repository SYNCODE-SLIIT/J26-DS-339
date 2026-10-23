# Future portable pipeline decision - 14 October 2026

The owner intends to build a melody/chord pipeline supporting **macOS and Windows**, with an optional BPDR preparation/export capability, and integrate it with BPDR later. **For current work, continue using the recordings, stems, crops and evidence already extracted.** This is a future implementation direction, not a request to implement the pasted specification now or to repeat extraction.

The full supplied specification is preserved in [Portable_Pipeline_Brief_2026-10-14.txt](Portable_Pipeline_Brief_2026-10-14.txt), SHA-256 `dcf1a2aa636cb41e56e02fedde2606b204ec52630f658af4eb769ee1b3e13fd2`. Its imperative instructions describe the later exporter implementation. Existing ownership boundaries still apply; this decision changes no shared pipeline or teammate files.

## Requirements to carry into the later integration

- Add a separate exporter/companion and explicit opt-in command or flag. Preserve all existing melody, beat, chord, key, Roman outputs, schemas, consumers, caches and the published +12-semitone convention. BPDR reads unshifted fractional `model_midi_pitch` from vocal-only predictions.
- Support Vocadito solo recordings with human F0/A1/A2 references and Jamendo separated vocals with `model_estimate` references. Preserve existing singer partitions; split source/artist before cropping, keep descendants together, filter licenses and duplicates, and keep held-out audio out of design/tuning.
- Select deterministic, versioned phrases <=20 seconds, preferably around natural pauses with stable support and neighboring context. Preserve rests/time origin, record sample/time mapping and edge warnings, protect uncertain edges, and test crop/full-context effects.
- Reuse BPDR technical preparation and `analysis.py`: mono 22,050 Hz PCM24, frozen PESTO mir-1k_g7, aligned 441-sample hop, centered 1,024-sample spectral windows, 64 mel bands and confidence 0.8. Export the defined raw `[T,71]` float32 matrix, timestamps, measured F0 with unvoiced NaNs and validity masks. The frame-only input is the first 70 columns. Label the boundary condition; keep reference/target arrays separate and fit normalization on training sources only.
- Disable initial denoising and musical correction. Technical status is `ACCEPT_UNCHANGED`, `RERECORD_REQUIRED` or `TECHNICAL_FAILURE`; this exporter cannot grant `ACCEPT_PREPARED`.
- Retain provisional screening, two real independent musical reviews, approved stable intervals, integrity checks and unchanged v0.3 audit requirements. Feature export alone cannot create research eligibility. Gated A/B/C/D generation is a separate opt-in stage/handoff; verify compatibility with `examples.py` and `require_training_example`, rather than treating a generic export manifest as loader-compatible.
- Keep immutable/atomic runs in a separate configurable export root with relative paths and source/model/configuration hashes. Resume valid artifacts without replacing reviews or repeating separation. Reuse a temporary vocal stem before normal cleanup where feasible.
- Default to compact accepted crops, compressed features/masks and essential provenance. Full stems, repeated decodes and detailed debugging are explicit options. Avoid duplicate waveforms/features, MIDI copies, dashboards and model/source copies by default. These future storage rules do not authorize removing existing evidence.
- Use explicit backend/capability adapters: native pinned MLX where supported on Apple Silicon; validated same-checkpoint PyTorch on Windows, conditional GPU support and documented CPU fallback. Resolve `bin/python` versus `Scripts/python.exe`, portable locking/paths, isolated dependencies and platform-specific sleep prevention. Record backend/device/precision/settings, release large models sequentially and restore sleep behavior on exit.
- Deliver setup/run instructions for each OS, the output/loader contract, a small verified package, regression results and measured storage/runtime. Compare a declared sample on both platforms; an untested platform remains unverified. Do not promise faster or numerically identical results.
- Export preparation data; Transformer training, production repair/controller/renderer and new harmony outputs remain outside this pipeline task.

The brief names `datasets/mtg-jamendo/melody-chord-extraction-pipeline`; resolve the actual checkout/assets path and hook ownership when implementation starts. Verify its actual sample/output schemas at that time.

## Current baseline remains active

Use [the completed five-artist sample report](Jamendo_Five_Artist_Sample_2026-10-13.md) and `runs/preparation/d274ed728d3d46f3b2661ba5291e61da`. It contains three Vocadito and seven Jamendo unchanged technical outputs. Nine crops are provisional review candidates: three Vocadito and six Jamendo from three artists. Eighteen review rows remain pending; zero research-ready examples exist.

The retained batch is `runs/jamendo_sample/bf912553c00c40598b8f0d64ae266de0`; the current intervention evidence is `runs/jamendo_yield/de2f85a01b194c4eab009aa95b14e496`. Five of eighteen tested shifts pass v0.3 across two artists. Independent review and rendering/audit-coverage diagnosis remain the next work. The future portable pipeline does not block these tasks, and its promised features must not be credited as current implementation.
