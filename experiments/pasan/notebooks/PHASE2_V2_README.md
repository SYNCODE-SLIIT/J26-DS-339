# Normalized melody harmonization experiments

Open `03_phase2_clean_dataset.ipynb`, then `04_phase2_harmony_transformer.ipynb`.
Both have been executed top to bottom, with saved tables/plots. The original
notebooks, v1 dataset and original checkpoints were left unchanged.

## What this implements

Melody MIDI events plus genre tags, major/minor mode and beat/meter timing go
through a key- and octave-invariant representation. A small transformer predicts
Roman chords on each beat. Consecutive identical chords are combined into a
progression, retaining beat/second durations. Transposition, global octave shifts
and original-key versus C-major/A-minor inference have been checked.

Unlike simply retaining pitch-class order, this representation also preserves
rhythm, rests, relative contour and coarse within-beat order. Genre conditioning
accepts multiple labels. The model never receives the original key, absolute
register or reference chords.

Roman roots are relative to the tonic using the pipeline's chromatic convention
in both modes: A minor's C major chord is `bIII:MAJ`. Quality is explicit, such as
`i:MIN` and `V:DOM7`. There are 108 root × quality combinations. Unsupported
annotations, `N` and ambiguous beat labels remain masked, not reassigned to tonic.

## Executed findings

- Audited 1,372 extracted folders. The strict agreement/high-confidence cohort
  contains 377 candidate tracks. The saved dataset retains 4,446 windows from
  373 songs: train 2,463 / validation 952 / test 1,031 windows.
- Artists remain separated across the official splits. Source and artifact
  hashes, structural checks, rejection reasons and note edit/flag records are
  saved. Rests, chromatic tones and jumps are not indiscriminately removed.
- Trained a local MLP, two ordinary transformer variants, and two smaller
  transformers with multi-scale chroma/chord-tone evidence. Repeated the frozen
  winning architecture at seeds 7 and 123 for validation stability.
- Seed-42 winner: ordinary transformer, epoch 5. Validation accuracy 51.52%,
  supported-class macro-F1 0.0666 and chord-change F1 0.1898. The mode-only prior
  reaches 53.87% accuracy. Validation rejects transition smoothing (weight 0).
- Held-out test: transformer accuracy **44.64%**, versus **45.45%** for the
  mode-only prior. Macro-F1 improves from 0.0289 to 0.0575 and change-F1 from
  0 to 0.1892. The paired artist-bootstrap 95% interval for the accuracy difference
  is −3.05 to +1.61 percentage points. This run does not establish an accuracy gain.
- Pitch alteration reduces validation accuracy substantially; removing all
  melody falls back toward tonic predictions and can increase accuracy while
  destroying minority/chord-change metrics. Genre ablation has little effect in
  this run. The model still overpredicts tonic chords; harmonization is not solved.

The extracted chords and keys are pseudo-labels. Structural cleaning cannot
establish their musical correctness. Key agreement is a selection rule, not a
ground-truth guarantee. Listening review and better labels are the highest-value
next steps; more model capacity is not established as the remedy.

## Files

`phase2_dataset_v2/` (generated, ignored by Git):

- `features.npz`: bounded numeric inputs, targets, masks, mode and genres.
- `examples.parquet`: accepted window metadata, split and beat boundaries.
- `track_audit.csv`, `window_audit.parquet`: all inspected tracks/windows.
- `cleaned_notes.jsonl`, `note_actions.jsonl`: conservative notes and action log.
- `metadata.json`: policy, vocabulary, raw-source and saved-artifact checksums.

`phase2_runs_v2/` (generated, ignored by Git): resumable experiments, best/last
checkpoints, CSV comparisons, per-class results, seed checks, test uncertainty,
`summary.json`, and `selected_bundle/`.

`phase2_harmony/data.py` and `model.py` contain the inspectable implementation.
The notebooks define the experiment controls and show the evidence. The helper
`tools/create_phase2_notebooks.py` regenerates notebook sources and clears their
outputs; do not run it merely to train or inspect the notebooks.

## Reproduce

Use Python 3.12 and this repository's `.venv` kernel. The exact package versions
used are in `requirements-phase2.txt`. The notebooks locate the repository from
their current directory; they do not depend on a user-specific absolute path.

From the repository root:

```sh
uv pip install --python .venv/bin/python -r experiments/pasan/notebooks/requirements-phase2.txt
.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace experiments/pasan/notebooks/03_phase2_clean_dataset.ipynb
.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=600 experiments/pasan/notebooks/04_phase2_harmony_transformer.ipynb
.venv/bin/python -m pytest -q experiments/pasan/notebooks/tests/test_phase2_harmony.py
```

Reruns verify/reuse the snapshot and resume/reuse matching experiments. For new
cleaning rules, use a new snapshot directory. For changed hyperparameters, use a
new run directory. Keep the test split frozen; use validation for iteration.

## Predict on your melody

The generated `phase2_runs_v2/selected_bundle/` contains weights, vocabulary,
transition matrix, metadata, implementation source and `example_input.json`.
The example is a real held-out validation melody transposed to its canonical key.

```sh
PYTHONPATH=experiments/pasan/notebooks .venv/bin/python -m phase2_harmony.predict \
  --bundle experiments/pasan/notebooks/phase2_runs_v2/selected_bundle \
  --input experiments/pasan/notebooks/phase2_runs_v2/selected_bundle/example_input.json \
  --output /tmp/roman_progression.json
```

Input JSON has `notes` (`onset_sec`, `offset_sec`, `midi_pitch_rounded`),
`beat_boundaries` in seconds, `mode`, `genres`, and either `normalized: true`
(C-major or A-minor MIDI) or `tonic_pc` for original-key MIDI. Optional `beats`
records provide `beat_in_bar`, `meter_numerator`, `meter_denominator`; the default
is explicitly 4/4. Arbitrary-length songs use overlapping 48-beat windows. Beat
tracking/key estimation from raw audio are outside this inference function.

To move the bundle elsewhere, install torch/numpy/pandas/pyarrow from the recorded
environment and put the bundle directory on `PYTHONPATH`. Its included
`phase2_harmony` package supports the same CLI. Inference checks artifact hashes
and returns beat chords plus Roman progression segments and durations.

## Validation boundaries

Both notebooks execute cleanly; all eight rendered figures were visually
inspected. Ten regression tests cover transposition/octave invariance, real rests,
chromatic-note preservation, masking, complete bars, padding isolation, decoding,
boundary-fragment parity and long-song coverage. CLI inference was run against the
exported example. No listening validation or manual chord annotation was claimed.
