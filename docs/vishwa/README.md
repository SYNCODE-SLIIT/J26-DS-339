# Vishwa research workflow

The authoritative decision and experiment record is [research_progress.md](research_progress.md).
Read it before continuing. Record actual dates, configurations, executed commands,
results, failures, interpretation, decisions, next steps, and related commit hashes.
Append corrections rather than erasing failed hypotheses. Small verified milestones
may be committed locally on `feature/vishwa-pp1-melody-understanding` under the user's
2026-10-21 authorization. Do not fabricate elapsed research time or modify teammates'
components. Keep the official TEST split reserved for final model evaluation.

## Reproduce

Run from the repository root with the existing Python 3.12 uv environment:

```sh
uv pip install --python .venv/bin/python -r requirements-vishwa-research.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit_humtrans.py --manifest datasets/humtrans/derived/research/manifest.jsonl --report experiments/vishwa/results/alignment_audit.json
.venv/bin/python scripts/run_baseline_pilot.py --output experiments/vishwa/results/pilot_001_rerun
.venv/bin/python scripts/diagnose_pitch_timing.py
.venv/bin/python scripts/synthetic_pitch_check.py
.venv/bin/python scripts/run_energy_pilot.py
```

The audit re-creates its derived manifest/report. Pilot output directories must be
new, so existing results remain available for comparison. Diagnostics use the
original `pilot_001` cohort and cached features. These caches are excluded from Git;
if absent, rerun the original pilot in a fresh checkout with output `pilot_001`.
Reinstall the research requirements after a `uv sync` that removes extra packages.

Execute notebooks from the project environment without registering a global kernel:

```python
from pathlib import Path
import nbformat
from nbclient import NotebookClient
path = Path('notebooks/vishwa/02_baseline_alignment_and_evaluation.ipynb')
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=1800, kernel_name='python3',
    resources={'metadata': {'path': str(Path.cwd())}}).execute()
nbformat.write(notebook, path)
```

## What the results mean

- Full-corpus audit checks parsing, pairing, split coverage and timeline bounds.
  It does not certify acoustic alignment or correct sung octaves.
- Pilot note metrics use explicit onset, pitch and offset tolerances; frame metrics
  use quantized MIDI labels, not continuous vocal F0 ground truth.
- Median smoothing did not improve frozen validation results. It remains an
  experiment, and the production baseline default is unchanged.
- No trained CNN–Transformer, validated CFP, beat/key/phrase head, or learned
  harmony embedding is claimed. The next priority is understanding label/audio
  timing and octave disagreement before choosing supervised targets.

## R006–R008: annotation review and weak-score pilot

The pre-R008 status above is superseded by a small CNN/CNN–Transformer feasibility
pilot. Neither model beats the matched pYIN baseline on development VALID. See
`target_policy.md` and `research_progress.md` for target semantics and limitations.

From the repository root, after installing the pinned research requirements into
its existing uv environment:

```sh
.venv/bin/python scripts/review_annotations.py
.venv/bin/python scripts/build_score_dataset.py
.venv/bin/python scripts/train_score_pilot.py
.venv/bin/python -m unittest discover -s tests -v
```

The review script requires the existing pilot feature cache produced by the earlier
baseline pilot. Execute notebooks `03_annotation_review_and_targets.ipynb` and
`04_weak_score_model_pilot.ipynb` using the NotebookClient recipe above with the
corresponding filename. Notebook 03 shows pitch/annotation evidence; notebook 04
explains target masks, model comparisons and actual loss curves. No raw audio or
model checkpoint belongs in Git.

## R009: listening review

See `listening_review.md` for the human review procedure and notebook 05 for
executed diagnostics. Build the local playable page and validate the decision form:

```sh
.venv/bin/python scripts/build_listening_review.py
open datasets/humtrans/derived/research/listening_review/index.html
.venv/bin/python scripts/check_listening_review.py
```

The builder preserves existing review decisions. Generated audio/images stay
ignored. All eight human reviews are initially pending; algorithm comparisons do
not certify annotations.
