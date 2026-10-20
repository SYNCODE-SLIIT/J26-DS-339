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
