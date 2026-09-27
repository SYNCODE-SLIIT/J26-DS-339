# BPDR input component - PP1 experiments

This folder contains Liyanage L. N. P.'s (IT23141506) experimental Vocal Input Validation and Minimal Musical Repair component for J26-DS-339. All component code, tests, configuration, documents and generated runs stay under `experiments/bpdr_input` while the final pipeline is being established.

## Current state

Phase 0 foundation: standalone Python package, draft typed accepted-only hand-off contracts, a read-only vocadito inventory command, contract/data-audit tests and the PP1 plan package. There is no trained BPDR model, pitch renderer or general recording-acceptance algorithm yet. Dataset audit checks readability and reference-file presence; it does not establish musical correctness or intended melody.

## Start here

1. Read [the PP1 master plan](docs/PP1_Master_Plan.md).
2. Follow [the phase execution plan](docs/PP1_Phase_Execution_Plan.md).
3. Update [the evidence tracker](docs/PP1_Evidence_Tracker.md) from actual artifacts/results.

Run commands from this component directory using Python 3.12 or later. The foundation requires only the standard library. In PowerShell, use the source directory without changing the repository's root environment:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -v
python -m bpdr_input.inventory --dataset-root 'C:\Users\lnipu\Projects\Music Rp\Input Validation\vocadito' --output 'runs\dataset_inventory.json'
```

The package also supports an editable installation in a component-local virtual environment when needed. Later audio/model dependencies will be added here after a compatibility check, not to the repository's root configuration.

## Data and results

Use the existing vocadito download as a read-only data source; `--dataset-root` makes its location configurable. Generated inventory reports belong in ignored `runs/`. Raw recordings, participant material and checkpoints are excluded by this component's `.gitignore` and should not be copied into tracked source files.

## Scope

BPDR prepares accepted vocal audio before melody understanding. The proposal's research comparison is BPDR versus same-data augmentation-only and matched frame-only models, with original audio and conservative rules as reference conditions. False split/merge boundary changes and known local detuning are separate interventions. A model's comparative benefit remains unproven until the planned experiments run.

Next implementation gate: the timing-preserving unchanged-audio/failure paths and frozen acoustic analysis described in Phase 1. The current contract draft must expand to full operation/interval/version metadata before live integration.
