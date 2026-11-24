# Selected notebook-run inference export

Selected run: `2759404560ab4b559a633659d316bec0`.
This is the active notebook-trained ExtraTrees preview, not the older recovery run.
It remains experimental: research/generalization gates did not pass and user review is required.

## Contents

- `selected_model.joblib`: fitted estimator and feature/operating-policy payload.
- `evidence.json`, `declaration.json`, `locked_selection.json`: matching loader integrity inputs.
- `fit_receipt.json`: completed train-only fit receipt.
- `demo_model.json`: original notebook demo card.
- `rendered_verification.json`: original rendered verification receipt.
- `original_active_pointer.json`: provenance of the selected runtime run.
- `manifest.json`: SHA256 digests and serialization-library versions.
- `install.py`: verifies the bundle and installs a relocatable runtime copy.

The seven model/runtime files are byte-for-byte copies. Do not edit the declaration,
selection, receipt, verification or checkpoint: their checksums are cross-linked.
The original card and pointer contain the author's absolute paths.

## Use in another checkout

Use the BPDR source code and Python environment from this repository. Match the
scikit-learn, joblib and numpy versions recorded in `manifest.json` when loading
the serialized estimator. The source environment must provide the existing BPDR
audio and rendering dependencies; this export does not install packages.

From `experiments/bpdr_input`, with `bpdr_input` installed in that environment:

```powershell
.\.venv\Scripts\python.exe BPDR_Workspace\exports\selected_run_inference\install.py --activate
```

For an extracted ZIP, pass the absolute path to its `install.py` instead. Omit
`--activate` to validate and install without changing the active notebook pointer.
Installation refuses to overwrite an existing exported run. Activation backs up
the previous pointer in the installed run.

The installer copies the seven files to a new notebook-run directory, relocates
the demo card's checkpoint/evidence paths, validates with the real guarded loader,
and optionally writes a newly hashed active pointer. Evidence receipts remain
unchanged; the original card and pointer stay in this bundle for comparison.

Then use the existing audio inference entry point:

```python
from bpdr_input.repair.learned_runtime import load_model, run
model, payload, card = load_model()
output_directory, report = run("path/to/your_vocal.wav")
```

The export supports inference on supplied audio. Historical demo listening
examples, downloaded datasets, training pairs and rendered audio are not included.
Their original paths remain historical references in the evidence. The installed
card has an empty `demo_cases` list so it does not advertise unavailable samples.
The strict `load_candidate` gate must still reject this failed-gate model; the
notebook preview uses the existing integrity-checked exploratory runtime.

To rebuild the folder and ZIP from the active local run, execute
`BPDR_Workspace/tools/export_selected_run.py` with the component Python environment.
