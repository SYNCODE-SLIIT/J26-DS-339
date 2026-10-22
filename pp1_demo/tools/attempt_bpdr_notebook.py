"""Execute an unmodified notebook copy, preserving the owner's original."""
import json
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parents[2]
workspace = ROOT / 'experiments/bpdr_input/BPDR_Workspace'
source = workspace / 'notebooks/04_model_training_and_evaluation.ipynb'
out = ROOT / 'pp1_demo/runtime/bpdr_notebook'
out.mkdir(parents=True, exist_ok=True)
nb = nbformat.read(source, as_version=4)
os.environ['BPDR_TRAIN_NOTEBOOK_MODEL'] = '1'
receipt = {'source': str(source.relative_to(ROOT)), 'training_requested': True}
try:
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(workspace)}}).execute()
    receipt['status'] = 'completed'
except Exception as exc:
    receipt.update(status='blocked', error=str(exc)[-5000:])
finally:
    nbformat.write(nb, out / 'executed.ipynb')
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
