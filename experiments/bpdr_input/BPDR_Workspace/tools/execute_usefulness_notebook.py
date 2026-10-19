"""Execute the short current-development notebook without training models."""
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

WORKSPACE = Path(__file__).resolve().parents[1]
COMPONENT = WORKSPACE.parent
sys.path.insert(0, str(WORKSPACE / 'source'))
from bpdr_input.audio.pipeline import checksum


def main():
    runtime = WORKSPACE / 'results/notebook_runtime'
    kernel = runtime / 'kernels/bpdr-local'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / 'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable, '-P', '-B', '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='BPDR local Python', language='python',
        env=dict(PYTHONPATH=str(WORKSPACE / 'source'), PYTHONDONTWRITEBYTECODE='1', BPDR_COMPONENT_ROOT=str(COMPONENT)))), encoding='utf-8')
    for key, path in [('JUPYTER_PATH', runtime), ('JUPYTER_RUNTIME_DIR', runtime / 'runtime'),
                      ('JUPYTER_CONFIG_DIR', runtime / 'config'), ('IPYTHONDIR', runtime / 'ipython')]:
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    path = WORKSPACE / 'notebooks/06_real_voice_pilot_and_usefulness.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    started = time.perf_counter()
    try:
        NotebookClient(notebook, kernel_name='bpdr-local', timeout=120, startup_timeout=40,
                       resources={'metadata': {'path': str(path.parent)}}).execute()
    except Exception:
        nbformat.write(notebook, WORKSPACE / 'results/06_usefulness_failed.ipynb')
        raise
    outputs = [output for cell in notebook.cells for output in cell.get('outputs', [])]
    plots = sum('image/png' in output.get('data', {}) for output in outputs)
    audio = sum('<audio' in output.get('data', {}).get('text/html', '') for output in outputs)
    errors = sum(output.get('output_type') == 'error' for output in outputs)
    required = notebook.metadata.get('bpdr_verification', {})
    if errors or plots < required.get('minimum_plots', 0) or audio < required.get('minimum_audio', 0):
        raise RuntimeError(f'Incomplete notebook outputs: errors={errors}, plots={plots}, audio={audio}')
    nbformat.write(notebook, path)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    html_path = path.with_suffix('.html')
    html_path.write_text(html, encoding='utf-8')
    receipt = dict(notebook=str(path), notebook_sha256=checksum(path), html_export=str(html_path),
                   html_sha256=checksum(html_path), code_cells=sum(cell.cell_type == 'code' for cell in notebook.cells),
                   errors=errors, plot_outputs=plots, audio_players=audio,
                   elapsed_seconds=round(time.perf_counter() - started, 3),
                   scope='Fresh kernel; original-data evidence verification and one acoustic-analysis pass; no training')
    (WORKSPACE / 'results/usefulness_notebook_execution.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
