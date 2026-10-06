"""Project-local Jupyter kernel and repeatable notebook execution/export."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def configure_kernel(component):
    root = component/'.jupyter'
    kernel = root/'kernels'/'bpdr-local'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                                                    display_name='BPDR component (uv)', language='python')), encoding='utf-8')
    os.environ['JUPYTER_PATH'] = str(root)+(os.pathsep+os.environ['JUPYTER_PATH'] if os.environ.get('JUPYTER_PATH') else '')
    # Keep configuration/history/runtime files component-local, without global registration.
    os.environ['JUPYTER_CONFIG_DIR'] = str(root/'config')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(root/'runtime')
    os.environ['IPYTHONDIR'] = str(root/'ipython')
    return 'bpdr-local'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['launch', 'execute'])
    parser.add_argument('--dataset-root', type=Path)
    args = parser.parse_args()
    component = Path(__file__).resolve().parents[2]
    if args.dataset_root:
        os.environ['BPDR_DATASET_ROOT'] = str(args.dataset_root.resolve())
    kernel = configure_kernel(component)
    if args.command == 'launch':
        return subprocess.call([sys.executable, '-m', 'jupyterlab', '--no-browser', '--ServerApp.ip=127.0.0.1',
                                '--notebook-dir='+str(component/'notebooks')], cwd=component)
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    destination = component/'runs'/'notebooks'/uuid.uuid4().hex
    destination.mkdir(parents=True)
    rows = []
    for path in sorted((component/'notebooks').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        print('Executing', path.name, flush=True)
        NotebookClient(notebook, kernel_name=kernel, timeout=600, resources={'metadata': {'path': str(component)}}).execute()
        outputs = [o for cell in notebook.cells for o in cell.get('outputs', [])]
        plots = sum('image/png' in o.get('data', {}) for o in outputs)
        audio = sum('<audio' in o.get('data', {}).get('text/html', '') for o in outputs)
        required = notebook.metadata.get('bpdr_verification', {})
        if plots < required.get('minimum_plots', 0) or audio < required.get('minimum_audio', 0):
            raise RuntimeError(f'{path.name} is missing expected plots or audio controls')
        nbformat.write(notebook, destination/path.name)
        html, _ = HTMLExporter().from_notebook_node(notebook)
        (destination/(path.stem+'.html')).write_text(html, encoding='utf-8')
        rows.append(dict(notebook=path.name, executed_copy=str(destination/path.name), html_export=str(destination/(path.stem+'.html')),
                         code_cells=sum(cell.cell_type == 'code' for cell in notebook.cells), plot_outputs=plots, audio_outputs=audio, errors=0))
    (destination/'verification.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (component/'runs'/'notebooks'/'latest.json').write_text(json.dumps({'path': str(destination)}, indent=2), encoding='utf-8')
    print('NOTEBOOK_RUN:', destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
