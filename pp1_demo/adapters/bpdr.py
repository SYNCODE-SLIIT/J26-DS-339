"""Run the owner's guarded export in a subprocess; all writes are app-local."""
import json
import os
from pathlib import Path
import subprocess

from pp1_demo.config import APP, BPDR_PYTHON


def predict(source, folder):
    if not BPDR_PYTHON.is_file():
        raise FileNotFoundError('BPDR interpreter missing. Set PP1_BPDR_PYTHON to its compatible Python.')
    output = Path(folder) / 'bpdr_result.json'
    command = [str(BPDR_PYTHON), '-B', str(APP/'workers/bpdr_worker.py'), str(source), str(folder), str(output)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (Path(folder)/'bpdr_worker.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                cwd=APP, timeout=240)
    if result.returncode or not output.exists():
        detail=(Path(folder)/'bpdr_worker.log').read_text()[-1800:]
        raise RuntimeError('BPDR could not run: '+detail)
    report=json.loads(output.read_text())
    if not report.get('accepted_audio'):
        raise ValueError('BPDR did not accept this recording: '+report.get('message','See its report.'))
    return report
