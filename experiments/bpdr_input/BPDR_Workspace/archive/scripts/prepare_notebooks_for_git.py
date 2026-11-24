"""Preserve local executed notebooks, then keep only sources in version control."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nbformat

WORKSPACE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = WORKSPACE / 'results/notebook_output_backups' / stamp
    rows = []
    for path in sorted((WORKSPACE / 'notebooks').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        dirty = bool(notebook.metadata.get('widgets')) or any(
            c.get('outputs') or c.get('execution_count') is not None for c in notebook.cells)
        if not dirty:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        backup = destination / path.name
        backup.write_bytes(path.read_bytes())
        original_hash = sha(path)
        if sha(backup) != original_hash:
            raise RuntimeError('Executed notebook backup failed verification')
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop('execution', None)
        notebook.metadata.pop('widgets', None)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        rows.append(dict(source=str(path), executed_backup=str(backup),
                         executed_sha256=original_hash, source_sha256=sha(path)))
    if rows:
        receipt = destination / 'preservation.json'
        receipt.write_text(json.dumps(dict(notebooks=rows), indent=2)+'\n', encoding='utf-8')
        print(receipt)
    print('Notebook sources cleaned:', len(rows), '; executed originals remain local.')


if __name__ == '__main__':
    main()
