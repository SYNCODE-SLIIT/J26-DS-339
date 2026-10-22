"""Install this verified export into the BPDR checkout running this script.

Run with that checkout's Python environment. Does not activate the model unless
--activate is supplied. Original exported evidence files are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

BUNDLE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(activate=False):
    from bpdr_input.repair import learned_runtime as runtime
    manifest = json.loads((BUNDLE / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        if Path(name).name != name or sha(BUNDLE / name) != digest:
            raise ValueError('Export integrity check failed: ' + name)
    destination = runtime.WORKSPACE / 'results/model_recovery/notebook_training' / (manifest['source_run'] + '_inference_export')
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing run: ' + str(destination))
    destination.mkdir(parents=True)
    for name in manifest['files']:
        if name != 'original_active_pointer.json':
            shutil.copy2(BUNDLE / name, destination / name)
    card_path = destination / 'demo_model.json'
    card = json.loads(card_path.read_text(encoding='utf-8'))
    card['checkpoint'] = str(destination / 'selected_model.joblib')
    card['evidence'] = str(destination / 'evidence.json')
    # Sample audio is an external dataset dependency, not an inference input.
    # Do not advertise missing demo samples in a relocated inference export.
    card['demo_cases'] = []
    card['export_note'] = 'Relocated inference export; original demo card is preserved in the bundle. Supply your own audio.'
    card_path.write_text(json.dumps(card, indent=2) + '\n', encoding='utf-8')
    runtime.load_model(card_path=card_path)
    if activate:
        pointer = runtime.ACTIVE_DEMO
        pointer.parent.mkdir(parents=True, exist_ok=True)
        if pointer.exists():
            shutil.copy2(pointer, destination / 'previous_active_pointer.json')
        temporary = destination / 'active_pointer.tmp'
        temporary.write_text(json.dumps(dict(card=str(card_path), card_sha256=sha(card_path),
            training_run=str(destination), checkpoint_sha256=sha(destination / 'selected_model.joblib')),
            indent=2) + '\n', encoding='utf-8')
        temporary.replace(pointer)
        runtime.load_model()
    return card_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    print(install(activate=args.activate))
