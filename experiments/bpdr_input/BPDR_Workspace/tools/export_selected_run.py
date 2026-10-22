"""Export the active notebook checkpoint and its complete guarded-loader inputs."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

WORKSPACE = Path(__file__).resolve().parents[1]
FILES = ('selected_model.joblib', 'evidence.json', 'declaration.json',
         'locked_selection.json', 'fit_receipt.json', 'demo_model.json',
         'rendered_verification.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(WORKSPACE / 'source'))
    from bpdr_input.repair.learned_runtime import active_card_path, load_model
    card_path = active_card_path()
    _, _, card = load_model(card_path)
    if not card.get('notebook_training'):
        raise ValueError('Expected an active notebook-trained model')
    run = card_path.parent
    destination = WORKSPACE / 'exports/selected_run_inference'
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(run / name, destination / name)
    shutil.copy2(WORKSPACE / 'results/notebook_training/current.json',
                 destination / 'original_active_pointer.json')
    import sklearn
    import joblib
    import numpy
    manifest = dict(source_run=run.name, checkpoint_sha256=card['checkpoint_sha256'],
                    research_gates_passed=False,
                    runtime_versions=dict(python=sys.version.split()[0],
                                          scikit_learn=sklearn.__version__,
                                          joblib=joblib.__version__, numpy=numpy.__version__),
                    files={name: sha(destination / name) for name in (*FILES, 'original_active_pointer.json')})
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    archive = WORKSPACE / 'results/selected_run_inference.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(destination.iterdir()):
            if path.is_file():
                bundle.write(path, 'selected_run_inference/' + path.name)
    print(json.dumps(dict(export=str(destination), archive=str(archive),
                          run=run.name, checkpoint_sha256=card['checkpoint_sha256']), indent=2))


if __name__ == '__main__':
    main()
