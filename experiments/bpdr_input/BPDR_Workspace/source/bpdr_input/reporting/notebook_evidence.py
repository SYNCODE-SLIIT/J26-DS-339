"""Component-local, read-only evidence checks for the maintained panel notebooks."""
import json
from pathlib import Path

from bpdr_input.audio.pipeline import checksum


def component_path(component, path):
    component = Path(component).resolve()
    candidate = (component/Path(path)).resolve()
    if not candidate.is_relative_to(component):
        raise ValueError('Panel evidence path leaves the BPDR component')
    if not candidate.exists():
        raise FileNotFoundError(f'Missing panel evidence: {candidate}')
    return candidate


def load_index(component):
    index = component_path(component, 'notebooks/evidence_index.json')
    data = json.loads(index.read_text(encoding='utf-8'))
    if len(set(data['maintained_notebooks'])) != len(data['maintained_notebooks']):
        raise ValueError('Duplicate maintained notebook name')
    for name in data['maintained_notebooks']:
        if Path(name).name != name or Path(name).suffix != '.ipynb':
            raise ValueError('Maintained notebook must be a direct notebook filename')
        component_path(component, Path('notebooks')/name)
    return data, {key: component_path(component, path) for key, path in data['evidence'].items()}


def verified_hash(component, path, expected):
    path = component_path(component, path)
    if checksum(path) != expected:
        raise ValueError(f'Changed panel evidence: {path}')
    return path


def verify_calibration(component, path):
    """Reject altered source/artifacts rather than silently display stale pass counts."""
    path = component_path(component, path)
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['run_status'] != 'complete' or result['adopted'] or result['training_ready_examples'] != 0:
        raise ValueError('Expected a completed, unadopted exploratory calibration')
    for source, digest in result['tracked_inputs'].items():
        verified_hash(component, source, digest)
    for artifact in result['artifacts']:
        verified_hash(component, artifact['path'], artifact['sha256'])
    for source in result['sources']:
        if 'accepted_path' in source:
            verified_hash(component, source['accepted_path'], source['accepted_sha256'])
    return result


def verify_bundle(component, directory):
    directory = component_path(component, directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['run_status'] != 'complete' or manifest['training_ready_examples'] != 0:
        raise ValueError('Expected completed exploratory reference-review bundle')
    for path, digest in manifest['source_hashes'].items():
        verified_hash(component, path, digest)
    for artifact in manifest['artifacts']:
        verified_hash(component, artifact['path'], artifact['sha256'])
    for candidate in manifest['candidates']:
        verified_hash(component, candidate['audio_path'], candidate['audio_sha256'])
    return manifest
