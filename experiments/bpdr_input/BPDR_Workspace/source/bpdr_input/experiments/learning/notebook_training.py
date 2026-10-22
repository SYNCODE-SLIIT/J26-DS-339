"""Explicit notebook fit/save/reload lifecycle; old experiments stay immutable.

This is reproduction on existing development data, not a new independent test.
The notebook owns the visible .fit() and joblib.dump() calls. This module checks
their inputs/receipts and verifies the delivered audio before demo activation.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os
import uuid

import numpy as np
from sklearn.utils.validation import check_is_fitted

from bpdr_input import COMPONENT_ROOT
from bpdr_input.audio.pipeline import checksum
from bpdr_input.modeling.recovery_model import FEATURE_VERSION, load_locked_diagnostic

WORKSPACE = COMPONENT_ROOT / 'BPDR_Workspace'
ROOT = WORKSPACE / 'results/model_recovery/notebook_training'
POINTER = WORKSPACE / 'results/notebook_training/current.json'
CATALOGS = [f'{split}_{kind}.json' for split in ('train', 'calibration', 'pilot_evaluation')
            for kind in ('sources', 'pairs')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_arrays(x, y, weight):
    if (x.ndim != 2 or x.shape[1] != 31 or not len(x) or
            y.shape != (len(x),) or weight.shape != y.shape or
            not all(np.isfinite(a).all() for a in (x, y, weight)) or
            np.any(weight <= 0) or not np.isclose(weight.sum(), 1.)):
        raise ValueError('Expected finite training rows x 31, aligned targets and positive unit-sum weights')


def arrays_sha256(x, y, weight):
    validate_arrays(x, y, weight)
    digest = hashlib.sha256()
    for array in (x, y, weight):
        digest.update(str((array.shape, str(array.dtype))).encode('utf-8'))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def run_folder(folder):
    folder = Path(folder).resolve()
    if folder == ROOT.resolve() or not folder.is_relative_to(ROOT.resolve()):
        raise ValueError('Notebook training must use its own new run directory')
    return folder


def verify_inputs(folder):
    for path, expected in read(run_folder(folder) / 'input_hashes.json').items():
        if checksum(path) != expected:
            raise ValueError('Training/replay input changed: ' + path)


def begin_training(source_run):
    """Declare a new fixed ExtraTrees reproduction; reuse existing audited pairs."""
    source_run = Path(source_run).resolve()
    if not source_run.is_relative_to((WORKSPACE / 'results/model_recovery').resolve()):
        raise ValueError('Source run must remain component-local')
    old, payload = load_locked_diagnostic(source_run / 'selected_model.joblib')
    if old['kind'] != 'extra_trees':
        raise ValueError('This workflow reproduces the fixed ExtraTrees configuration')
    manifest = read(WORKSPACE / 'datasets/vocalset_pilot/manifest.json')
    groups = {s: {r['singer_id'] for r in manifest['records'] if r['split'] == s}
              for s in ('train', 'calibration', 'pilot_evaluation')}
    if any(groups[a] & groups[b] for a, b in [('train', 'calibration'), ('train', 'pilot_evaluation'), ('calibration', 'pilot_evaluation')]):
        raise ValueError('Singer overlap between declared splits')
    inputs = {}
    paths = [source_run / name for name in CATALOGS]
    paths += [source_run / name for name in ('declaration.json', 'evidence.json', 'locked_selection.json', 'selected_model.joblib')]
    paths += [WORKSPACE / 'datasets/vocalset_pilot/manifest.json']
    for split in groups:
        for pair in read(source_run / f'{split}_pairs.json'):
            if pair['source']['singer_id'] not in groups[split]:
                raise ValueError('Pair singer disagrees with its declared split')
            if 'excluded' in pair:
                continue
            if pair['audit']['passed'] is not True:
                raise ValueError('A nonexcluded training/replay pair lacks a passing intervention audit')
            parent = Path(pair['path_A']).parent
            paths += [Path(pair[k]) for k in ('path_A', 'path_C', 'target_path')]
            paths += [parent / 'original_features.npz', parent / f"features_{pair['cents']:+g}.npz"]
    for path in paths:
        path = path.resolve()
        if not path.is_relative_to(WORKSPACE.resolve()):
            raise ValueError('Training input leaves workspace: ' + str(path))
        inputs[str(path)] = checksum(path)
    folder = ROOT / uuid.uuid4().hex
    folder.mkdir(parents=True)
    for name in CATALOGS:
        (folder / name).write_bytes((source_run / name).read_bytes())
    params = old['estimator'].get_params()
    params = {k: params[k] for k in ('n_estimators', 'max_depth', 'min_samples_leaf', 'criterion',
                                    'bootstrap', 'max_features', 'random_state', 'n_jobs')}
    declaration = dict(version='notebook-extra-trees-reproduction-1',
        declared_utc=datetime.now(timezone.utc).isoformat(), source_run=str(source_run),
        source_checkpoint_sha256=checksum(source_run / 'selected_model.joblib'),
        manifest_sha256=checksum(WORKSPACE / 'datasets/vocalset_pilot/manifest.json'),
        model_parameters=params, operating_policy=payload['operating_policy'],
        feature_version=FEATURE_VERSION, fit_split='train', feature_count=31,
        training_weights='Original .50, shifted plateau .35, shifted outside .15; singer balanced',
        selection='Reproduce fixed ExtraTrees and reuse previously locked policy; no new model/policy search',
        evaluation='Reuse development calibration/pilot; NOT an untouched final test',
        research_gates_passed=False, requires_user_review=True,
        package_versions={p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'joblib')},
        source_hashes={str(p): checksum(p) for p in [Path(__file__),
            WORKSPACE / 'source/bpdr_input/modeling/local_context_pilot.py',
            WORKSPACE / 'source/bpdr_input/experiments/learning/model_recovery.py']})
    snapshots = folder / 'source_snapshots'
    snapshots.mkdir()
    for source, digest in declaration['source_hashes'].items():
        target = snapshots / Path(source).name
        target.write_bytes(Path(source).read_bytes())
        if checksum(target) != digest:
            raise ValueError('Could not preserve the declared training source')
    write(folder / 'declaration.json', declaration)
    write(folder / 'input_hashes.json', inputs)
    write(folder / 'locked_selection.json', dict(model='extra_trees', operating_policy=payload['operating_policy'],
        calibration_gate_passes=False, research_gates_passed=False, reused_settings_from=str(source_run),
        interpretation='Fixed settings copied before this fit; historical pilot was already inspected.'))
    return folder, declaration


def record_fit(folder, estimator, x, y, weight, seconds):
    folder = run_folder(folder)
    verify_inputs(folder)
    check_is_fitted(estimator)
    digest = arrays_sha256(x, y, weight)
    if estimator.n_features_in_ != 31:
        raise ValueError('Unexpected fitted feature count')
    pairs = [p for p in read(folder / 'train_pairs.json') if 'excluded' not in p]
    receipt = dict(fit_completed=True, model='extra_trees', training_rows=len(y), features=x.shape[1],
        audited_pairs=len(pairs), original_recordings=len({p['source']['sha256'] for p in pairs}),
        training_singers=len({p['source']['singer_id'] for p in pairs}), fit_seconds=float(seconds),
        arrays_sha256=digest, declaration_sha256=checksum(folder / 'declaration.json'),
        train_weighted_mae_cents=float(np.average(np.abs(estimator.predict(x)-y), weights=weight)),
        zero_train_mae_cents=float(np.average(np.abs(y), weights=weight)),
        calibration_or_pilot_used_for_fit=False, research_gates_passed=False)
    write(folder / 'fit_receipt.json', receipt)
    return receipt


def checkpoint_payload(folder):
    folder = run_folder(folder)
    declaration = read(folder / 'declaration.json')
    return dict(feature_version=FEATURE_VERSION, kind='extra_trees', model_type='extra_trees',
        operating_policy=declaration['operating_policy'],
        declaration_sha256=checksum(folder / 'declaration.json'),
        selection_sha256=checksum(folder / 'locked_selection.json'),
        scope='Notebook-trained reproduction: stable-vocal controlled-detuning preview; development evidence only',
        requires_user_review=True, notebook_training=True,
        fit_receipt_sha256=checksum(folder / 'fit_receipt.json'))


def record_checkpoint(folder):
    folder = run_folder(folder)
    receipt = read(folder / 'fit_receipt.json')
    if receipt['declaration_sha256'] != checksum(folder / 'declaration.json'):
        raise ValueError('Training declaration changed after fitting')
    verify_inputs(folder)
    path = folder / 'selected_model.joblib'
    evidence = dict(checkpoint=str(path), checkpoint_sha256=checksum(path),
        declaration_sha256=checksum(folder / 'declaration.json'), selection=read(folder / 'locked_selection.json'),
        fit_receipts={'extra_trees': dict(seconds=receipt['fit_seconds'], training_rows=receipt['training_rows'],
                                        completed_within_cap=receipt['fit_seconds'] <= 120.)},
        fit_receipt_sha256=checksum(folder / 'fit_receipt.json'), outer_proposal_gate_passes=False,
        rendered_demo_gate_passes=False, model_adopted=False, old_research_gate_preserved=True)
    write(folder / 'evidence.json', evidence)
    _, payload = load_locked_diagnostic(path)
    if payload.get('fit_receipt_sha256') != checksum(folder / 'fit_receipt.json'):
        raise ValueError('Checkpoint is not bound to this fit receipt')
    return evidence


def verify_for_demo(folder):
    """Recompute development scores and all delivered pilot outcomes for this file."""
    from bpdr_input.experiments.learning.model_recovery import cached_pairs, score
    from bpdr_input.experiments.learning.verify_model_recovery import run as replay
    folder = run_folder(folder)
    verify_inputs(folder)
    model, payload = load_locked_diagnostic(folder / 'selected_model.joblib')
    evidence = read(folder / 'evidence.json')
    for split, key in [('calibration', 'calibration_replay'), ('pilot_evaluation', 'outer')]:
        summary, rows = score(model, payload['operating_policy'], cached_pairs(read(folder / f'{split}_pairs.json')))
        evidence[key] = summary
        write(folder / f'{split}_scores.json', rows)
    # Replayed development data cannot become new independent research approval.
    evidence['outer_proposal_gate_passes'] = False
    write(folder / 'evidence.json', evidence)
    with (folder / 'rendered_verification.log').open('w', encoding='utf-8') as log, redirect_stdout(log):
        verification = replay(folder, diagnostic=True)
    build_demo_card(folder, verification)
    verify_inputs(folder)
    return verification


def build_demo_card(folder, verification):
    folder = run_folder(folder)
    checkpoint = folder / 'selected_model.joblib'
    originals, shifts = verification['originals'], verification['shifts']
    valid = [r for r in shifts if 'excluded' not in r]
    improved = [r for r in valid if r.get('verified_improvement')]
    if (len(originals) != 4 or not all(r.get('preserved') for r in originals) or len(shifts) != 16
            or not improved or verification['checkpoint_sha256'] != checksum(checkpoint)):
        raise ValueError('The new checkpoint lacks preserved originals and a verified audio improvement; demo not activated')
    selected = improved[0]
    original = next(r for r in originals if r['input'] == selected['reference'])
    before = float(np.mean([r['input_mae_cents'] for r in valid]))
    after = float(np.mean([r['output_mae_cents'] for r in valid]))
    metrics = dict(model_type='extra_trees', attempted_shifts=len(shifts), audited_shifts=len(valid),
        singers_with_valid_shift_labels=len({r['singer'] for r in valid}),
        heldout_singers=len(verification['by_singer']), verified_improvements=len(improved),
        delivered_previews=sum(r['delivered_preview'] for r in valid),
        originals_preserved=len(originals), originals_tested=len(originals),
        input_mae_cents=before, output_mae_cents=after, error_reduction_fraction=1-after/before,
        by_singer=verification['by_singer'])
    cases = [dict(id='learned_pitch', label='Notebook-trained model: real voice with a controlled local shift',
        input=selected['input'], reference=selected['reference'], score_mask_path=selected['target_path'],
        source_kind='Post-hoc selected verified example from reused development pilot; reference/mask enter scoring only.'),
        dict(id='learned_original', label='Notebook-trained model: preservation of an unchanged real voice',
        input=original['input'], reference=None, source_kind='Original working audio from the reused development pilot.')]
    card = dict(version='notebook-trained-preview-1', notebook_training=True, eligible_for_demo=True,
        validation_status='exploratory_preview_failed_generalization_gate', research_gates_passed=False,
        requires_user_review=True, checkpoint=str(checkpoint), checkpoint_sha256=checksum(checkpoint),
        verification_sha256=checksum(folder / 'rendered_verification.json'),
        fit_receipt_sha256=checksum(folder / 'fit_receipt.json'),
        scope='Reproduced fixed ExtraTrees; reused development data; not validated natural-error repair',
        metrics=metrics, operating_policy=read(folder / 'locked_selection.json')['operating_policy'],
        demo_cases=cases, evidence=str(folder / 'evidence.json'),
        example_selection='First verified improvement in existing source/shift order; selected after replay.')
    write(folder / 'demo_model.json', card)
    return card


def activate_demo(folder):
    """Publish only the separate active-demo pointer; never rewrite historical cards."""
    from bpdr_input.repair.learned_runtime import load_model
    folder = run_folder(folder)
    verify_inputs(folder)
    card_path = folder / 'demo_model.json'
    load_model(card_path=card_path)
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    if POINTER.exists():
        (folder / 'previous_active_pointer.json').write_bytes(POINTER.read_bytes())
    temporary = POINTER.parent / (uuid.uuid4().hex + '.tmp')
    write(temporary, dict(card=str(card_path), card_sha256=checksum(card_path),
        training_run=str(folder), checkpoint_sha256=checksum(folder / 'selected_model.joblib')))
    os.replace(temporary, POINTER)
    return read(POINTER)


def active_training_run():
    from bpdr_input.repair.learned_runtime import active_card_path, load_model
    card_path = active_card_path()
    _, _, card = load_model(card_path=card_path)
    if not card.get('notebook_training'):
        raise ValueError('No notebook-trained checkpoint yet. Set TRAIN_NEW_MODEL=True in notebook 04 first.')
    return run_folder(card_path.parent)
