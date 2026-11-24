"""Check a qualified response-model refit on known development WAVs only."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.modeling.joint_transformer import load, predict_raw
from bpdr_input.modeling.correction_accuracy import assert_independent
import run_transformer_curve_study as curve


def fitting_rows(root,declaration,arm):
    path=Path(root)/'train_pairs.json'
    if 'fit_manifests' in declaration:
        path=Path(declaration['fit_manifests'][arm]).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError('Fitting manifest must be locked inside its model run')
        if ex.sha(path)!=declaration['fit_manifest_hashes'][arm]:
            raise ValueError('Fitting manifest changed after declaration')
    rows=ex.read(path)
    approved=declaration.get('approved_training_sources')
    if approved is not None:
        for row in rows:
            source=row['source']
            if (source['sha256'] not in approved or source['singer_id']!=approved[source['sha256']]
                    or ex.group(row)!=source['singer_id']):
                raise ValueError('Unapproved source or singer in fitting manifest')
    return [p for p in rows if 'excluded' not in p],ex.sha(path)


def lock_model(root):
    root = Path(root).resolve()
    if not root.is_relative_to(WORKSPACE):
        raise ValueError('Model run must remain in BPDR_Workspace')
    declaration = ex.read(root/'declaration.json')
    selection = ex.read(root/'locked_selection.json')
    if not selection['prediction_gate_passed'] or not selection['all_originals_identity']:
        raise ValueError('Development waits for prediction and original-preservation gates')
    if selection['arm'] == 'matched_baseline':
        raise ValueError('No improved condition selected; the existing baseline remains a reference')
    if (declaration['policy'] != curve.POLICY.__dict__ or declaration['curve_condition'] != 'median11'
            or declaration['renderer'] != 'Praat unchanged'):
        raise ValueError('Unexpected policy, curve or renderer; declare a separate experiment')
    for p, h in declaration['source_hashes'].items():
        if ex.sha(p) != h:
            raise ValueError(f'Frozen model/training source changed: {p}')
    for p, h in declaration['input_hashes'].items():
        if ex.sha(p) != h:
            raise ValueError(f'Frozen training input changed: {p}')
    for p, h in declaration.get('parent_hashes', {}).items():
        if ex.sha(p) != h:
            raise ValueError(f'Frozen warm-start parent changed: {p}')
    if 'pool_receipt' in declaration:
        if ex.sha(declaration['pool_receipt'])!=declaration['pool_receipt_sha256']:
            raise ValueError('Audio-pool qualification receipt changed')
        pool=ex.read(declaration['pool_receipt'])
        if not pool['training_ready'] or not pool['base108_all_revalidated_without_label_loss']:
            raise ValueError('Audio pool did not pass label revalidation')
    family = declaration['model_family']
    if family == 'joint_context':
        loader = load
    elif family == 'multiscale_context':
        from bpdr_input.modeling.multiscale_transformer import load as loader
    else:
        raise ValueError('Unsupported candidate model family')
    rows,fit_manifest_hash = fitting_rows(root,declaration,selection['arm'])
    anchors = held.anchor_pool(ex.read(root/'anchor_sources.json'))
    expected_groups = {ex.group(p) for p in rows} | {p['group'] for p in anchors}
    expected_sources = {p['source']['sha256'] for p in rows+anchors}
    hashes = {}
    for seed in declaration['final_seeds']:
        folder = root/f'final/seed_{seed}'
        if not (folder/'fit_receipt.json').exists() or not (folder/'checkpoint.pt').exists():
            raise ValueError('Required qualified refits are incomplete; development is not unlocked')
        receipt = ex.read(folder/'fit_receipt.json')
        expected = ex.sha(folder/'checkpoint.pt')
        if (expected != receipt['checkpoint_sha256'] or not receipt['reload_verified']
                or receipt['declaration_sha256'] != ex.sha(root/'declaration.json')
                or receipt['arm'] != selection['arm'] or receipt['seed'] != seed
                or receipt['updates'] != declaration['final_updates']):
            raise ValueError('Refit receipt differs from declared selection/checkpoint')
        _, fitted = loader(folder/'checkpoint.pt')
        if (set(fitted['training_groups']) != expected_groups
                or set(fitted['training_sources']) != expected_sources
                or fitted['seed'] != seed or fitted['updates'] != declaration['final_updates']
                or fitted['arm'] != selection['arm']):
            raise ValueError('Checkpoint is not the declared full-data refit')
        if 'fit_manifests' in declaration:
            if (fitted.get('fit_manifest_sha256')!=fit_manifest_hash
                    or receipt.get('fit_manifest_sha256')!=fit_manifest_hash):
                raise ValueError('Refit did not use the selected fitting manifest')
        if 'full_parents' in declaration:
            parent_path = declaration['full_parents'][str(seed)]
            _, parent = load(parent_path)
            subset=selection['arm'] in declaration.get('parent_subset_arms',[])
            groups_match=(set(parent['training_groups'])<=expected_groups if subset else set(parent['training_groups'])==expected_groups)
            sources_match=(set(parent['training_sources'])<=expected_sources if subset else set(parent['training_sources'])==expected_sources)
            if (not groups_match or not sources_match or parent['seed'] != seed
                    or fitted.get('parent_checkpoint_sha256') != ex.sha(parent_path)
                    or receipt.get('parent_checkpoint_sha256') != ex.sha(parent_path)
                    or fitted.get('parent_updates') != parent['updates']
                    or fitted.get('total_updates') != parent['updates']+declaration['final_updates']
                    or fitted.get('initial_equivalence_max_error_cents', float('inf')) > 1e-4):
                raise ValueError('Warm-start parent lineage or initial equivalence differs')
        hashes[str(seed)] = expected
    lock = dict(selected=dict(policy=declaration['policy']), checkpoints=hashes,
                default_seed=339, model_family=family, curve_condition='median11',
                model_arm=selection['arm'], source_selection_sha256=ex.sha(root/'locked_selection.json'),
                source_declaration_sha256=ex.sha(root/'declaration.json'))
    path = root/'locked_policy.json'
    if path.exists():
        if ex.read(path) != lock:
            raise ValueError('An existing model/policy lock differs; preserve the old result')
    else:
        ex.write(path, lock)
    model, payload = loader(root/'final/seed_339/checkpoint.pt')
    return model, payload, lock


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=WORKSPACE/'results/transformer_usability/2026-10-29_v12_response')
    parser.add_argument('--replay', action='store_true', help='Create a separate reproducibility replay; not fresh confirmation')
    args = parser.parse_args()
    root = args.run_root.resolve()
    model, payload, lock = lock_model(root)
    inference = predict_raw
    if lock['model_family'] == 'multiscale_context':
        from bpdr_input.modeling.multiscale_transformer import predict_raw as inference
    source = root.parent/'2026-10-28_v8_joint/development'
    manifest = source/'pairs.json'
    rows = ex.read(manifest)
    assert_independent(payload['training_groups'], payload['training_sources'], rows)
    output = root/'accuracy_replays'/uuid.uuid4().hex if args.replay else root/'selected_evaluation'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'development/median11/evidence.json').exists():
        raise ValueError('Development result already exists; use --replay for a separate repeat')
    hashes = ex.input_hashes(rows)
    sources = [ex.read(p)['source'] for p in sorted((root.parent/'2026-10-28_v7_guard/development').glob('source_*.json'))]
    for source_row in sources:
        path = Path(source_row['working_audio'])
        for original_file in (path, path.parent/'original_features.npz'):
            hashes[str(original_file)] = ex.sha(original_file)
    ex.write(output/'declaration.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
             model_policy_lock_sha256=ex.sha(root/'locked_policy.json'), input_hashes=hashes,
             evaluator_sha256=ex.sha(Path(__file__)), dataset_role='Previously inspected development; not fresh confirmation'))
    pairs = held.prepared(rows)
    cache = {}
    for p in pairs:
        if 'excluded' not in p:
            for view in ('A', 'C'):
                if p['path_'+view] not in cache:
                    cache[p['path_'+view]] = np.stack([inference(model, p[view]['views'][0])[0]]*3)
    models = {s['source']['singer_id']: (model, payload) for s in sources}
    curve.ROOT = output
    previous = curve.predict_raw
    curve.predict_raw = inference
    try:
        evidence = curve.run_condition(rows, pairs, cache, models, sources, 'median11', 'development', manifest)
    finally:
        curve.predict_raw = previous
    assert all(ex.sha(p) == h for p, h in hashes.items())
    assert ex.sha(root/'final/seed_339/checkpoint.pt') == lock['checkpoints']['339']
    print('RESPONSE_DEVELOPMENT', evidence['gates'], flush=True)
    print('REPORT', output/'development/median11/index.html', flush=True)


if __name__ == '__main__':
    main()
