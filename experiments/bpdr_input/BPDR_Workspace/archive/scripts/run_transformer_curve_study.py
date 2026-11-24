"""Frozen-model curve study; training-side selection precedes development."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.modeling.joint_transformer import load, predict_raw
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair.curve_conditioning import CONDITIONS, conditioned_proposal
from bpdr_input.repair.held_voice_policy import HeldPolicy
from bpdr_input.reporting.represented_singer_gates import assess
from bpdr_input.reporting.transformer_accuracy_report import complete_report

ROOT = WORKSPACE/'results/transformer_usability/2026-10-28_v9_curve'
UPSTREAM = ROOT.parent/'2026-10-28_v8_joint'
POLICY = HeldPolicy(version='held-voice-input-policy-2', activation_cents=8.,
                    region_split_cents=70., minimum_step_fit_gain=.75)


def declare():
    ROOT.mkdir(exist_ok=True)
    sources = [Path(__file__), WORKSPACE/'source/bpdr_input/repair/curve_conditioning.py',
               WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py',
               WORKSPACE/'source/bpdr_input/repair/repair_controller.py',
               WORKSPACE/'source/bpdr_input/repair/framewise_audit.py',
               WORKSPACE/'source/bpdr_input/repair/framewise_renderer.py']
    if not (ROOT/'declaration.json').exists():
        checkpoints = list(UPSTREAM.glob('cross_validation/**/joint_augmented/checkpoint.pt'))
        checkpoints += [UPSTREAM/'final/seed_339/checkpoint.pt']
        assert len(checkpoints) == 5
        ex.write(ROOT/'declaration.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
                 conditions=list(CONDITIONS), source_model_run=str(UPSTREAM), policy=POLICY.__dict__,
                 selection='Training-side out-of-fold actual-WAV results only. First require all original controls identity; then minimize singer-macro delivered error. Tie prefer unchanged. No new model fit.',
                 constraint='Retain exact existing permission supports, all numeric limits, protected/reliable masks and dual-tracker plus encoded-WAV audits. No reference/target enters conditioning.',
                 model_checkpoints={str(p): ex.sha(p) for p in checkpoints},
                 source_hashes={str(p): ex.sha(p) for p in sources},
                 training_input_hashes=ex.read(UPSTREAM/'training_input_hashes.json'),
                 historical_development='IDs5/6 already inspected; not fresh final evidence',
                 final_cohort_access=False, demo_activated=False, research_validation_passed=False))
        snapshot = ROOT/'executed_sources'
        snapshot.mkdir(exist_ok=True)
        for p in sources:
            (snapshot/p.name).write_bytes(p.read_bytes())
    d = ex.read(ROOT/'declaration.json')
    for name in ('model_checkpoints', 'source_hashes', 'training_input_hashes'):
        for p, h in d[name].items():
            assert ex.sha(p) == h, p
    return d


def training_cache(rows):
    cache, models = {}, {}
    d = ex.read(UPSTREAM/'declaration.json')
    for fold, groups in enumerate(d['folds'], 1):
        folder = UPSTREAM/f'cross_validation/fold_{fold}/joint_augmented'
        model, payload = load(folder/'checkpoint.pt')
        subset = [p for p in rows if ex.group(p) in groups]
        assert_independent(payload['training_groups'], payload['training_sources'], subset)
        for group in groups:
            models[group] = (model, payload)
        with np.load(folder/'validation/predictions.npz') as z:
            for i, p in enumerate(subset):
                if 'excluded' not in p:
                    for view in ('A', 'C'):
                        cache[p['path_'+view]] = z[f'{i}_{view}'].copy()
    return cache, models


def original_controls(source_rows, models, kind, folder):
    result, cache = [], {}
    for i, source in enumerate(source_rows):
        path = Path(source['working_audio'])
        audio, analysis = recovery.from_cached(str(path), path.parent/'original_features.npz', cache)
        group = source['source']['singer_id']
        model, payload = models[group]
        assert_independent(payload['training_groups'], payload['training_sources'], [source])
        one = predict_raw(model, analysis['views'][0])[0]
        pred = np.stack([one]*3)
        pair = dict(audio_A=audio, A=analysis)
        curve, plan, _ = conditioned_proposal(pair, 'A', pred, POLICY, kind)
        output, transaction = ex.rendered_output(pair, 'A', curve, plan, folder/f'original_{i:02d}')
        reliable = analysis['reliable']
        result.append(dict(group=group, source_sha256=source['source']['sha256'],
                    raw_original_abs_cents=float(abs(one[reliable]).mean()) if reliable.any() else None,
                    delivered_identity=bool(np.array_equal(audio, output)), outcome=transaction['outcome'],
                    transaction=transaction, reliable_frames=int(reliable.sum())))
    ex.write(folder/'original_controls.json', result)
    return result


def run_condition(rows, pairs, predictions, models, sources, kind, role, manifest):
    folder = ROOT/role/kind
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'evidence.json').exists():
        return ex.read(folder/'evidence.json')
    ex.write(folder/'evaluation_declaration.json', dict(pairs_manifest=str(manifest),
             role='Training-side singer-held-out curve selection' if role == 'training' else 'Previously inspected development replay',
             curve_condition=kind, policy=POLICY.__dict__, source_study_declaration_sha256=ex.sha(ROOT/'declaration.json')))
    old_pred, old_prop = ex.predictions, ex.legacy_proposal
    ex.predictions = lambda model, p, view: predictions[p['path_'+view]]
    ex.legacy_proposal = lambda p, view, pp: conditioned_proposal(p, view, pp, POLICY, kind)
    try:
        result = ex.evaluate(None, dict(seed=339), pairs, folder, render=True, seed=339)
    finally:
        ex.predictions, ex.legacy_proposal = old_pred, old_prop
    result.update(eligibility_policy=POLICY.__dict__, curve_condition=kind,
                  metric_scope='Frozen Transformer plus curve shaping; engineering WAV evidence, not musical accuracy')
    ex.write(folder/'accuracy.json', result)
    ex.write_report(folder, result, 'Frozen Transformer curve study')
    complete_report(folder)
    controls = original_controls(sources, models, kind, folder)
    gates = assess(result, controls)
    ex.write(folder/'gates.json', gates)
    evidence = dict(condition=kind, summary=result['summary'], gates=gates,
                    original_controls=len(controls), accuracy_sha256=ex.sha(folder/'accuracy.json'),
                    controls_sha256=ex.sha(folder/'original_controls.json'),
                    research_validation_passed=False, demo_activated=False)
    ex.write(folder/'evidence.json', evidence)
    print('CURVE_CONDITION_COMPLETE', role, kind, gates['cohort_improved'], '/', gates['cohort_attempted'],
          'originals', gates['original_controls_all_identity'], flush=True)
    return evidence


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['training', 'development'])
    args = parser.parse_args()
    declare()
    if args.stage == 'training':
        manifest = UPSTREAM/'train_pairs.json'
        rows = ex.read(manifest)
        pairs = held.prepared(rows)
        cache, models = training_cache(rows)
        sources = ex.read(UPSTREAM/'anchor_sources.json')
        evidence = {}
        for kind in CONDITIONS:
            evidence[kind] = run_condition(rows, pairs, cache, models, sources, kind, 'training', manifest)
            ex.write(ROOT/'progress.json', dict(completed_training_conditions=list(evidence), planned=list(CONDITIONS)))
        selected = min(CONDITIONS, key=lambda c: (not evidence[c]['gates']['original_controls_all_identity'],
                         evidence[c]['summary']['delivered_after_mae_cents'], CONDITIONS.index(c)))
        ex.write(ROOT/'locked_selection.json', dict(condition=selected, results=evidence,
                 selection_data='Training-side out-of-fold actual-WAV only', diagnostic_only=True,
                 model_refit=False, demo_activated=False, research_validation_passed=False))
        print('CURVE_SELECTION_LOCKED', selected, flush=True)
    else:
        selection = ex.read(ROOT/'locked_selection.json')
        source = UPSTREAM/'development'
        manifest = source/'pairs.json'
        rows = ex.read(manifest)
        model, payload = load(UPSTREAM/'final/seed_339/checkpoint.pt')
        assert_independent(payload['training_groups'], payload['training_sources'], rows)
        pairs = held.prepared(rows)
        cache = {}
        for p in pairs:
            if 'excluded' in p:
                continue
            for view in ('A', 'C'):
                if p['path_'+view] not in cache:
                    cache[p['path_'+view]] = np.stack([predict_raw(model, p[view]['views'][0])[0]]*3)
        # Original sources are references to the already prepared V7 development
        # originals, not previously reserved or newly inspected recordings.
        original_folder = UPSTREAM.parent/'2026-10-28_v7_guard/development'
        sources = [ex.read(p)['source'] for p in sorted(original_folder.glob('source_*.json'))]
        models = {s['source']['singer_id']: (model, payload) for s in sources}
        run_condition(rows, pairs, cache, models, sources, selection['condition'], 'development', manifest)
    declare()  # Recheck immutable weights/inputs/source hashes after execution.


if __name__ == '__main__':
    main()
