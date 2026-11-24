"""One new input-context condition against a frozen, verified baseline."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import localized_transformer_training as training
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.correction_accuracy import summarize
import train_response_transformer as workflow

ROOT = WORKSPACE/'results/transformer_usability/2026-10-29_v13_localized'
BASE = ROOT.parent/'2026-10-29_v12_response'
ARM = 'localized_context'


def declare():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        parent = ex.read(BASE/'declaration.json')
        baseline = BASE/'arms/matched_baseline/training/median11/evidence.json'
        files = [Path(__file__), Path(training.__file__), Path(workflow.__file__),
                 WORKSPACE/'source/bpdr_input/modeling/joint_transformer.py',
                 WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py',
                 WORKSPACE/'source/bpdr_input/modeling/step_context.py',
                 WORKSPACE/'source/bpdr_input/modeling/step_transformer.py',
                 WORKSPACE/'source/bpdr_input/experiments/learning/joint_transformer_training.py',
                 WORKSPACE/'source/bpdr_input/repair/curve_conditioning.py',
                 WORKSPACE/'source/bpdr_input/repair/framewise_renderer.py',
                 WORKSPACE/'source/bpdr_input/repair/framewise_audit.py']
        d = dict(created_at=datetime.now(timezone.utc).isoformat(), model_family='joint_context',
                 arms=['matched_baseline', ARM], trained_arms=[ARM], folds=parent['folds'],
                 cv_seed=339, cv_updates=1200, final_seeds=[339,340,341], default_seed=339, final_updates=2400,
                 change='Only input config.scope_same_run=True. Continuous context and attention padding use the current audio-derived reliable +/-70c region. Provisional boundary metadata excluded; no label/reference input.',
                 matching='Same original point loss, positive weight1,36 windows/update, initialization, sampling, augmentation, optimizer and updates as verified V12 baseline. All200 attempts remain; no evaluation scope filtering.',
                 selection=parent['selection'], policy=parent['policy'], curve_condition='median11', renderer='Praat unchanged',
                 baseline_evidence=str(baseline), baseline_evidence_sha256=ex.sha(baseline),
                 baseline_cv=str(BASE/'cross_validation.json'), baseline_cv_sha256=ex.sha(BASE/'cross_validation.json'),
                 baseline_checkpoints={str(BASE/f'cross_validation/fold_{f}/matched_baseline/checkpoint.pt'):
                        ex.sha(BASE/f'cross_validation/fold_{f}/matched_baseline/checkpoint.pt') for f in range(1,5)},
                 source_hashes={str(p):ex.sha(p) for p in files}, input_hashes=parent['input_hashes'],
                 new_confirmation=parent['new_confirmation'],
                 research_validation_passed=False, demo_activated=False)
        ex.write(ROOT/'declaration.json', d)
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for p in files:(snapshot/p.name).write_bytes(p.read_bytes())
        for name in ('train_pairs.json','anchor_sources.json'):(ROOT/name).write_bytes((BASE/name).read_bytes())
    d=ex.read(ROOT/'declaration.json')
    for name in ('source_hashes','input_hashes','baseline_checkpoints'):
        for p,h in d[name].items():assert ex.sha(p)==h,p
    assert ex.sha(d['baseline_evidence'])==d['baseline_evidence_sha256']
    assert ex.sha(d['baseline_cv'])==d['baseline_cv_sha256']
    return d


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['cv','delivery','refit'])
    args=parser.parse_args();d=declare()
    if args.stage=='cv':
        rows=[]
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks={pool.submit(training.cv_worker,(str(ROOT),f,ARM)):f for f in range(4)}
            for task in as_completed(tasks):
                fold=tasks[task];result=task.result();rows.extend(result['rows'])
                before=ex.read(BASE/f'cross_validation/fold_{fold+1}/matched_baseline/fit_receipt.json')
                after=ex.read(ROOT/f'cross_validation/fold_{fold+1}/{ARM}/fit_receipt.json')
                assert before['sampling_sha256']==after['sampling_sha256']
                print('LOCALIZED_CV_COMPLETE',fold+1,result['summary']['raw_mae_cents'],flush=True)
        baseline=ex.read(d['baseline_cv'])['results']['matched_baseline']
        results=dict(matched_baseline=baseline,localized_context=summarize(rows))
        ex.write(ROOT/'cross_validation.json',dict(results=results,new_neural_fits=4,reused_baseline_fits=4,
                 baseline_is_existing_evidence=True,sampling_matched_all_folds=True))
        print('LOCALIZED_CV_FINISHED',{a:r['summary'] for a,r in results.items()},flush=True)
    elif args.stage=='delivery':
        cv=ex.read(ROOT/'cross_validation.json')
        workflow.ROOT=ROOT;workflow.training=training
        localized=workflow.delivery(d,ARM)
        results=dict(matched_baseline=ex.read(d['baseline_evidence']),localized_context=localized)
        selected=min(d['arms'],key=lambda a:(not cv['results'][a]['prediction_gate_passed'],
                     not results[a]['gates']['original_controls_all_identity'],
                     results[a]['summary']['delivered_after_mae_cents'],d['arms'].index(a)))
        ex.write(ROOT/'locked_selection.json',dict(arm=selected,results=results,
                 prediction_gate_passed=cv['results'][selected]['prediction_gate_passed'],
                 all_originals_identity=results[selected]['gates']['original_controls_all_identity'],
                 default_seed=339,diagnostic_only=True,selection_data='Training-side gates and actual WAV only',
                 demo_activated=False,research_validation_passed=False))
        print('LOCALIZED_DELIVERY_SELECTION',selected,flush=True)
    else:
        selected=ex.read(ROOT/'locked_selection.json')
        if selected['arm']!=ARM:
            raise ValueError('No new condition selected; retain existing failed baseline without duplicate refits')
        if not selected['prediction_gate_passed'] or not selected['all_originals_identity']:
            raise ValueError('Prediction/original gates failed; no refit')
        pairs=training.held.prepared(ex.read(ROOT/'train_pairs.json'))
        anchors=training.held.anchor_pool(ex.read(ROOT/'anchor_sources.json'))
        for seed in d['final_seeds']:
            folder=ROOT/f'final/seed_{seed}'
            if (folder/'fit_receipt.json').exists():continue
            training.fit(pairs,anchors,ARM,seed,d['final_updates'],folder,ex.sha(ROOT/'declaration.json'))
            print('LOCALIZED_REFIT_COMPLETE',seed,flush=True)
    declare()


if __name__=='__main__':main()
