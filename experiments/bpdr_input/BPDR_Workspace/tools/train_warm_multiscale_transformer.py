"""Prospective warm-start engineering candidate; unchanged delivery qualification."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import warm_multiscale_training as training
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.correction_accuracy import summarize
import train_multiscale_transformer as previous

ROOT=WORKSPACE/'results/transformer_usability/2026-10-29_v15_warm'
BASE=ROOT.parent/'2026-10-29_v12_response'
ARM='warm_multiscale'


def declare():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        reference=ex.read(BASE/'declaration.json')
        previous_d=ex.read(ROOT.parent/'2026-10-29_v14_multiscale/declaration.json')
        files={Path(p) for p in previous_d['source_hashes']}
        files.update([Path(__file__),Path(training.__file__),
                      WORKSPACE/'source/bpdr_input/modeling/multiscale_transfer.py'])
        cv_parents={str(f):str(BASE/f'cross_validation/fold_{f}/matched_baseline/checkpoint.pt') for f in range(1,5)}
        full_parents={str(s):str(ROOT.parent/f'2026-10-28_v8_joint/final/seed_{s}/checkpoint.pt') for s in (339,340,341)}
        # Full parent paths are recorded only after checking the canonical V8 declaration location.
        if not all(Path(p).exists() for p in full_parents.values()):
            candidates=[p for p in ROOT.parent.glob('*v8*') if (p/'final/seed_339/checkpoint.pt').exists()]
            if len(candidates)!=1:raise ValueError('Ambiguous full-refit parent location')
            full_parents={str(s):str(candidates[0]/f'final/seed_{s}/checkpoint.pt') for s in (339,340,341)}
        parent_paths=list(cv_parents.values())+list(full_parents.values())
        d=dict(created_at=datetime.now(timezone.utc).isoformat(),model_family='multiscale_context',
               arms=['matched_baseline',ARM],trained_arms=[ARM],folds=reference['folds'],
               cv_seed=339,cv_updates=600,final_seeds=[339,340,341],default_seed=339,final_updates=1200,
               cv_parents=cv_parents,full_parents=full_parents,
               parent_hashes={p:ex.sha(p) for p in parent_paths},
               architecture='Copy broad-context Transformer; initialize five added local-context projection columns to zero. Same bounded output and architecture as V14.',
               training='AdamW lr0.0002; 50% difficult-original sampling; weight0.5 parent-output preservation on zero-target examples only. Parent frozen. Existing other losses retained.',
               comparison='Engineering candidate with 600 additional CV updates after1200 parent updates; baseline is frozen1200-update V12. This is not a matched-budget architecture or optimizer superiority study. Full refits add1200 to2400 parent updates.',
               initial_contract='Before SGD compare original and corrupted probe predictions against corresponding partition parent. Parent singer/source sets and seed must match exactly.',
               selection=reference['selection'],policy=reference['policy'],curve_condition='median11',renderer='Praat unchanged',
               baseline_evidence=previous_d['baseline_evidence'],baseline_evidence_sha256=previous_d['baseline_evidence_sha256'],
               baseline_cv=previous_d['baseline_cv'],baseline_cv_sha256=previous_d['baseline_cv_sha256'],
               source_hashes={str(p):ex.sha(p) for p in sorted(files)},input_hashes=reference['input_hashes'],
               new_confirmation=reference['new_confirmation'],research_validation_passed=False,demo_activated=False)
        ex.write(ROOT/'declaration.json',d)
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for p in files:(snapshot/p.name).write_bytes(p.read_bytes())
        for name in ('train_pairs.json','anchor_sources.json'):(ROOT/name).write_bytes((BASE/name).read_bytes())
    d=ex.read(ROOT/'declaration.json')
    for name in ('source_hashes','input_hashes','parent_hashes'):
        for p,h in d[name].items():assert ex.sha(p)==h,p
    assert ex.sha(d['baseline_evidence'])==d['baseline_evidence_sha256']
    assert ex.sha(d['baseline_cv'])==d['baseline_cv_sha256']
    return d


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['cv','delivery','refit'])
    args=parser.parse_args();d=declare()
    if args.stage=='cv':
        rows=[]
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks={pool.submit(training.cv_worker,(str(ROOT),f,ARM)):f for f in range(4)}
            for task in as_completed(tasks):
                fold=tasks[task];result=task.result();rows.extend(result['rows'])
                print('WARM_CV_COMPLETE',fold+1,result['summary']['raw_mae_cents'],flush=True)
        results={'matched_baseline':ex.read(d['baseline_cv'])['results']['matched_baseline'],ARM:summarize(rows)}
        ex.write(ROOT/'cross_validation.json',dict(results=results,new_neural_fits=4,reused_reference_fits=4,
                 matched_budget=False,additional_updates=600))
        print('WARM_CV_FINISHED',{a:v['summary'] for a,v in results.items()},flush=True)
    elif args.stage=='delivery':
        previous.ROOT=ROOT
        cv=ex.read(ROOT/'cross_validation.json')
        results={'matched_baseline':ex.read(d['baseline_evidence']),ARM:previous.delivery(d,ARM)}
        selected=min(d['arms'],key=lambda a:(not cv['results'][a]['prediction_gate_passed'],
                     not results[a]['gates']['original_controls_all_identity'],
                     results[a]['summary']['delivered_after_mae_cents'],d['arms'].index(a)))
        ex.write(ROOT/'locked_selection.json',dict(arm=selected,results=results,
                 prediction_gate_passed=cv['results'][selected]['prediction_gate_passed'],
                 all_originals_identity=results[selected]['gates']['original_controls_all_identity'],
                 default_seed=339,diagnostic_only=True,selection_data='Training-side prediction/original gates then actual WAV error',
                 demo_activated=False,research_validation_passed=False))
        print('WARM_DELIVERY_SELECTION',selected,flush=True)
    else:
        selected=ex.read(ROOT/'locked_selection.json')
        if selected['arm']!=ARM:raise ValueError('New condition not selected; no duplicate baseline refit')
        if not selected['prediction_gate_passed'] or not selected['all_originals_identity']:raise ValueError('Prediction/original gates failed; no refit')
        pairs=training.held.prepared(ex.read(ROOT/'train_pairs.json'));anchors=training.held.anchor_pool(ex.read(ROOT/'anchor_sources.json'))
        for seed in d['final_seeds']:
            folder=ROOT/f'final/seed_{seed}'
            if (folder/'fit_receipt.json').exists():continue
            training.fit(pairs,anchors,ARM,seed,d['final_updates'],folder,ex.sha(ROOT/'declaration.json'),Path(d['full_parents'][str(seed)]))
            print('WARM_REFIT_COMPLETE',seed,flush=True)
    declare()


if __name__=='__main__':main()
