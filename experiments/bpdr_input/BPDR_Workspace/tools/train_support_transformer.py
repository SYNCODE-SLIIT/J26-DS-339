"""Matched continued-point versus complete-support training; fixed WAV gates."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import support_transformer_training as training
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.correction_accuracy import summarize
import train_response_transformer as previous

ROOT=WORKSPACE/'results/transformer_usability/2026-10-29_v16_support'
BASE=ROOT.parent/'2026-10-29_v12_response'
ARMS=['continued_point','support_blocks']


def declare():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        reference=ex.read(BASE/'declaration.json')
        previous_d=ex.read(BASE/'declaration.json')
        files={Path(p) for p in previous_d['source_hashes']}
        files.update([Path(__file__),Path(training.__file__),
                      WORKSPACE/'source/bpdr_input/experiments/learning/support_supervision.py',
                      WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py',
                      WORKSPACE/'source/bpdr_input/experiments/learning/held_voice_training.py',
                      WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py',
                      WORKSPACE/'source/bpdr_input/repair/repair_controller.py',
                      WORKSPACE/'source/bpdr_input/modeling/step_context.py',
                      WORKSPACE/'source/bpdr_input/modeling/step_transformer.py',Path(previous.__file__)])
        cv_parents={str(f):str(BASE/f'cross_validation/fold_{f}/matched_baseline/checkpoint.pt') for f in range(1,5)}
        full_parents={str(s):str(ROOT.parent/f'2026-10-28_v8_joint/final/seed_{s}/checkpoint.pt') for s in (339,340,341)}
        # Full parent paths are recorded only after checking the canonical V8 declaration location.
        if not all(Path(p).exists() for p in full_parents.values()):
            candidates=[p for p in ROOT.parent.glob('*v8*') if (p/'final/seed_339/checkpoint.pt').exists()]
            if len(candidates)!=1:raise ValueError('Ambiguous full-refit parent location')
            full_parents={str(s):str(candidates[0]/f'final/seed_{s}/checkpoint.pt') for s in (339,340,341)}
        parent_paths=list(cv_parents.values())+list(full_parents.values())
        d=dict(created_at=datetime.now(timezone.utc).isoformat(),model_family='joint_context',
               arms=['matched_baseline']+ARMS,trained_arms=ARMS,folds=reference['folds'],
               cv_seed=339,cv_updates=600,final_seeds=[339,340,341],default_seed=339,final_updates=1200,
               cv_parents=cv_parents,full_parents=full_parents,
               parent_hashes={p:ex.sha(p) for p in parent_paths},
               architecture='Unchanged eight-channel broad+step two-layer Transformer. Both conditions load exactly the same partition parent; architecture and inference are unchanged.',
               training='Both conditions retain original point losses and uniform originals, AdamW lr0.0002; compute identical86 block windows in addition to36 point windows. Support arm adds weight0.25 block objective: dense paired labels, activity and conservative signed soft-min margin across43 genuinely known contiguous frames. No new audio, altered labels or runtime support.',
               comparison='Two new conditions share parent, architecture, optimizer, update budget, point and block sampling traces and122 window forwards per update. Only block objective weight differs0 versus0.25. Frozen V12 reference has fewer updates; no matched-budget superiority claim against it. Each CV child has1200 parent+600 new updates. Full refits2400+1200.',
               initial_contract='Before SGD compare original and corrupted probe predictions against corresponding partition parent. Parent singer/source sets and seed must match exactly.',
               selection=reference['selection'],policy=reference['policy'],curve_condition='median11',renderer='Praat unchanged',
               baseline_evidence=str(BASE/'arms/matched_baseline/training/median11/evidence.json'),
               baseline_evidence_sha256=ex.sha(BASE/'arms/matched_baseline/training/median11/evidence.json'),
               baseline_cv=str(BASE/'cross_validation.json'),baseline_cv_sha256=ex.sha(BASE/'cross_validation.json'),
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
        rows={a:[] for a in ARMS}
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks={pool.submit(training.cv_worker,(str(ROOT),f,a)):(f,a) for a in ARMS for f in range(4)}
            for task in as_completed(tasks):
                fold,arm=tasks[task];result=task.result();rows[arm].extend(result['rows'])
                print('SUPPORT_CV_COMPLETE',arm,fold+1,result['summary']['raw_mae_cents'],flush=True)
        for f in range(1,5):
            receipts=[ex.read(ROOT/f'cross_validation/fold_{f}/{a}/fit_receipt.json') for a in ARMS]
            for key in ('sampling_sha256','block_sampling_sha256','parent_checkpoint_sha256'):
                assert receipts[0][key]==receipts[1][key],key
        results={'matched_baseline':ex.read(d['baseline_cv'])['results']['matched_baseline']}
        results.update({a:summarize(rr) for a,rr in rows.items()})
        ex.write(ROOT/'cross_validation.json',dict(results=results,new_neural_fits=8,reused_reference_fits=4,
                 new_arms_matched_budget=True,all_sampling_traces_match=True,additional_updates=600))
        print('SUPPORT_CV_FINISHED',{a:v['summary'] for a,v in results.items()},flush=True)
    elif args.stage=='delivery':
        previous.ROOT=ROOT
        cv=ex.read(ROOT/'cross_validation.json')
        results={'matched_baseline':ex.read(d['baseline_evidence'])}
        for arm in ARMS:results[arm]=previous.delivery(d,arm)
        selected=min(d['arms'],key=lambda a:(not cv['results'][a]['prediction_gate_passed'],
                     not results[a]['gates']['original_controls_all_identity'],
                     results[a]['summary']['delivered_after_mae_cents'],d['arms'].index(a)))
        ex.write(ROOT/'locked_selection.json',dict(arm=selected,results=results,
                 prediction_gate_passed=cv['results'][selected]['prediction_gate_passed'],
                 all_originals_identity=results[selected]['gates']['original_controls_all_identity'],
                 default_seed=339,diagnostic_only=True,selection_data='Training-side prediction/original gates then actual WAV error',
                 demo_activated=False,research_validation_passed=False))
        print('SUPPORT_DELIVERY_SELECTION',selected,flush=True)
    else:
        selected=ex.read(ROOT/'locked_selection.json')
        if selected['arm'] not in ARMS:raise ValueError('New condition not selected; no duplicate baseline refit')
        if not selected['prediction_gate_passed'] or not selected['all_originals_identity']:raise ValueError('Prediction/original gates failed; no refit')
        pairs=training.held.prepared(ex.read(ROOT/'train_pairs.json'));anchors=training.held.anchor_pool(ex.read(ROOT/'anchor_sources.json'))
        for seed in d['final_seeds']:
            folder=ROOT/f'final/seed_{seed}'
            if (folder/'fit_receipt.json').exists():continue
            training.fit(pairs,anchors,selected['arm'],seed,d['final_updates'],folder,ex.sha(ROOT/'declaration.json'),Path(d['full_parents'][str(seed)]))
            print('SUPPORT_REFIT_COMPLETE',seed,flush=True)
    declare()


if __name__=='__main__':main()
