"""Frozen learned-center readout ablation; no new fitting or fresh evaluation."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex,held_voice_training as held
from bpdr_input.modeling.joint_transformer import load
from bpdr_input.modeling.candidate_center_projection import project,predict_raw,VERSION
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.reporting.transformer_accuracy_report import complete_report
import run_transformer_curve_study as curve

ROOT=WORKSPACE/'results/transformer_usability/2026-10-30_v24_center_readout'
BASE=ROOT.parent/'2026-10-29_v12_response'


def declare():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        reference=ex.read(BASE/'declaration.json')
        files=[Path(__file__).resolve(),Path(ex.__file__),Path(curve.__file__),
               WORKSPACE/'source/bpdr_input/modeling/candidate_center_projection.py',
               WORKSPACE/'source/bpdr_input/modeling/step_candidates.py',
               WORKSPACE/'source/bpdr_input/modeling/step_context.py',
               WORKSPACE/'source/bpdr_input/modeling/joint_transformer.py',
               WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py',
               WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py',
               WORKSPACE/'source/bpdr_input/repair/curve_conditioning.py',
               WORKSPACE/'source/bpdr_input/repair/repair_controller.py',
               WORKSPACE/'source/bpdr_input/repair/framewise_audit.py',
               WORKSPACE/'source/bpdr_input/repair/framewise_renderer.py']
        folders=[BASE/f'cross_validation/fold_{f}/matched_baseline' for f in range(1,5)]
        full={str(s):str(ROOT.parent/f'2026-10-28_v8_joint/final/seed_{s}/checkpoint.pt') for s in (339,340,341)}
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
                 model_family='joint_candidate_center_readout',decoder_version=VERSION,folds=reference['folds'],default_seed=339,
                 rule='For every input-measured candidate>=43 frames, repeat the ORIGINAL learned prediction at its measured center. Keep dense predictions elsewhere. No fitted head, pooling change, target routing, padding, clipping or threshold change.',
                 comparison='Diagnostic readout ablation of original frozen Transformer, not a learned region-head or matched training-budget superiority claim.',
                 policy=reference['policy'],curve_condition='median11',renderer='Praat unchanged',
                 source_hashes={str(p):ex.sha(p) for p in files},input_hashes=reference['input_hashes'],
                 checkpoint_hashes={str(p/'checkpoint.pt'):ex.sha(p/'checkpoint.pt') for p in folders},
                 prediction_hashes={str(p/'validation/predictions.npz'):ex.sha(p/'validation/predictions.npz') for p in folders},
                 full_refits=full,full_refit_hashes={p:ex.sha(p) for p in full.values()},
                 baseline_cv=str(BASE/'cross_validation.json'),baseline_cv_sha256=ex.sha(BASE/'cross_validation.json'),
                 baseline_evidence=str(BASE/'arms/matched_baseline/training/median11/evidence.json'),
                 baseline_evidence_sha256=ex.sha(BASE/'arms/matched_baseline/training/median11/evidence.json'),
                 selection='All-singer prediction gate, all50 original WAV identities, then lower singer-macro delivered error. Tie retain reference. Known development only after locked new selection; no confirmation access.',
                 new_neural_fits=0,demo_activated=False,research_validation_passed=False))
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for p in files:(snapshot/p.name).write_bytes(p.read_bytes())
        for name in ('train_pairs.json','anchor_sources.json'):(ROOT/name).write_bytes((BASE/name).read_bytes())
    d=ex.read(ROOT/'declaration.json')
    for label in ('source_hashes','input_hashes','checkpoint_hashes','prediction_hashes','full_refit_hashes'):
        for path,digest in d[label].items():assert ex.sha(path)==digest,path
    for label in ('baseline_cv','baseline_evidence'):assert ex.sha(d[label])==d[label+'_sha256']
    return d


def build(d,rows,pairs):
    cache={};models={};pair_map={p['path_C']:p for p in pairs if 'excluded' not in p}
    for fold,groups in enumerate(d['folds'],1):
        folder=BASE/f'cross_validation/fold_{fold}/matched_baseline';model,payload=load(folder/'checkpoint.pt')
        subset=[p for p in rows if ex.group(p) in groups]
        assert_independent(payload['training_groups'],payload['training_sources'],subset)
        for group in groups:models[group]=(model,payload)
        with np.load(folder/'validation/predictions.npz') as z:
            for i,p in enumerate(subset):
                if 'excluded' in p:continue
                pair=pair_map[p['path_C']]
                for view in ('A','C'):
                    cache[p['path_'+view]]=np.stack([project(pair[view]['views'][0],z[f'{i}_{view}'][0])]*3)
        probe=next(p for p in subset if 'excluded' not in p);pair=pair_map[probe['path_C']]
        np.testing.assert_allclose(predict_raw(model,pair['C']['views'][0])[0],cache[probe['path_C']][0],atol=1e-5)
    return cache,models


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['cv','delivery']);args=parser.parse_args()
    d=declare();rows=ex.read(ROOT/'train_pairs.json');pairs=held.prepared(rows);cache,models=build(d,rows,pairs)
    if args.stage=='cv':
        folder=ROOT/'validation';folder.mkdir(exist_ok=True)
        if (folder/'accuracy.json').exists():raise ValueError('Existing readout result retained')
        ex.write(folder/'evaluation_declaration.json',dict(pairs_manifest=str(ROOT/'train_pairs.json'),role='Training-side held-out frozen center readout, not fresh'))
        old=(ex.predictions,ex.legacy_proposal)
        ex.predictions=lambda model,p,view:cache[p['path_'+view]]
        ex.legacy_proposal=lambda p,view,pp:curve.conditioned_proposal(p,view,pp,curve.POLICY,'median11')
        try:result=ex.evaluate(None,dict(seed=339),pairs,folder,render=False,seed=339)
        finally:ex.predictions,ex.legacy_proposal=old
        result.update(model_family=d['model_family'],decoder_version=VERSION)
        ex.write(folder/'accuracy.json',result);ex.write_report(folder,result,'Frozen learned-center readout');complete_report(folder)
        ex.write(ROOT/'cross_validation.json',dict(results={'center_readout':result},new_neural_fits=0))
        print('CENTER_READOUT_CV',result['prediction_gate_passed'],result['summary'],flush=True)
    else:
        folder=ROOT/'arm';folder.mkdir(exist_ok=True)
        if not (folder/'declaration.json').exists():ex.write(folder/'declaration.json',dict(parent=str(ROOT/'declaration.json'),parent_sha256=ex.sha(ROOT/'declaration.json')))
        oldroot,oldpredict=curve.ROOT,curve.predict_raw;curve.ROOT=folder;curve.predict_raw=predict_raw
        try:evidence=curve.run_condition(rows,pairs,cache,models,ex.read(ROOT/'anchor_sources.json'),'median11','training',ROOT/'train_pairs.json')
        finally:curve.ROOT,curve.predict_raw=oldroot,oldpredict
        cv=ex.read(ROOT/'cross_validation.json')['results']['center_readout'];baseline=ex.read(d['baseline_evidence'])
        selected=bool(cv['prediction_gate_passed'] and evidence['gates']['original_controls_all_identity']
                      and evidence['summary']['delivered_after_mae_cents']<baseline['summary']['delivered_after_mae_cents'])
        ex.write(ROOT/'locked_selection.json',dict(arm='center_readout' if selected else 'matched_baseline',
                 prediction_gate_passed=cv['prediction_gate_passed'],all_originals_identity=evidence['gates']['original_controls_all_identity'],
                 diagnostic_only=True,demo_activated=False,research_validation_passed=False))
        print('CENTER_READOUT_DELIVERY',evidence['gates']['cohort_improved'],'selected',selected,flush=True)
    declare()


if __name__=='__main__':main()
