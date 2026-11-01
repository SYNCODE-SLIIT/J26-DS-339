"""Frozen three-member OOF ensemble; no fitting or fresh evaluation access."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex,held_voice_training as held,model_recovery as recovery
from bpdr_input.modeling.joint_transformer import load,predict_raw
from bpdr_input.modeling.joint_ensemble import combine,predict_views
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair.ensemble_policy import conditioned_proposal
from bpdr_input.reporting.represented_singer_gates import assess
from bpdr_input.reporting.transformer_accuracy_report import complete_report
import run_transformer_curve_study as curve

ROOT=WORKSPACE/'results/transformer_usability/2026-10-29_v19_ensemble'
MEMBERS=[('reference','2026-10-29_v12_response','matched_baseline'),
         ('support','2026-10-29_v16_support','support_blocks'),
         ('audio','2026-10-29_v18_audio_augmented','audio_augmented')]


def declare():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        base=ROOT.parent/MEMBERS[0][1];data=ex.read(base/'declaration.json')
        checkpoints={};predictions={};parents={}
        for name,run,arm in MEMBERS:
            parent=ROOT.parent/run;parents[str(parent/'declaration.json')]=ex.sha(parent/'declaration.json')
            for fold in range(1,5):
                folder=parent/f'cross_validation/fold_{fold}/{arm}'
                checkpoints[str(folder/'checkpoint.pt')]=ex.sha(folder/'checkpoint.pt')
                predictions[str(folder/'validation/predictions.npz')]=ex.sha(folder/'validation/predictions.npz')
                parents[str(folder/'fit_receipt.json')]=ex.sha(folder/'fit_receipt.json')
        paths=[Path(__file__).resolve(),Path(curve.__file__),Path(ex.__file__),
               WORKSPACE/'source/bpdr_input/modeling/joint_ensemble.py',
               WORKSPACE/'source/bpdr_input/modeling/joint_transformer.py',
               WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py',
               WORKSPACE/'source/bpdr_input/repair/ensemble_policy.py',
               WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py',
               WORKSPACE/'source/bpdr_input/repair/curve_conditioning.py',
               WORKSPACE/'source/bpdr_input/repair/repair_controller.py',
               WORKSPACE/'source/bpdr_input/repair/framewise_audit.py',
               WORKSPACE/'source/bpdr_input/repair/framewise_renderer.py']
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
                 model_family='joint_context_ensemble',members=MEMBERS,folds=data['folds'],default_seed=339,
                 rule='Fixed framewise median of three member outputs; retain min/max envelopes. Primary-only regionization, identical median5 then within-support median11; final controller sees FULL high-low range against unchanged5c uncertainty limit.',
                 constraint='No target/reference routing, no curve clipping, no support extension, no changed numerical or encoded-audio audits. Member dispersion is not calibrated confidence; it is not boundary perturbation evidence.',
                 selection='First raw/original prediction gates; then all50 original WAV identities; then lower singer-macro delivered error than frozen reference. No individual failed member is activated. Composite full refits conditional on a new selected ensemble.',
                 policy=data['policy'],curve_condition='median11',renderer='Praat unchanged',
                 checkpoint_hashes=checkpoints,prediction_hashes=predictions,parent_hashes=parents,
                 source_hashes={str(p):ex.sha(p) for p in paths},
                 input_hashes=ex.read(ROOT.parent/MEMBERS[2][1]/'declaration.json')['input_hashes'],
                 baseline_cv=str(base/'cross_validation.json'),baseline_evidence=str(base/'arms/matched_baseline/training/median11/evidence.json'),
                 baseline_cv_sha256=ex.sha(base/'cross_validation.json'),
                 baseline_evidence_sha256=ex.sha(base/'arms/matched_baseline/training/median11/evidence.json'),
                 fresh_audio_access=False,demo_activated=False,research_validation_passed=False))
        for name in ('train_pairs.json','anchor_sources.json'):(ROOT/name).write_bytes((base/name).read_bytes())
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for p in paths:(snapshot/p.name).write_bytes(p.read_bytes())
    d=ex.read(ROOT/'declaration.json')
    for name in ('checkpoint_hashes','prediction_hashes','parent_hashes','source_hashes','input_hashes'):
        for path,digest in d[name].items():assert ex.sha(path)==digest,path
    for name in ('baseline_cv','baseline_evidence'):assert ex.sha(d[name])==d[name+'_sha256']
    return d


def build(d,rows):
    cache={};models={}
    for fold,groups in enumerate(d['folds'],1):
        subset=[r for r in rows if ex.group(r) in groups]
        loaded=[];payloads=[];arrays=[]
        for name,run,arm in MEMBERS:
            folder=ROOT.parent/run/f'cross_validation/fold_{fold}/{arm}'
            model,payload=load(folder/'checkpoint.pt');receipt=ex.read(folder/'fit_receipt.json')
            assert receipt['checkpoint_sha256']==ex.sha(folder/'checkpoint.pt') and receipt['reload_verified']
            assert payload['declaration_sha256']==ex.sha(ROOT.parent/run/'declaration.json')
            assert_independent(payload['training_groups'],payload['training_sources'],subset)
            loaded.append(model);payloads.append(payload)
            with np.load(folder/'validation/predictions.npz') as z:arrays.append({k:z[k].copy() for k in z.files})
            probe=next((i,p) for i,p in enumerate(subset) if 'excluded' not in p)
            pair=held.prepared([probe[1]])[0]
            np.testing.assert_allclose(predict_raw(model,pair['C']['views'][0])[0],arrays[-1][f'{probe[0]}_C'][0],atol=1e-5)
        union=dict(training_groups=sorted(set().union(*(set(p['training_groups']) for p in payloads))),
                   training_sources=sorted(set().union(*(set(p['training_sources']) for p in payloads))))
        for group in groups:models[group]=(loaded,union)
        for i,p in enumerate(subset):
            if 'excluded' in p:continue
            for view in ('A','C'):cache[p['path_'+view]]=combine([z[f'{i}_{view}'][0] for z in arrays])
    return cache,models


def rename_dispersion(value):
    if isinstance(value,dict):
        return {('member_envelope_deviation_cents' if k=='boundary_disagreement_cents' else k):rename_dispersion(v) for k,v in value.items()}
    if isinstance(value,list):return [rename_dispersion(v) for v in value]
    return value


def controls(sources,models,folder,d):
    result=[];cache={}
    for i,source in enumerate(sources):
        path=Path(source['working_audio']);audio,analysis=recovery.from_cached(str(path),path.parent/'original_features.npz',cache)
        group=source['source']['singer_id'];members,payload=models[group]
        assert_independent(payload['training_groups'],payload['training_sources'],[source])
        pp=predict_views(members,analysis['views'][0])
        curve_values,plan,_=conditioned_proposal(dict(audio_A=audio,A=analysis),'A',pp,curve.POLICY,'median11')
        output,transaction=ex.rendered_output(dict(audio_A=audio,A=analysis),'A',curve_values,plan,folder/f'original_{i:02d}')
        reliable=analysis['reliable']
        result.append(dict(group=group,source_sha256=source['source']['sha256'],
                      raw_original_abs_cents=float(abs(pp[0,reliable]).mean()) if reliable.any() else None,
                      delivered_identity=bool(np.array_equal(output,audio)),transaction=transaction,
                      outcome=transaction['outcome'],reliable_frames=int(reliable.sum())))
    ex.write(folder/'original_controls.json',result);return result


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['cv','delivery']);args=parser.parse_args()
    d=declare();rows=ex.read(ROOT/'train_pairs.json');pairs=held.prepared(rows);cache,models=build(d,rows)
    folder=ROOT/('validation' if args.stage=='cv' else 'training/median11');folder.mkdir(parents=True,exist_ok=True)
    if (folder/'accuracy.json').exists():raise ValueError('Existing outcome retained; do not overwrite an ensemble evaluation')
    ex.write(folder/'evaluation_declaration.json',dict(pairs_manifest=str(ROOT/'train_pairs.json'),
             role='Training-side singer/source-held-out fixed ensemble; not fresh confirmation',
             source_study_declaration_sha256=ex.sha(ROOT/'declaration.json'),member_dispersion_not_boundary_views=True))
    previous=(ex.predictions,ex.legacy_proposal)
    ex.predictions=lambda model,p,view:cache[p['path_'+view]]
    ex.legacy_proposal=lambda p,view,pp:conditioned_proposal(p,view,pp,curve.POLICY,'median11')
    try:result=ex.evaluate(None,dict(seed=339),pairs,folder,render=args.stage=='delivery',seed=339)
    finally:ex.predictions,ex.legacy_proposal=previous
    result=rename_dispersion(result)
    result.update(prediction_views='Median/min/max learned member envelope, NOT boundary views or calibrated confidence',
                  model_family=d['model_family'],eligibility_policy=d['policy'],curve_condition='median11')
    ex.write(folder/'accuracy.json',result);ex.write_report(folder,result,'Fixed three-Transformer ensemble');complete_report(folder)
    if args.stage=='cv':
        ex.write(ROOT/'cross_validation.json',dict(results={'ensemble':result},new_neural_fits=0,frozen_member_checkpoints=12))
        print('ENSEMBLE_CV',result['prediction_gate_passed'],result['summary'],flush=True)
    else:
        original=controls(ex.read(ROOT/'anchor_sources.json'),models,folder,d);gates=assess(result,original)
        evidence=dict(summary=result['summary'],gates=gates,original_controls=len(original),
                      accuracy_sha256=ex.sha(folder/'accuracy.json'),controls_sha256=ex.sha(folder/'original_controls.json'),
                      demo_activated=False,research_validation_passed=False)
        ex.write(folder/'evidence.json',evidence);ex.write(folder/'gates.json',gates)
        base=ex.read(d['baseline_evidence'])
        selected=bool(result['prediction_gate_passed'] and gates['original_controls_all_identity']
                      and result['summary']['delivered_after_mae_cents']<base['summary']['delivered_after_mae_cents'])
        ex.write(ROOT/'locked_selection.json',dict(arm='ensemble' if selected else 'matched_baseline',
                 prediction_gate_passed=result['prediction_gate_passed'],all_originals_identity=gates['original_controls_all_identity'],
                 diagnostic_only=True,demo_activated=False,research_validation_passed=False))
        print('ENSEMBLE_DELIVERY',gates['cohort_improved'],'/',gates['cohort_attempted'],gates['original_controls_all_identity'],'selected',selected,flush=True)
    declare()


if __name__=='__main__':main()
