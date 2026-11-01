"""Training-only policy selection and locked development/final WAV evaluation."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import held_voice_training as training
from bpdr_input.experiments.learning import context_improvement as existing
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.modeling.held_voice_transformer import load
from bpdr_input.modeling.correction_accuracy import assert_independent,summarize
from bpdr_input.repair.held_voice_policy import HeldPolicy,proposal
from bpdr_input.reporting.transformer_accuracy_report import complete_report

ROOT=training.ROOT


def complete_trace(result,pairs,folder,policy):
    """Add candidate-level rejection reasons without changing measurements."""
    with np.load(folder/'predictions.npz') as arrays:
        for index,(row,pair) in enumerate(zip(result['rows'],pairs)):
            if 'excluded' in pair:continue
            for view in ('A','C'):
                detail=proposal(pair['audio_'+view],pair[view]['features']['times_seconds'],pair[view]['views'],arrays[f'{index}_{view}'],policy)
                row['trace'][view]['candidate_selection']=detail['candidates']
                row['trace'][view]['candidate_reasons']=dict(Counter(c['reason'] or 'SELECTED' for c in detail['candidates']))
    existing.write(folder/'accuracy.json',result)
    existing.write_report(folder,result,'Held-voice Transformer accuracy and candidate trace')
    complete_report(folder)
    return result


def choose_policy():
    selection=existing.read(ROOT/'locked_selection.json');arm=selection['arm']
    rows=existing.read(ROOT/'train_pairs.json');decl=existing.read(ROOT/'declaration.json')
    pairs=training.prepared(rows);cache={}
    for fold,groups in enumerate(decl['folds'],1):
        subset=[p for p in pairs if p.get('group',p['source']['singer_id']) in groups]
        folder=ROOT/f'cross_validation/fold_{fold}/{arm}/validation'
        with np.load(folder/'predictions.npz') as arrays:
            for index,p in enumerate(subset):
                if 'excluded' in p:continue
                for view in ('A','C'):cache[p['path_'+view]]=arrays[f'{index}_{view}'].copy()
    original=training.predictions
    training.predictions=lambda model,p,view:cache[p['path_'+view]]
    candidates=[]
    try:
        for activation in (8.,12.,16.):
            for gap in (0,2,4):
                policy=HeldPolicy(activation_cents=activation,gap_frames=gap,**decl.get('policy_spec',{}))
                folder=ROOT/f'policy_selection/activation_{activation:g}_gap_{gap}'
                folder.mkdir(parents=True,exist_ok=True)
                existing.write(folder/'evaluation_declaration.json',dict(pairs_manifest=str(ROOT/'train_pairs.json'),role='training-side out-of-fold policy selection, not a final test'))
                result=existing.read(folder/'accuracy.json') if (folder/'accuracy.json').exists() else training.evaluate(None,dict(seed=339),pairs,folder,policy,False)
                originals=all(r.get('permitted_original_abs_cents',0.)==0. for r in result['rows'])
                score=result['summary']['permitted_mae_cents']
                candidates.append(dict(policy=asdict(policy),originals_unchanged=originals,permitted_mae_cents=score,report=str(folder)))
                print('POLICY',activation,gap,'originals',originals,'permitted_mae',score,flush=True)
    finally:training.predictions=original
    # Preservation first, then permitted prediction error. No outcome-driven
    # change to support duration, correction magnitude or audit tolerance.
    selected=min(candidates,key=lambda r:(not r['originals_unchanged'],r['permitted_mae_cents'],r['policy']['gap_frames'],-r['policy']['activation_cents']))
    existing.write(ROOT/'policy_candidates.json',dict(selected=selected,candidates=candidates,selection_data='training-side out-of-fold predictions only'))
    if not all((ROOT/f'final/seed_{seed}/fit_receipt.json').exists() for seed in (339,340,341)):
        print('POLICY_CANDIDATES_READY; checkpoint lock waits for all three complete refits',flush=True)
        return
    existing.write(ROOT/'locked_policy.json',dict(selected=selected,candidates=candidates,selection_data='training-side out-of-fold predictions only',default_seed=339,
        checkpoints={str(seed):existing.sha(ROOT/f'final/seed_{seed}/checkpoint.pt') for seed in (339,340,341)},locked_at=datetime.now(timezone.utc).isoformat()))
    print('POLICY_LOCKED',selected,flush=True)


def original_controls(model,sources,policy,folder):
    controls=[];cache={}
    for index,source in enumerate(sources):
        if 'working_audio' not in source:
            controls.append(dict(group=source['source']['singer_id'],unassessed=source.get('excluded','No working audio')));continue
        path=Path(source['working_audio'])
        audio,analysis=recovery.from_cached(str(path),path.parent/'original_features.npz',cache)
        pred=np.stack([training.predict_raw(model,raw)[0] for raw in analysis['views']])
        detail=proposal(audio,analysis['features']['times_seconds'],analysis['views'],pred,policy)
        output,transaction=existing.rendered_output(dict(A=analysis,audio_A=audio),'A',detail['curve'],detail['plan'],folder/f'original_{index:02d}')
        reliable=analysis['reliable']
        controls.append(dict(group=source['source']['singer_id'],source_sha256=source['source']['sha256'],raw_original_abs_cents=float(abs(pred[0,reliable]).mean()) if reliable.any() else None,
            delivered_identity=bool(np.array_equal(audio,output)),outcome=transaction['outcome'],transaction=transaction))
    existing.write(folder/'original_controls.json',controls)
    return controls


def gates(result,controls):
    all_rows=result['rows'];groups=sorted({r['group'] for r in all_rows});by_group={}
    for group in groups:
        attempted=[r for r in all_rows if r['group']==group]
        valid=[r for r in attempted if 'excluded' not in r]
        originals=[r for r in controls if r['group']==group]
        requests=[r['raw_original_abs_cents'] for r in originals if r.get('raw_original_abs_cents') is not None]
        before=np.mean([r['delivered_before_mae_cents'] for r in valid]) if valid else None
        after=np.mean([r['delivered_after_mae_cents'] for r in valid]) if valid else None
        improved=sum(r['delivered_verified_improvement'] for r in valid)
        change=1-after/before if before else None
        preserved=bool(originals) and all(r.get('delivered_identity',False) for r in originals)
        raw_preserved=bool(requests) and np.mean(requests)<=3.
        passed=bool(preserved and raw_preserved and valid and improved/len(attempted)>=.25 and change>=.25)
        by_group[group]=dict(attempted=len(attempted),scoreable=len(valid),improved=int(improved),all_attempt_coverage=improved/len(attempted),delivered_error_reduction=change,
            original_controls=len(originals),originals_identity=preserved,raw_original_mean_cents=float(np.mean(requests)) if requests else None,passed=passed)
    directions={int(np.sign(r['cents'])) for r in all_rows if r.get('delivered_verified_improvement')}
    return dict(passed=bool(by_group) and all(r['passed'] for r in by_group.values()) and directions=={-1,1},by_singer=by_group,improved_directions=sorted(directions),independent_listening='not assessed',research_validation_passed=False)


def evaluate_partition(partition):
    lock=existing.read(ROOT/'locked_policy.json');policy=HeldPolicy(**lock['selected']['policy'])
    if partition=='final_reserved':
        gate=existing.read(ROOT/'development/gates.json')
        if not gate['passed']:raise ValueError('Reserved final singers remain locked: development usability gates failed')
    folder=ROOT/('final_evaluation' if partition=='final_reserved' else 'development')
    folder.mkdir(exist_ok=True)
    for seed,expected in lock['checkpoints'].items():assert existing.sha(ROOT/f'final/seed_{seed}/checkpoint.pt')==expected
    checkpoint=ROOT/'final/seed_339/checkpoint.pt';model,payload=load(checkpoint)
    dataset=WORKSPACE/'datasets/vocalset_expansion_2026-10-28'
    amendment_path=ROOT/'evaluation_partition_amendment.json'
    if amendment_path.exists():
        amendment=existing.read(amendment_path)
        singers=set(amendment['final_singers' if partition=='final_reserved' else 'development_singers'])
        pool=existing.read(dataset/'manifest.json')['records']+existing.read(WORKSPACE/'datasets/vocalset_pilot/manifest.json')['records']
        supplement=ROOT/'evaluation_vowel_supplement.json'
        vowels=('a','e')
        if supplement.exists():
            supplied=existing.read(supplement);extra=Path(supplied['manifest'])
            assert existing.sha(extra)==supplied['manifest_sha256']
            pool+=existing.read(extra)['records'];vowels=tuple(supplied['vowels'])
        rows=[r for r in pool if r['singer_id'] in singers and r['vowel'] in vowels]
    else:rows=[r for r in existing.read(dataset/'manifest.json')['records'] if r['split']==partition]
    assert_independent(payload['training_groups'],payload['training_sources'],[dict(source=r) for r in rows])
    # Freeze evaluation's source/model/policy identities before any target creation.
    declaration=folder/'evaluation_declaration.json'
    family=existing.read(ROOT/'declaration.json')
    dependency_files=[Path(__file__),WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py',WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py',WORKSPACE/'source/bpdr_input/experiments/learning/held_voice_training.py']+[Path(p) for p in family.get('additional_source_paths',[])]
    amendment=ROOT/'gate_protocol_amendment.json'
    if amendment.exists():
        gate_contract=existing.read(amendment)
        assert existing.sha(gate_contract['module'])==gate_contract['module_sha256']
        dependency_files.append(Path(gate_contract['module']))
    if not declaration.exists():existing.write(declaration,dict(created_at=datetime.now(timezone.utc).isoformat(),partition=partition,pairs_manifest=str(folder/'pairs.json'),checkpoint_sha256=existing.sha(checkpoint),policy_lock_sha256=existing.sha(ROOT/'locked_policy.json'),sources=rows,
        source_hashes={str(p):existing.sha(p) for p in dependency_files}))
    source_records=[];pairs=[]
    for i,row in enumerate(rows,1):
        record=folder/f'source_{i:02d}.json'
        if record.exists():prepared=existing.read(record)
        else:
            if 'long_v2' in ROOT.name or family.get('intervention_protocol') in ('long_praat_v2','step_praat_v6'):
                from math import gcd
                import soundfile as sf
                from scipy.signal import resample_poly
                sys.path.insert(0,str(WORKSPACE/'tools'))
                import prepare_long_held_voice_pairs as long_pairs
                original=Path(row['path']);assert existing.sha(original)==row['sha256']
                wave,sr=sf.read(original,dtype='float32',always_2d=True)
                channel=int(np.argmax(np.mean(wave.astype(float)**2,axis=0)));wave=wave[:round(20*sr),channel]
                divisor=gcd(sr,22050)
                if sr!=22050:wave=resample_poly(wave,22050//divisor,sr//divisor).astype(np.float32)
                baseline_folder=folder/'working_originals'/f'{i:02d}';baseline_folder.mkdir(parents=True,exist_ok=True)
                baseline=baseline_folder/'original_working.wav';sf.write(baseline,wave,22050,subtype='PCM_24');wave,_=sf.read(baseline,dtype='float32')
                analysis=recovery.measured(wave);np.savez_compressed(baseline_folder/'original_features.npz',**analysis['features'])
                source=dict(source=row,working_audio=str(baseline),working_sha256=existing.sha(baseline),crop_offset_seconds=0.,source_channel=channel,duration_seconds=len(wave)/22050)
                if family.get('intervention_protocol')=='step_praat_v6':
                    import prepare_recoverable_step_pairs as step_pairs
                    spec=dict(existing.read(ROOT/'data_declaration.json')['spec']);spec['root']=str(folder)
                    prepared=step_pairs.worker((i,source,spec))
                else:
                    long_pairs.ROOT=folder
                    prepared=long_pairs.worker((i,source))
                source,pp=prepared['source'],prepared['pairs']
            else:source,pp=recovery.prepare_source(row,folder,i)
            prepared=dict(source=source,pairs=[recovery.serialise_pair(p) for p in pp]);existing.write(record,prepared)
        source_records.append(prepared['source']);pairs.extend(prepared['pairs'])
        print('EVALUATION_SOURCE',partition,i,'/',len(rows),flush=True)
    existing.write(folder/'pairs.json',pairs)
    prepared_pairs=training.prepared(pairs)
    result=training.evaluate(model,payload,prepared_pairs,folder,policy,True)
    complete_trace(result,prepared_pairs,folder,policy)
    controls=original_controls(model,source_records,policy,folder)
    gate=gates(result,controls);existing.write(folder/'gates.json',gate)
    existing.write(folder/'evidence.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=existing.sha(checkpoint),policy=asdict(policy),accuracy_sha256=existing.sha(folder/'accuracy.json'),controls_sha256=existing.sha(folder/'original_controls.json'),gates=gate,partition=partition))
    for p,h in existing.read(declaration)['source_hashes'].items():assert existing.sha(p)==h
    print('GATES',partition,gate,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['select-policy','development','final'])
    parser.add_argument('--run-root',type=Path,default=WORKSPACE/'results/transformer_usability/2026-10-28_v4')
    args=parser.parse_args();torch.set_num_threads(2)
    ROOT=args.run_root.resolve()
    if args.stage=='select-policy':choose_policy()
    else:evaluate_partition('final_reserved' if args.stage=='final' else 'development')
