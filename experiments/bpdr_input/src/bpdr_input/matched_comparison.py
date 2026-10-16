"""Matched exploratory fits using existing model/loss code and frozen input catalogs."""
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import torch

from .examples import Normalizer, collate, read_pair
from .losses import LossWeights, objective
from .model import CorrectionEncoder, ModelConfig
from .shared_exports import digest
from .shared_pair_audit import game_boundary_track
from .interventions import nominal_boundaries


HIERARCHY=('dataset','group_id','source_id','phrase_id','cents','corruption')
STATUS='exploratory_not_research'


def macro(rows, metric, keys=HIERARCHY):
    """Equal children at every level; report missing datasets separately, never as zero."""
    rows=[r for r in rows if r.get(metric) is not None]
    if not rows:return None
    if not keys:return float(np.mean([r[metric] for r in rows]))
    groups=defaultdict(list)
    for r in rows:groups[r[keys[0]]].append(r)
    return float(np.mean([macro(group,metric,keys[1:]) for group in groups.values()]))


def metric_row(record,predictions,inputs,mask,detuning):
    common=np.asarray(mask,dtype=bool);plateau=common&(detuning!=0);outside=common&(detuning==0)
    if not common.any() or not plateau.any() or not outside.any():raise ValueError('Incomplete development metric masks')
    if any(np.asarray(predictions[v]).shape!=common.shape or not np.isfinite(predictions[v]).all() for v in 'ABCD'):
        raise ValueError('Non-finite or misaligned predictions')
    r={k:record[k] for k in HIERARCHY}
    r.update(group_kind=record['group_kind'],reference_type=record['reference_type'])
    p=predictions;target=-detuning
    joint=(inputs['A'][:,66]>.5)&(inputs['C'][:,66]>.5)
    mean=lambda x,m:float(np.abs(x[m]).mean()) if m.any() else None
    r.update(correction_mae_cents=float(np.mean([mean(p[v]-(0 if v in 'AB' else target),common) for v in 'ABCD'])),
        repair_error_cents=float(np.mean([mean(p[v]-target,plateau) for v in 'CD'])),
        original_audited_alteration_cents=float(np.mean([mean(p[v],common) for v in 'AB'])),
        outside_alteration_cents=float(np.mean([mean(p[v],outside) for v in 'CD'])),
        boundary_sensitivity_cents=float(np.mean([mean(p['A']-p['B'],common),mean(p['C']-p['D'],common)])),
        counter_detuning_error_cents=float(np.mean([mean(p['C']-p['A']+detuning,common),mean(p['D']-p['B']+detuning,common)])),
        original_joint_voiced_no_change_cents=None if not joint.any() else float(np.mean([mean(p[v],joint) for v in 'AB'])),
        common_frames=int(common.sum()),plateau_frames=int(plateau.sum()),joint_voiced_no_change_frames=int(joint.sum()))
    return r


METRICS=('correction_mae_cents','repair_error_cents','original_audited_alteration_cents','outside_alteration_cents',
         'boundary_sensitivity_cents','counter_detuning_error_cents','original_joint_voiced_no_change_cents')


def aggregate(rows,expected_datasets=('vocadito','jamendo')):
    present=sorted({r['dataset'] for r in rows})
    missing=sorted(set(expected_datasets)-set(present))
    result=dict(pairs=len(rows),available_datasets=present,missing_datasets=missing,
        available_dataset_macro={m:macro(rows,m) for m in METRICS},
        full_two_dataset_macro=None if missing else {m:macro(rows,m) for m in METRICS},
        by_dataset={ds:dict(pairs=sum(r['dataset']==ds for r in rows),
            group_count=len({r['group_id'] for r in rows if r['dataset']==ds}),
            metrics={m:macro([r for r in rows if r['dataset']==ds],m,HIERARCHY[1:]) for m in METRICS}) for ds in present},
        by_group={g:dict(dataset=next(r['dataset'] for r in rows if r['group_id']==g),
            group_kind=next(r['group_kind'] for r in rows if r['group_id']==g),
            pairs=sum(r['group_id']==g for r in rows),
            metrics={m:macro([r for r in rows if r['group_id']==g],m,HIERARCHY[2:]) for m in METRICS})
            for g in sorted({r['group_id'] for r in rows})})
    result['selection_criterion']=result['available_dataset_macro']['correction_mae_cents']
    result['selection_scope']='Available prepared development datasets only; missing datasets unavailable, never imputed'
    return result


def matched_model(protocol,width):
    torch.manual_seed(protocol['fit']['seed'])
    common=CorrectionEncoder(ModelConfig(features=71,**protocol['model']))
    state={k:v.detach().clone() for k,v in common.state_dict().items()}
    rng=torch.get_rng_state()
    model=common if width==71 else CorrectionEncoder(ModelConfig(features=width,**protocol['model']))
    if width==70:state['project.weight']=state['project.weight'][:,:70]
    model.load_state_dict(state)
    torch.set_rng_state(rng)
    return model


def check_catalog(records,split):
    if not records:raise ValueError('No usable '+split+' pairs; checkpoint selection unavailable')
    if any(r['split']!=split or not r['audit_passed'] or not r['exploratory'] or r['training_eligible'] is not False
           or r['data_status']!=STATUS for r in records):
        raise ValueError('Wrong partition or research claim in '+split+' catalog')
    keys=[tuple(r[k] for k in HIERARCHY) for r in records]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate catalog identity')


def evaluate_no_change(model,records,normalizer,width,directory=None):
    """Unpaired originals: voiced-frame alteration without an injected-target mask."""
    model.eval()
    rows=[];saved={}
    with torch.inference_mode():
        for i,r in enumerate(records):
            with np.load(r['input_path']) as z:raw=z['boundary_features_71']
            voiced=raw[:,66]>.5
            device=next(model.parameters(),torch.empty(0)).device
            value=model(torch.from_numpy(normalizer(raw[:,:width])[None,:,:]).to(device)).squeeze(0).cpu().numpy().copy()
            if not np.isfinite(value).all():raise ValueError('Invalid unpaired original prediction')
            row={k:r[k] for k in HIERARCHY}
            row.update(reference_type=r['reference_type'],group_kind=r['group_kind'],voiced_frames=int(voiced.sum()),
                original_all_voiced_alteration_cents=float(abs(value[voiced]).mean()) if voiced.any() else None,
                original_all_voiced_edit_fraction=float((abs(value[voiced])>=5).mean()) if voiced.any() else None)
            rows.append(row);saved[f'original_{i:04d}']=value
    summary=dict(originals=len(rows),masks='Original confidence >= .8 voiced frames; independent of detuning audit masks',
        musical_approval=False,
        by_dataset={ds:dict(originals=sum(r['dataset']==ds for r in rows),
            voiced_originals=sum(r['dataset']==ds and r['voiced_frames']>0 for r in rows),
            mean_alteration_cents=macro([r for r in rows if r['dataset']==ds],'original_all_voiced_alteration_cents'),
            edit_fraction=macro([r for r in rows if r['dataset']==ds],'original_all_voiced_edit_fraction'))
            for ds in sorted({r['dataset'] for r in rows})})
    if directory is not None:
        np.savez_compressed(Path(directory)/'no_change_predictions.npz',**saved)
        (Path(directory)/'no_change_metrics.json').write_text(json.dumps(dict(summary=summary,originals=rows),indent=2)+'\n',encoding='utf-8')
    return summary


def evaluate_catalog(model,records,normalizer,width,directory=None,no_change_records=()):
    """Full context, one unique input at a time; cache predictions, never a corpus tensor."""
    model.eval();cache={};rows=[];saved={}
    with torch.inference_mode():
        for index,r in enumerate(records):
            pair=read_pair(r,'engineering');pred={}
            for view in 'ABCD':
                raw=pair['inputs'][view][:,:width]
                key=hashlib.sha256(raw.tobytes()).hexdigest()
                if key not in cache:
                    x=torch.from_numpy(normalizer(raw)[None,:,:]).to(next(model.parameters()).device)
                    value=model(x).squeeze(0).cpu().numpy().copy()
                    if not np.isfinite(value).all():raise ValueError('Non-finite development prediction')
                    cache[key]=value
                pred[view]=cache[key]
                if directory is not None:saved[f'p{index:04d}_{view}']=pred[view]
            rows.append(metric_row(r,pred,pair['inputs'],pair['mask'],pair['detuning']))
    summary=aggregate(rows)
    if directory is not None:
        directory=Path(directory);directory.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(directory/'predictions.npz',**saved)
    if no_change_records:
        summary['unpaired_original_preservation']=evaluate_no_change(model,no_change_records,normalizer,width,directory)
    if directory is not None:
        (directory/'metrics.json').write_text(json.dumps(dict(data_status=STATUS,summary=summary,pairs=rows,
            unique_forward_inputs=len(cache),prediction_key='p{catalog_index:04d}_{view}',
            selection_partition='development'),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return summary


def prepare_no_change_inputs(dev,destination):
    """New diagnostic views from technically processed originals, including support refusals."""
    e=json.loads((dev/'evidence.json').read_text());d=json.loads((dev/'declaration.json').read_text())
    selected={(r['dataset'],r['phrase_id']):r for r in d['selected_sources']}
    result=[]
    for s in e['sources']:
        baseline=dev/s['dataset']/s['phrase_id']/'baseline_acoustics.npz'
        if not baseline.exists():continue
        if digest(baseline)!=e['artifacts'][str(baseline.relative_to(dev))]:raise ValueError('Original baseline changed')
        with np.load(baseline) as z:times=z['times_seconds'];frame=z['frame_features_70']
        row=selected[s['dataset'],s['phrase_id']]
        if s['dataset']=='vocadito':
            notes=json.loads(Path(row['notes_path']).read_text());edges=nominal_boundaries(times,notes)
        else:
            bounds=json.loads((Path(row['record_path']).parent/'boundaries.json').read_text())
            edges=sorted({float(n[k]) for n in bounds['notes'] for k in ('onset_sec','offset_sec')})
        folder=destination/'no_change_inputs'/s['dataset'];folder.mkdir(parents=True,exist_ok=True)
        path=folder/(s['phrase_id']+'_A_input.npz')
        np.savez_compressed(path,times_seconds=times,frame_features_70=frame,
            boundary_features_71=np.column_stack([frame,game_boundary_track(times,edges)]).astype(np.float32))
        result.append(dict(dataset=s['dataset'],group_id=s['group_id'],group_kind=s['group_kind'],
            source_id=s['source_id'],phrase_id=s['phrase_id'],cents=0.,corruption='nominal',
            reference_type=s['reference_type'],input_path=str(path),input_sha256=digest(path),
            paired_source_status=s['status'],split='development',musical_approval=False))
    return result


def run(device='cpu',microbatch_size=1):
    if device not in ('cpu','xpu') or (device=='xpu' and not torch.xpu.is_available()):
        raise ValueError('Requested fitting device is unavailable')
    root=Path(__file__).resolve().parents[2]
    prep=Path(json.loads((root/'runs/comparison_preparation_current.json').read_text())['path'])
    dev=Path(json.loads((root/'runs/development_preparation_current.json').read_text())['path'])
    train_records=json.loads((prep/'summary.json').read_text())['examples']
    dev_records=json.loads((dev/'summary.json').read_text())['examples']
    verification=json.loads((dev/'independent_verification.json').read_text())
    if (verification['status']!='verified' or verification['summary_sha256']!=digest(dev/'summary.json')
            or verification['evidence_sha256']!=digest(dev/'evidence.json')):
        raise ValueError('Development verification is missing or stale')
    check_catalog(train_records,'train');check_catalog(dev_records,'development')
    if {r['group_id'] for r in train_records}&{r['group_id'] for r in dev_records}:raise ValueError('Training/development group overlap')
    if {(r['dataset'],r['source_id']) for r in train_records}&{(r['dataset'],r['source_id']) for r in dev_records}:
        raise ValueError('Training/development source overlap')
    protocol=json.loads((prep/'locked_protocol.json').read_text())
    with np.load(prep/'normalizer.npz') as z:normalizer=Normalizer(z['mean'],z['scale'])
    with np.load(prep/'sampling_schedule.npz') as z:schedule=z['indices']
    fit=protocol['fit']
    if microbatch_size not in (1,2,4,8) or microbatch_size>fit['batch_size']:
        raise ValueError('Invalid physical microbatch size')
    if schedule.shape!=(fit['updates'],fit['batch_size']) or schedule.min()<0 or schedule.max()>=len(train_records):
        raise ValueError('Saved schedule does not match locked budget')
    frozen={str(p):digest(p) for p in [prep/'summary.json',prep/'locked_protocol.json',prep/'normalizer.npz',
        prep/'sampling_schedule.npz',dev/'summary.json',dev/'evidence.json',dev/'independent_verification.json']}
    previous=json.loads((dev/'declaration.json').read_text())
    frozen.update(previous['frozen_inputs'])
    for r in [*train_records,*dev_records]:
        for view in 'ABCD':frozen[str(Path(r['input_directory'])/(view+'_input.npz'))]=r['input_sha256'][view]
        frozen[r['target_path']]=r['targets_sha256']
        frozen[r['audit_path']]=r['audit_sha256']
    destination=root/'runs/matched_comparison'/uuid.uuid4().hex;destination.mkdir(parents=True)
    copied=destination/'source_code_at_run';copied.mkdir()
    for name in ['matched_comparison','model','losses','examples','development_preparation']:
        p=Path(__file__).with_name(name+'.py');frozen[str(p)]=digest(p);shutil.copy2(p,copied/p.name)
    def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for p,h in {**frozen,**previous['pipeline_files']}.items():
        if digest(p)!=h:raise ValueError('Changed frozen comparison input: '+p)
    no_change_records=prepare_no_change_inputs(dev,destination)
    write(destination/'no_change_catalog.json',dict(records=no_change_records,
        purpose='Supplemental original-input alteration only; not checkpoint selection or musical acceptance',
        mask='All original confidence >= .8 voiced frames; no detuning mask required',
        excluded='Sources lacking technical baseline acoustics; recorded source refusals remain unchanged'))
    write(destination/'declaration.json',dict(protocol_path=str(prep/'locked_protocol.json'),protocol_sha256=digest(prep/'locked_protocol.json'),
        frozen_inputs=frozen,pipeline_files=previous['pipeline_files'],training_catalog=str(prep/'summary.json'),
        development_catalog=str(dev/'summary.json'),conditions=list(protocol['conditions']),
        effective_batch_size=fit['batch_size'],microbatch_size=microbatch_size,gradient_accumulation=fit['batch_size']//microbatch_size,
        execution='Float32, two CPU threads, full phrase per microbatch; same device/effective samples/updates/initial state for all conditions',
        device=device,device_name=torch.xpu.get_device_name(0) if device=='xpu' else 'CPU',
        numerical_verification=dict(reloaded_predictions_atol_cents=1e-5,reloaded_predictions_rtol=1e-5,
            shared_warmup_parameters_atol=1e-5,shared_warmup_parameters_rtol=1e-5,
            exact_equality_recorded_separately=True),
        checkpoint_selection='Locked hierarchical criterion on available development datasets; missing datasets reported unavailable',
        available_development_datasets=sorted({r['dataset'] for r in dev_records}),
        additional_no_change_originals=len(no_change_records),no_change_catalog_sha256=digest(destination/'no_change_catalog.json'),
        no_human_reference_claim_for_predicted_only_development=True,no_hyperparameter_search=True,
        no_audio_rendering=True,no_shared_pipeline_execution=True,no_pp1_or_final_holdout_access=True,
        research_ready_examples=0,data_status=STATUS,seed=fit['seed'],torch_version=str(torch.__version__)))
    shutil.copy2(prep/'locked_protocol.json',destination/'locked_protocol.json')
    evidence=dict(status='running',data_status=STATUS,conditions={},research_ready_examples=0,
        declaration_sha256=digest(destination/'declaration.json'),development_example_count=len(dev_records),training_example_count=len(train_records))
    write(destination/'evidence.json',evidence)
    print('MATCHED_RUN:',destination,flush=True)
    # A zero-correction baseline makes inability to recover injected shifts visible.
    zero_rows=[]
    for r in dev_records:
        pair=read_pair(r,'engineering');pred={v:np.zeros(len(pair['mask']),dtype=np.float32) for v in 'ABCD'}
        zero_rows.append(metric_row(r,pred,pair['inputs'],pair['mask'],pair['detuning']))
    write(destination/'zero_correction_baseline.json',dict(data_status=STATUS,summary=aggregate(zero_rows),pairs=zero_rows))
    torch.set_num_threads(2)
    weights=LossWeights(**protocol['loss_weights'])
    for condition,width in protocol['conditions'].items():
        started=time.monotonic();folder=destination/condition;folder.mkdir()
        model=matched_model(protocol,width)
        initial=folder/'initial_state.pt';torch.save(model.state_dict(),initial)
        model=model.to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=fit['learning_rate'])
        history=[];candidates=[];best=None
        @lru_cache(maxsize=16)
        def pair_at(index):return read_pair(train_records[index],'engineering')
        for step,indices in enumerate(schedule,1):
            model.train();optimizer.zero_grad();parts_total=defaultdict(float);loss_total=0.
            for start in range(0,len(indices),microbatch_size):
                chosen=indices[start:start+microbatch_size]
                batch=collate([pair_at(int(index)) for index in chosen],normalizer,width)
                batch={k:({v:t.to(device) for v,t in value.items()} if isinstance(value,dict) else
                    value.to(device) if torch.is_tensor(value) else value) for k,value in batch.items()}
                pred={view:model(value,batch['padding']) for view,value in batch['inputs'].items()}
                loss,parts,skipped=objective(pred,batch,condition,weights,response=step>fit['warmup'])
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                proportion=len(chosen)/fit['batch_size']
                (loss*proportion).backward()
                loss_total+=float(loss.detach())*proportion
                for k,v in parts.items():parts_total[k]+=float(v.detach())*proportion
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),fit['gradient_clip_norm']))
            if not np.isfinite(norm):raise ValueError('Non-finite gradient norm')
            optimizer.step()
            history.append(dict(update=step,loss=loss_total,gradient_norm_before_clip=norm,**parts_total))
            if step%10==0:print('FIT:',condition,step,'/',fit['updates'],'loss',round(loss_total,5),'elapsed_s',round(time.monotonic()-started),flush=True)
            if step%fit['checkpoint_every']==0:
                checkpoint=folder/f'update_{step:04d}';checkpoint.mkdir()
                torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer_state=optimizer.state_dict(),rng_state=torch.get_rng_state(),
                    xpu_rng_state=torch.xpu.get_rng_state() if device=='xpu' else None,device=device,
                    config=asdict(model.config),normalizer=normalizer.state(),condition=condition,update=step,
                    data_status=STATUS,training_catalog_sha256=digest(prep/'summary.json'),
                    development_catalog_sha256=digest(dev/'summary.json'),protocol_sha256=digest(prep/'locked_protocol.json')),
                    checkpoint/'checkpoint.pt')
                summary=evaluate_catalog(model,dev_records,normalizer,width,checkpoint,no_change_records)
                value=summary['selection_criterion']
                candidate=dict(update=step,criterion=value,checkpoint_path=str(checkpoint/'checkpoint.pt'),
                    checkpoint_sha256=digest(checkpoint/'checkpoint.pt'),metrics_sha256=digest(checkpoint/'metrics.json'),
                    predictions_sha256=digest(checkpoint/'predictions.npz'))
                candidates.append(candidate)
                if value is not None and (best is None or value<best['criterion']):best=candidate
                write(folder/'history.json',dict(data_status=STATUS,steps=history))
                write(folder/'selection.json',dict(data_status=STATUS,partition='development',candidates=candidates,selected=best,
                    available_datasets=summary['available_datasets'],missing_datasets=summary['missing_datasets']))
                evidence['conditions'][condition]=dict(status='running',completed_updates=step,selected=best)
                write(destination/'evidence.json',evidence)
                print('DEV:',condition,step,'criterion',value,'datasets',summary['available_datasets'],flush=True)
        pair_at.cache_clear()
        if best is None:raise ValueError('Checkpoint selection unavailable')
        saved=torch.load(best['checkpoint_path'],weights_only=True,map_location='cpu')
        reloaded=CorrectionEncoder(ModelConfig(**saved['config']));reloaded.load_state_dict(saved['state']);reloaded.to(device)
        reloaded.eval()
        verify=evaluate_catalog(reloaded,dev_records,normalizer,width,no_change_records=no_change_records)
        original=json.loads((Path(best['checkpoint_path']).parent/'metrics.json').read_text())['summary']
        def close(a,b):
            if isinstance(a,dict):return set(a)==set(b) and all(close(a[k],b[k]) for k in a)
            if isinstance(a,(float,int)) and not isinstance(a,bool):return bool(np.isclose(a,b,atol=1e-5,rtol=1e-5))
            return a==b
        if not close(verify,original):raise ValueError('Selected checkpoint reload changed development metrics beyond declared numerical tolerance')
        evidence['conditions'][condition].update(status='complete',elapsed_seconds=time.monotonic()-started,
            initial_state_sha256=digest(initial),selected_reload_exact=verify==original,
            selected_reload_within_declared_tolerance=True,model=model.describe())
        write(destination/'evidence.json',evidence)
    for p,h in {**frozen,**previous['pipeline_files']}.items():
        if digest(p)!=h:raise ValueError('Frozen comparison input changed: '+p)
    evidence.update(status='complete',total_optimizer_updates=len(protocol['conditions'])*fit['updates'],
        artifacts={str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file() and p.name!='evidence.json'})
    write(destination/'evidence.json',evidence)
    (root/'runs/matched_comparison_current.json').write_text(json.dumps(dict(path=str(destination)),indent=2)+'\n',encoding='utf-8')
    print('MATCHED_COMPLETE:',destination,flush=True)
    return destination


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','xpu'],default='cpu')
    parser.add_argument('--microbatch-size',type=int,choices=[1,2,4,8],default=1)
    args=parser.parse_args();run(args.device,args.microbatch_size)
