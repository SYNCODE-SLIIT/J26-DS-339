"""Execute the separately prepared, fixed-budget expanded exploratory comparison.

Historical runners, catalogs, checkpoints and active preparation pointers are never
rewritten. No audio processing or musical/research acceptance occurs here.
"""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
import torch

from .examples import Normalizer, collate, read_pair
from .expanded_experiment_preparation import preservation_qualified, select_checkpoint
from .frozen_development_evaluation import evaluation_inventory, response_row, summarize
from .losses import LossWeights, objective
from .matched_comparison import HIERARCHY, STATUS, check_catalog, matched_model
from .model import CorrectionEncoder, ModelConfig
from .shared_exports import digest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def validate_protocol(p, schedule, training_count):
    """Refuse budget, architecture, ordering or execution drift before any fit."""
    expected_fit = dict(seed=339, updates=300, warmup=100, batch_size=8,
                        learning_rate=.001, optimizer='AdamW', gradient_clip_norm=1.,
                        checkpoint_every=25, weight_decay=.01)
    if p['version'] != 'expanded-training-experiment-0.1' or p['fit'] != expected_fit:
        raise ValueError('Locked fit budget/settings changed')
    if list(p['conditions'].items()) != [('augmentation_only',71),('frame_only',70),('bpdr',71)]:
        raise ValueError('Locked condition order/width changed')
    if p['model'] != dict(hidden=128,layers=4,heads=4,feed_forward=512,dropout=.1):
        raise ValueError('Locked architecture changed')
    if p['loss_weights'] != dict(boundary=1.,detuning=1.,smoothness=.1):
        raise ValueError('Locked objective changed')
    if p['checkpoint_selection']['candidates'] != list(range(25,301,25)) or p['checkpoint_selection']['preservation_limit_cents'] != 1.:
        raise ValueError('Locked selection changed')
    x=p['execution']
    if tuple(x[k] for k in ['device','dtype','cpu_threads','microbatch_size','effective_batch_size','gradient_accumulation']) != ('xpu','float32',2,1,8,8):
        raise ValueError('Locked execution changed')
    if training_count != 995 or schedule.shape != (300,8) or schedule.dtype.kind not in 'iu' or schedule.min()<0 or schedule.max()>=995:
        raise ValueError('Saved schedule or training inventory changed')


def assess(model, records, originals, normalizer, width, directory=None):
    """Assess full phrases with cached deterministic forwards; retain all controls."""
    if width not in (70,71):
        raise ValueError('Invalid input width')
    if model is not None:
        model.eval()
        device=next(model.parameters()).device
    cache={}; saved={}; rows=[]; original_rows=[]

    def forward(raw):
        raw=np.ascontiguousarray(raw[:,:width],dtype=np.float32)
        if raw.ndim!=2 or not np.isfinite(raw).all():
            raise ValueError('Non-finite or malformed development input')
        key=(raw.shape,hashlib.sha256(raw.tobytes()).hexdigest())
        if key not in cache:
            if model is None:
                value=np.zeros(len(raw),dtype=np.float32)
            else:
                with torch.inference_mode():
                    value=model(torch.from_numpy(normalizer(raw)[None]).to(device)).squeeze(0).cpu().numpy().copy()
            if value.shape!=(len(raw),) or not np.isfinite(value).all():
                raise ValueError('Invalid development prediction')
            cache[key]=value
        return cache[key]

    for i,r in enumerate(records):
        pair=read_pair(r,'engineering')
        pred={v:forward(pair['inputs'][v]) for v in 'ABCD'}
        saved.update({f'p{i:04d}_{v}':pred[v] for v in 'ABCD'})
        rows.append(response_row(r,pred,pair))
    for i,r in enumerate(originals):
        with np.load(r['input_path']) as z:
            raw=z['boundary_features_71'].astype(np.float32)
        voiced=raw[:,66]>.5
        value=forward(raw)
        row={k:r[k] for k in HIERARCHY}
        row.update(reference_type=r['reference_type'],group_kind=r['group_kind'],voiced_frames=int(voiced.sum()),
                   original_all_voiced_alteration_cents=float(abs(value[voiced]).mean()) if voiced.any() else None,
                   original_all_voiced_edit_fraction=float((abs(value[voiced])>=5).mean()) if voiced.any() else None)
        original_rows.append(row);saved[f'original_{i:04d}']=value
    summary=summarize(rows,original_rows)
    summary['scope']='Locked exploratory development checkpoint assessment; no PP1/final-holdout access or model adoption'
    result=dict(data_status=STATUS,summary=summary,pairs=rows,originals=original_rows,
                unique_forward_inputs=0 if model is None else len(cache),prediction_vectors=len(saved))
    if directory is not None:
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/'predictions.npz',**saved)
        write(directory/'metrics.json',result)
    return result


def to_device(batch, device):
    return {k:({v:t.to(device) for v,t in value.items()} if isinstance(value,dict) else
               value.to(device) if torch.is_tensor(value) else value) for k,value in batch.items()}


def accumulated_step(model, optimizer, pairs, normalizer, width, condition, weights, response, device):
    """Eight whole-phrase microbatches, equal example weight, one optimizer step."""
    if len(pairs)!=8:
        raise ValueError('Exactly eight examples per declared optimizer update')
    model.train();optimizer.zero_grad();parts_total=defaultdict(float);loss_total=0.;skipped_total=defaultdict(int)
    for pair in pairs:
        batch=to_device(collate([pair],normalizer,width),device)
        pred={v:model(value,batch['padding']) for v,value in batch['inputs'].items()}
        loss,parts,skipped=objective(pred,batch,condition,weights,response=response)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        (loss/8).backward()
        loss_total+=float(loss.detach())/8
        for k,value in parts.items():parts_total[k]+=float(value.detach())/8
        for k,value in skipped.items():skipped_total[k]+=value
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    if not np.isfinite(norm):
        raise ValueError('Non-finite gradient norm')
    optimizer.step()
    return dict(loss=loss_total,gradient_norm_before_clip=norm,skipped=skipped_total,**parts_total)


def same_vectors(first, second):
    with np.load(first) as a,np.load(second) as b:
        if set(a.files)!=set(b.files):
            raise ValueError('Reload vector inventory changed')
        exact=True;maximum=0.
        for key in a.files:
            if a[key].shape!=b[key].shape or not np.allclose(a[key],b[key],atol=1e-5,rtol=1e-5):
                raise ValueError('Reload prediction outside pre-existing numerical bounds: '+key)
            exact=exact and np.array_equal(a[key],b[key])
            maximum=max(maximum,float(abs(a[key]-b[key]).max()))
        return dict(vectors=len(a.files),exact=bool(exact),within_declared_tolerance=True,
                    maximum_absolute_difference_cents=maximum,atol_cents=1e-5,rtol=1e-5)


def authenticate(files):
    for path, expected in files.items():
        if digest(path)!=expected:
            raise ValueError('Frozen input changed: '+path)


def run(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[2]
    pointer=root/'runs/expanded_comparison_current.json'
    if pointer.exists():
        raise ValueError('A recorded expanded experiment already exists; do not silently repeat fitting')
    prep=Path(read(root/'runs/expanded_experiment_preparation_current.json')['path'])
    p=read(prep/'locked_protocol.json');pd=read(prep/'declaration.json');pe=read(prep/'evidence.json')
    proof=read(prep/'final_verification.json')
    if proof['status']!='verified' or proof['evidence_sha256']!=digest(prep/'evidence.json') or pe['status']!='complete':
        raise ValueError('Prepared experiment verification missing or stale')
    authenticate({str(prep/name):h for name,h in pe['artifacts'].items()})
    paths=p['artifacts'];train=read(paths['training_catalog'])['records']
    dev=read(paths['development_catalog'])['records'];originals=read(paths['original_controls'])['records']
    with np.load(paths['normalizer']) as z:normalizer=Normalizer(z['mean'],z['scale'])
    with np.load(paths['sampling_schedule']) as z:schedule=z['indices'].copy()
    validate_protocol(p,schedule,len(train));check_catalog(train,'train');evaluation_inventory(dev,originals)
    if {ds:sum(r['dataset']==ds for r in train) for ds in ['vocadito','jamendo']}!={'vocadito':77,'jamendo':918}:
        raise ValueError('Training evidence roles/counts changed')
    if {r['group_id'] for r in train}&{r['group_id'] for r in dev+originals} or {(r['dataset'],r['source_id']) for r in train}&{(r['dataset'],r['source_id']) for r in dev+originals}:
        raise ValueError('Training/development partition overlap')
    x=p['execution']
    if (Path(sys.executable).resolve()!=Path(x['interpreter']).resolve() or str(torch.__version__)!=x['torch_version']
            or not torch.xpu.is_available() or torch.xpu.get_device_name(0)!=x['device_name']):
        raise ValueError('Declared XPU interpreter, version or device unavailable; stopping')
    if normalizer.mean.shape!=(71,) or normalizer.scale.shape!=(71,) or not np.isfinite(normalizer.mean).all() or not np.isfinite(normalizer.scale).all() or (normalizer.scale<=0).any():
        raise ValueError('Invalid candidate normalizer')
    frozen=dict(pd['frozen_inputs'])
    for name in ['declaration.json','evidence.json','independent_verification.json','final_verification.json','locked_protocol.json']:
        frozen[str(prep/name)]=digest(prep/name)
    for path in paths.values():frozen[path]=digest(path)
    frozen[str(root/'runs/expanded_experiment_preparation_current.json')]=digest(root/'runs/expanded_experiment_preparation_current.json')
    for r in train+dev:
        for v in 'ABCD':frozen[str(Path(r['input_directory'])/(v+'_input.npz'))]=r['input_sha256'][v]
        frozen[r['target_path']]=r['targets_sha256'];frozen[r['audit_path']]=r['audit_sha256']
    for r in originals:frozen[r['input_path']]=r['input_sha256']
    for name in ['expanded_comparison','matched_comparison','frozen_development_evaluation','expanded_experiment_preparation','model','losses','examples']:
        path=Path(__file__).with_name(name+'.py');frozen[str(path)]=digest(path)
    runner=root/'runs/execute_expanded_comparison_2026-10-16.py';frozen[str(runner)]=digest(runner)
    authenticate({**frozen,**pd['pipeline_files']})
    out=root/'runs/expanded_comparison'/uuid.uuid4().hex;out.mkdir(parents=True)
    source=out/'source_code_at_run';source.mkdir()
    for path in frozen:
        if Path(path).parent==Path(__file__).parent or Path(path)==runner:shutil.copy2(path,source/Path(path).name)
    declaration=dict(declared_utc=datetime.now(timezone.utc).isoformat(),frozen_inputs=frozen,pipeline_files=pd['pipeline_files'],
        prepared_run=str(prep),protocol_path=str(prep/'locked_protocol.json'),protocol_sha256=digest(prep/'locked_protocol.json'),
        training_examples=995,development_examples=304,original_controls=88,planned_optimizer_updates=900,
        runtime=dict(interpreter=sys.executable,torch=str(torch.__version__),device=torch.xpu.get_device_name(0)),
        conditions=p['conditions'],execution=x,data_status=STATUS,research_ready_examples=0,model_adopted=False,
        numerical_verification=dict(reloaded_predictions_atol_cents=1e-5,reloaded_predictions_rtol=1e-5,
                                    shared_warmup_parameters_atol=1e-5,shared_warmup_parameters_rtol=1e-5,exact_equality_recorded_separately=True),
        no_audio_processing=True,no_shared_pipeline_execution=True,no_pp1_or_final_holdout_access=True,
        no_objective_architecture_or_budget_change=True,legacy_active_preparation_unchanged=True)
    write(out/'declaration.json',declaration);shutil.copy2(prep/'locked_protocol.json',out/'locked_protocol.json')
    e=dict(status='running',declaration_sha256=digest(out/'declaration.json'),conditions={},
           total_optimizer_updates=0,training_examples=995,development_examples=304,original_controls=88,
           data_status=STATUS,research_ready_examples=0,model_adopted=False,new_audio_files=0,new_extractions=0)
    write(out/'evidence.json',e);write(pointer,dict(path=str(out)))
    print('EXPANDED_RUN:',out,flush=True)
    torch.set_num_threads(2);weights=LossWeights(**p['loss_weights'])
    try:
        zero=assess(None,dev,originals,normalizer,71,out/'zero_correction')
        zero_error=zero['summary']['dataset_balanced_macro']['repair_error_cents']
        e['zero_correction_plateau_error_cents']=zero_error
        for condition,width in p['conditions'].items():
            started=time.monotonic();folder=out/condition;folder.mkdir()
            model=matched_model(p,width)
            torch.save(model.state_dict(),folder/'initial_state.pt');model.to('xpu')
            optimizer=torch.optim.AdamW(model.parameters(),lr=p['fit']['learning_rate'],weight_decay=p['fit']['weight_decay'])
            history=[];candidates=[]
            @lru_cache(maxsize=16)
            def pair_at(index):return read_pair(train[index],'engineering')
            e['conditions'][condition]=dict(status='running',completed_updates=0)
            write(out/'evidence.json',e)
            for step,indices in enumerate(schedule,1):
                row=accumulated_step(model,optimizer,[pair_at(int(i)) for i in indices],normalizer,width,
                                     condition,weights,step>p['fit']['warmup'],'xpu')
                history.append(dict(update=step,**row));e['total_optimizer_updates']+=1
                e['conditions'][condition]['completed_updates']=step
                if step%10==0:
                    print('FIT:',condition,step,'/300 loss',round(row['loss'],6),'elapsed_s',round(time.monotonic()-started),flush=True)
                    write(out/'evidence.json',e)
                if step%25==0:
                    checkpoint=folder/f'update_{step:04d}';checkpoint.mkdir()
                    torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                    optimizer_state=optimizer.state_dict(),rng_state=torch.get_rng_state(),xpu_rng_state=torch.xpu.get_rng_state(),
                                    config=asdict(model.config),normalizer=normalizer.state(),condition=condition,update=step,
                                    data_status=STATUS,protocol_sha256=declaration['protocol_sha256'],
                                    training_catalog_sha256=digest(paths['training_catalog']),schedule_sha256=digest(paths['sampling_schedule'])),checkpoint/'checkpoint.pt')
                    metrics=assess(model,dev,originals,normalizer,width,checkpoint)
                    candidate=dict(update=step,summary=metrics['summary'],preservation_qualified=preservation_qualified(metrics['summary']),
                                   checkpoint_path=str(checkpoint/'checkpoint.pt'),checkpoint_sha256=digest(checkpoint/'checkpoint.pt'),
                                   metrics_sha256=digest(checkpoint/'metrics.json'),predictions_sha256=digest(checkpoint/'predictions.npz'))
                    candidates.append(candidate)
                    selected=select_checkpoint(candidates,zero_error)
                    write(folder/'history.json',dict(data_status=STATUS,steps=history))
                    write(folder/'selection.json',dict(data_status=STATUS,partition='development',candidates=candidates,
                          selected_update=selected,selected_policy='zero_correction' if selected is None else 'exploratory_checkpoint',
                          zero_correction_plateau_error_cents=zero_error,model_adopted=False))
                    e['conditions'][condition].update(selected_update=selected,selected_policy='zero_correction' if selected is None else 'exploratory_checkpoint')
                    write(out/'evidence.json',e)
                    print('DEV:',condition,step,'plateau',metrics['summary']['dataset_balanced_macro']['repair_error_cents'],
                          'preservation',candidate['preservation_qualified'],'selected',selected,flush=True)
            pair_at.cache_clear()
            reload_update=selected if selected is not None else 300
            reload_path=folder/f'update_{reload_update:04d}'/'checkpoint.pt'
            saved=torch.load(reload_path,weights_only=True,map_location='cpu')
            reloaded=CorrectionEncoder(ModelConfig(**saved['config']));reloaded.load_state_dict(saved['state']);reloaded.to('xpu')
            assess(reloaded,dev,originals,normalizer,width,folder/'reload_verification')
            verification=same_vectors(reload_path.parent/'predictions.npz',folder/'reload_verification/predictions.npz')
            write(folder/'reload_verification/verification.json',dict(update=reload_update,**verification))
            e['conditions'][condition].update(status='complete',elapsed_seconds=time.monotonic()-started,
                                             reload_update=reload_update,reload_verification=verification,model=model.describe())
            write(out/'evidence.json',e)
            del reloaded,model,optimizer,saved
        authenticate({**frozen,**pd['pipeline_files']})
        e.update(status='complete',frozen_inputs_unchanged=len(frozen),common_pipeline_files_unchanged=len(pd['pipeline_files']),
                 artifacts={str(path.relative_to(out)):digest(path) for path in out.rglob('*') if path.is_file() and path.name!='evidence.json'})
        if e['total_optimizer_updates']!=900:raise ValueError('Incomplete declared optimizer budget')
        write(out/'evidence.json',e)
        print('EXPANDED_COMPLETE:',out,flush=True)
    except BaseException as error:
        e.update(status='stopped',failure=repr(error),no_automatic_repeat=True)
        write(out/'evidence.json',e)
        raise
    return out
