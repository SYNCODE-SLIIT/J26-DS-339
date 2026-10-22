"""Fixed group-width scaling ablation against the authenticated saved Gate E control."""
from bpdr_input import COMPONENT_ROOT, module_source
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
import uuid
import numpy as np
import torch
from bpdr_input.experiments.learning.learning_control import write, move, forward
from bpdr_input.experiments.learning.matched_comparison import matched_model
from bpdr_input.experiments.learning.multisource_control import load, measure, weighted_loss
from bpdr_input.preparation.sources.shared_exports import digest

TRANSFORM='standardized_log_mel_div_sqrt64_v1'
PROTOCOL_KEYS=('model','seed','updates_per_arm','learning_rate','weight_decay','gradient_clip',
               'batch_size','warmup','records','weights','normalizer')


def scale_standardized_batch(batch):
    """Model-input treatment only. Targets, masks and on-disk features remain untouched."""
    if 'bpdr_input_transform' in batch:raise ValueError('Input transform already applied')
    inputs={}
    for view,x in batch['inputs'].items():
        if x.ndim!=3 or x.shape[-1]!=71 or not x.is_floating_point():raise ValueError('Expected standardized B,T,71 float features')
        gain=x.new_tensor([.125]*64+[1.]*7)
        inputs[view]=x*gain
    return dict(batch,inputs=inputs,bpdr_input_transform=TRANSFORM)


def check_protocol(d):
    if d['seed']!=339 or d['updates_per_arm']!=100 or d['batch_size']!=8 or d['warmup']!=0:
        raise ValueError('Saved control fit protocol mismatch')
    if d['learning_rate']!=.001 or d['weight_decay']!=.01 or d['gradient_clip']!=1.:
        raise ValueError('Saved control optimizer mismatch')
    if len(d['records'])!=8 or abs(sum(d['weights'])-1)>1e-12:raise ValueError('Saved control batch mismatch')
    if any(r['split']!='train' or not r['audit_passed'] or r['training_eligible'] for r in d['records']):
        raise ValueError('Wrong data status/partition')


def declare(root):
    prior=Path(json.loads((root/'runs/multisource_control_current.json').read_text())['path'])
    old=json.loads((prior/'declaration.json').read_text());e=json.loads((prior/'evidence.json').read_text())
    verification=json.loads((prior/'independent_verification.json').read_text())
    if verification['status']!='verified' or digest(prior/'evidence.json')!=verification['evidence_sha256']:
        raise ValueError('Changed saved control receipt')
    if digest(prior/'declaration.json')!=e['declaration_sha256']:raise ValueError('Changed saved declaration')
    check_protocol(old)
    for name in ['multisource_control.py','model.py','losses.py','examples.py','matched_comparison.py','learning_control.py']:
        f=module_source(name)
        if digest(f)!=old['frozen_inputs'][str(f)]:raise ValueError('Changed shared fit implementation: '+name)
    frozen={str(f):digest(f) for f in prior.rglob('*') if f.is_file()}
    for relative,h in e['artifacts'].items():
        if digest(prior/relative)!=h:raise ValueError('Changed control artifact')
    for r in old['records']:
        for view in 'ABCD':
            f=Path(r['input_directory'])/(view+'_input.npz')
            if digest(f)!=r['input_sha256'][view]:raise ValueError('Changed pair input')
            frozen[str(f)]=r['input_sha256'][view]
        if digest(r['target_path'])!=r['targets_sha256']:raise ValueError('Changed target')
        frozen[r['target_path']]=r['targets_sha256']
    frozen[old['normalizer']]=digest(old['normalizer']);frozen.update(old['pipeline_files'])
    diagnosis=Path(json.loads((root/'runs/input_response_diagnosis_current.json').read_text())['path'])
    for name in ['declaration.json','evidence.json','independent_verification.json','final_verification.json']:
        frozen[str(diagnosis/name)]=digest(diagnosis/name)
    for f in (root/'src/bpdr_input').rglob('*.py'):frozen[str(f)]=digest(f)
    out=root/'runs/feature_weighting_control'/uuid.uuid4().hex;out.mkdir(parents=True)
    shutil.copyfile(__file__,out/'feature_weighting_control_at_declaration.py')
    declaration=dict(version='gate-e-feature-weighting-control-1',declared_utc=datetime.now(timezone.utc).isoformat(),
        data_status='exploratory_not_research',training_eligible=False,production_eligible=False,research_ready_examples=0,
        saved_control_run=str(prior),saved_control_arm='balanced',control_reuse=True,control_updates_reexecuted=0,
        control_protocol={k:old[k] for k in PROTOCOL_KEYS},new_optimizer_updates=100,
        treatment=dict(name=TRANSFORM,after='Frozen training-only standardizer',gain_first_64=.125,gain_last_7=1.),
        loss='Exact existing multisource_control weighted_loss(..., arm=balanced), unchanged for both arms',
        initialization='Reconstruct saved initial tensors exactly; verify all 32 unscaled initial prediction vectors before fitting',
        runtime_match=dict(torch_version=e['torch_version'],device=e['device'],interpreter=e['interpreter']),
        randomization='Same seed 339, reset after AdamW construction; same four view forwards/dropout calls and complete fixed batch each update',
        selection='Final update 100 only; history at 0/25/50/75/100 descriptive. No best checkpoint selection.',
        rule='Every pair: plateau error <=27 cents, original plateau alteration <=1 cent, detuned outside alteration <=1 cent. Provisional training triage only.',
        design='Post-hoc hypothesis; gain fixed by sqrt(64)=8, no gain search. Saved control is historical, not a concurrent rerun.',
        constraints='No paired-original or target-derived input feature, no new audio/pair generation, no development or reserved acoustic processing, no deployment/adoption.',
        frozen_inputs=frozen,pipeline_files=old['pipeline_files'])
    for f,h in frozen.items():
        if digest(f)!=h:raise ValueError('Changed pinned artifact: '+f)
    write(out/'declaration.json',declaration);write(root/'runs/feature_weighting_control_current.json',dict(path=str(out)))
    return out


def run(out):
    out=Path(out);d=json.loads((out/'declaration.json').read_text());old=d['control_protocol'];prior=Path(d['saved_control_run'])
    if (out/'initial_state.pt').exists():raise ValueError('Run already started; preserve evidence')
    for f,h in d['frozen_inputs'].items():
        if digest(f)!=h:raise ValueError('Changed pinned artifact: '+f)
    if not torch.xpu.is_available():raise ValueError('XPU unavailable')
    runtime=d['runtime_match']
    if torch.__version__!=runtime['torch_version'] or torch.xpu.get_device_name(0)!=runtime['device'] or Path(sys.executable).resolve()!=Path(runtime['interpreter']).resolve():
        raise ValueError('Saved control runtime mismatch; no historical-control comparison authorized by this declaration')
    torch.set_num_threads(2);pairs,cpu=load(old);raw_batch=move(cpu,'xpu')
    batch=move(scale_standardized_batch(cpu),'xpu')
    protocol=dict(fit=dict(seed=old['seed']),model=old['model']);model=matched_model(protocol,71).to('xpu')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    saved_initial=torch.load(prior/'balanced/initial_state.pt',map_location='cpu',weights_only=True)
    if not all(torch.equal(v,saved_initial[k]) for k,v in initial.items()):raise ValueError('Initialization mismatch')
    m,unscaled=measure(model,raw_batch,old['records'],pairs,old['weights'],'balanced')
    maximum=0.
    with np.load(prior/'balanced/initial_predictions.npz') as z:
        for k,x in unscaled.items():
            maximum=max(maximum,float(abs(x-z[k]).max()))
            if not np.allclose(x,z[k],atol=1e-5,rtol=1e-5):raise ValueError('Unscaled initial prediction mismatch')
    write(out/'control_compatibility.json',dict(status='verified',initial_state_exact=True,
        unscaled_initial_vectors_replayed=32,maximum_initial_prediction_difference_cents=maximum,
        implementation_hashes_match=True,runtime_match=True,control_retrained=False))
    torch.save(initial,out/'initial_state.pt')
    optimizer=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'],weight_decay=old['weight_decay'])
    torch.manual_seed(old['seed'])
    start=time.monotonic();m,predictions=measure(model,batch,old['records'],pairs,old['weights'],'balanced')
    history=[dict(update=0,**m)];np.savez_compressed(out/'initial_predictions.npz',**predictions)
    zero={v:torch.zeros_like(batch['detuning']) for v in 'ABCD'}
    zero_loss=float(weighted_loss(zero,batch,old['weights'],'balanced')[0].cpu())
    for step in range(1,old['updates_per_arm']+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        loss,_=weighted_loss(forward(model,batch),batch,old['weights'],'balanced')
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),old['gradient_clip'],error_if_nonfinite=True)
        optimizer.step()
        if step%25==0:
            m,predictions=measure(model,batch,old['records'],pairs,old['weights'],'balanced')
            history.append(dict(update=step,gradient_norm=float(norm.cpu()),**m));write(out/'history.json',history)
            print('GROUP_WEIGHTING',step,m['summary']['dataset_balanced_macro'],'passed_pairs',m['summary']['passed_pairs'],flush=True)
    np.savez_compressed(out/'final_predictions.npz',**predictions)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(dict(state=state,input_transform=TRANSFORM,diagnostic_only=True,training_eligible=False,production_eligible=False),out/'diagnostic_state.pt')
    delta=float(sum((state[k]-initial[k]).double().square().sum() for k in state).sqrt())
    prior_evidence=json.loads((prior/'evidence.json').read_text())
    for f,h in d['frozen_inputs'].items():
        if digest(f)!=h:raise ValueError('Changed pinned artifact: '+f)
    write(out/'evidence.json',dict(status='complete',declaration_sha256=digest(out/'declaration.json'),
        data_status=d['data_status'],training_eligible=False,production_eligible=False,research_ready_examples=0,
        new_optimizer_updates=100,control_optimizer_updates_reexecuted=0,source_count=4,pair_count=8,
        torch_version=torch.__version__,device=torch.xpu.get_device_name(0),interpreter=sys.executable,
        control=prior_evidence['arms']['balanced']['final'],treatment=history[-1],zero_loss=zero_loss,
        parameter_delta_l2=delta,seconds=time.monotonic()-start,input_transform=TRANSFORM,
        artifacts={str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file()}))
    print('COMPLETE',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['declare','run']);p.add_argument('--run',type=Path);a=p.parse_args()
    if a.command=='declare':print(declare(COMPONENT_ROOT))
    else:run(a.run)


if __name__=='__main__':main()
