"""Declared training-only Gate E preservation diagnostic; no model adoption."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np
import torch
from .examples import read_pair, collate, Normalizer
from .learning_control import balanced_objective, regions, move, forward, write
from .losses import objective, masked_mean_abs
from .matched_comparison import check_catalog, matched_model, macro, metric_row
from .shared_exports import digest

ARMS=('existing','balanced','preservation')
METRICS=('repair_error_cents','original_plateau_alteration_cents','original_audited_alteration_cents',
         'outside_alteration_cents','original_joint_voiced_no_change_cents','counter_detuning_error_cents','plateau_response_gain')


def select_records(records):
    """Catalog order only: first Vocadito source, first three different Jamendo artists, both shifts."""
    check_catalog(records,'train')
    groups={}
    for r in records:
        if r['corruption']=='split' and r['cents'] in (-30.,30.):
            key=(r['dataset'],r['group_id'],r['source_id'],r['phrase_id'])
            groups.setdefault(key,{})[r['cents']]=r
    selected=[];seen=set();counts={'vocadito':0,'jamendo':0}
    for key,versions in groups.items():
        ds,group,_,_=key
        if len(versions)!=2 or group in seen or counts[ds]>=(1 if ds=='vocadito' else 3):continue
        selected.extend([versions[-30.],versions[30.]])
        seen.add(group);counts[ds]+=1
    if counts!={'vocadito':1,'jamendo':3}:raise ValueError('Insufficient fixed sources with both audited shifts')
    return selected


def single(batch,i):
    return dict(inputs={v:x[i:i+1] for v,x in batch['inputs'].items()},
        voiced={v:x[i:i+1] for v,x in batch['voiced'].items()},
        **{k:batch[k][i:i+1] for k in ('mask','detuning','padding')})


def one_loss(r,batch,arm):
    if arm=='existing':
        total,parts,_=objective(r,batch,'bpdr',response=True)
        return total,parts
    if arm not in ARMS:raise ValueError('Unknown arm')
    total,parts=balanced_objective(r,batch)
    if arm=='preservation':
        p,o=regions(batch);zero=torch.zeros_like(batch['detuning'])
        protection=sum(.5*(masked_mean_abs(r[v],zero,p)[0]+masked_mean_abs(r[v],zero,o)[0]) for v in 'AB')/2
        total=total+2*protection
        parts=dict(parts,original_protection=protection)
    return total,parts


def weighted_loss(prediction,batch,weights,arm):
    total=prediction['A'].sum()*0;parts={}
    for i,weight in enumerate(weights):
        value,components=one_loss({v:x[i:i+1] for v,x in prediction.items()},single(batch,i),arm)
        total=total+weight*value
        for k,v in components.items():parts[k]=parts.get(k,0)+weight*v
    return total,parts


def prediction_rows(records,pairs,predictions):
    rows=[]
    for i,(record,pair) in enumerate(zip(records,pairs)):
        r={v:predictions[f'{i}_{v}'] for v in 'ABCD'}
        row=metric_row(record,r,pair['inputs'],pair['mask'],pair['detuning'])
        plateau=pair['mask']&(pair['detuning']!=0)
        row['original_plateau_alteration_cents']=float(np.mean([abs(r[v][plateau]).mean() for v in 'AB']))
        row['plateau_response_gain']=float(np.mean(np.concatenate([
            (r[v][plateau]-r[a][plateau])/(-pair['detuning'][plateau]) for v,a in [('C','A'),('D','B')]])))
        row['bounded_response_and_preservation_pass']=bool(row['repair_error_cents']<=27 and
            row['original_plateau_alteration_cents']<=1 and row['outside_alteration_cents']<=1)
        rows.append(row)
    return rows


def summarize(rows):
    return dict(dataset_balanced_macro={m:macro(rows,m) for m in METRICS},
        by_dataset={ds:{m:macro([r for r in rows if r['dataset']==ds],m) for m in METRICS} for ds in ['vocadito','jamendo']},
        direction={str(c):{m:macro([r for r in rows if r['cents']==c],m) for m in METRICS} for c in [-30.,30.]},
        passed_pairs=sum(r['bounded_response_and_preservation_pass'] for r in rows),
        all_pairs_pass=all(r['bounded_response_and_preservation_pass'] for r in rows))


def measure(model,batch,records,pairs,weights,arm):
    model.eval()
    with torch.no_grad():
        prediction=forward(model,batch)
        loss,parts=weighted_loss(prediction,batch,weights,arm)
    raw={f'{i}_{v}':prediction[v][i,:len(pair['mask'])].cpu().numpy().copy() for i,pair in enumerate(pairs) for v in 'ABCD'}
    rows=prediction_rows(records,pairs,raw)
    return dict(loss=float(loss.cpu()),parts={k:float(v.cpu()) for k,v in parts.items()},rows=rows,summary=summarize(rows)),raw


def declare(root):
    prep=Path(json.loads((root/'runs/comparison_preparation_current.json').read_text())['path'])
    records=select_records(json.loads((prep/'summary.json').read_text())['examples'])
    prior=Path(json.loads((root/'runs/learning_control_current.json').read_text())['path'])
    previous=json.loads((prior/'declaration.json').read_text())
    frozen=dict(previous['frozen_inputs'])
    frozen.update({str(f):digest(f) for f in prior.rglob('*') if f.is_file()})
    for record in records:
        for view in 'ABCD':
            path=Path(record['input_directory'])/(view+'_input.npz')
            if digest(path)!=record['input_sha256'][view]:raise ValueError('Changed catalog input')
            frozen[str(path)]=record['input_sha256'][view]
        if digest(record['target_path'])!=record['targets_sha256']:raise ValueError('Changed targets')
        frozen[record['target_path']]=record['targets_sha256']
    frozen[str(Path(__file__))]=digest(__file__)
    weights=[.25 if r['dataset']=='vocadito' else 1/12 for r in records]
    d=dict(version='gate-e-multisource-control-1',declared_utc=datetime.now(timezone.utc).isoformat(),
        data_status='exploratory_not_research',training_eligible=False,production_eligible=False,research_ready_examples=0,
        records=records,source_selection='Frozen catalog order: first Vocadito source and first three distinct Jamendo artists with both +/-30 split variants',
        arms=list(ARMS),updates_per_arm=100,total_updates=300,batch_size=8,weights=weights,
        loss_weighting='Equal dataset -> group/source -> shift; two Vocadito pairs weight .25 each, six Jamendo pairs weight 1/12 each',
        model=previous['model'],seed=339,learning_rate=.001,weight_decay=.01,gradient_clip=1.,warmup=0,
        full_phrase_context=True,normalizer=str(prep/'normalizer.npz'),
        preservation='Balanced objective + 2 * original A/B MAE, equal plateau/outside regions per example; no new input or modified target',
        selection='Final update 100 only; intermediate 0/25/50/75 descriptive; no development or heldout processing',
        diagnostic_rule='Every selected source/shift pair: detuned plateau MAE <=27 cents (10% improvement over zero), original plateau MAE <=1 cent, detuned outside MAE <=1 cent. Engineering triage only, not musical accuracy/approval.',
        limitations='Four training sources, one seed, fixed +/-30 synthetic shifts, one corruption. No independent accuracy, production eligibility or loss adoption. Original 900-update study and earlier 400-update diagnostic preserved.',
        frozen_inputs=frozen,pipeline_files=previous['pipeline_files'])
    for p,h in frozen.items():
        if digest(p)!=h:raise ValueError('Changed frozen evidence: '+p)
    out=root/'runs/multisource_control'/uuid.uuid4().hex;out.mkdir(parents=True)
    write(out/'declaration.json',d)
    write(root/'runs/multisource_control_current.json',dict(path=str(out)))
    return out


def load(d):
    pairs=[read_pair(r,'engineering') for r in d['records']]
    with np.load(d['normalizer']) as z:batch=collate(pairs,Normalizer(z['mean'],z['scale']))
    regions(batch)
    return pairs,batch


def run(out):
    out=Path(out);d=json.loads((out/'declaration.json').read_text())
    if any((out/a).exists() for a in ARMS):raise ValueError('Run started; preserve evidence')
    for p,h in d['frozen_inputs'].items():
        if digest(p)!=h:raise ValueError('Changed frozen evidence: '+p)
    if not torch.xpu.is_available():raise ValueError('XPU unavailable')
    torch.set_num_threads(2);pairs,cpu=load(d);batch=move(cpu,'xpu')
    protocol=dict(fit=dict(seed=d['seed']),model=d['model'])
    initial=matched_model(protocol,71).state_dict()
    torch.save(initial,out/'initial_state.pt')
    baseline=prediction_rows(d['records'],pairs,{f'{i}_{v}':np.zeros(len(p['mask']),dtype=np.float32) for i,p in enumerate(pairs) for v in 'ABCD'})
    results={}
    for arm in ARMS:
        start=time.monotonic();folder=out/arm;folder.mkdir()
        model=matched_model(protocol,71).to('xpu')
        # Save the actual initial tensors, not just a claim that the seeds match.
        torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},folder/'initial_state.pt')
        optimizer=torch.optim.AdamW(model.parameters(),lr=d['learning_rate'],weight_decay=d['weight_decay'])
        torch.manual_seed(d['seed'])
        history=[]
        m,raw=measure(model,batch,d['records'],pairs,d['weights'],arm)
        history.append(dict(update=0,**m));np.savez_compressed(folder/'initial_predictions.npz',**raw)
        zero={v:torch.zeros_like(batch['detuning']) for v in 'ABCD'}
        zero_loss=float(weighted_loss(zero,batch,d['weights'],arm)[0].cpu())
        for step in range(1,d['updates_per_arm']+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            loss,_=weighted_loss(forward(model,batch),batch,d['weights'],arm)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),d['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
            if step%25==0:
                m,raw=measure(model,batch,d['records'],pairs,d['weights'],arm)
                history.append(dict(update=step,gradient_norm=float(norm.cpu()),**m));write(folder/'history.json',history)
                print(arm,step,m['summary']['dataset_balanced_macro'],'passed_pairs',m['summary']['passed_pairs'],flush=True)
        np.savez_compressed(folder/'final_predictions.npz',**raw)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save(dict(state=state,diagnostic_only=True,training_eligible=False,production_eligible=False),folder/'diagnostic_state.pt')
        delta=float(sum((state[k]-initial[k]).double().square().sum() for k in state).sqrt())
        results[arm]=dict(final=history[-1],zero_loss=zero_loss,parameter_delta_l2=delta,seconds=time.monotonic()-start)
        del model,optimizer
    for p,h in d['frozen_inputs'].items():
        if digest(p)!=h:raise ValueError('Frozen evidence changed: '+p)
    write(out/'evidence.json',dict(status='complete',declaration_sha256=digest(out/'declaration.json'),
        data_status=d['data_status'],training_eligible=False,production_eligible=False,research_ready_examples=0,
        interpreter=sys.executable,torch_version=torch.__version__,device=torch.xpu.get_device_name(0),
        optimizer_updates=300,source_count=4,pair_count=8,zero_baseline=dict(rows=baseline,summary=summarize(baseline)),
        arms=results,artifacts={str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file()}))
    print('COMPLETE',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['declare','run']);p.add_argument('--run',type=Path);a=p.parse_args()
    if a.command=='declare':print(declare(Path(__file__).resolve().parents[2]))
    else:run(a.run)


if __name__=='__main__':main()
