"""Post-hoc Gate E input/response diagnosis. Inference only; no corpus changes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import uuid
import numpy as np
import torch
from .examples import read_pair, collate, Normalizer
from .learning_control import move, write
from .matched_comparison import matched_model
from .multisource_control import ARMS, prediction_rows, summarize
from .shared_exports import digest

GROUPS=dict(log_mel=(0,64),pitch=(64,65),confidence=(65,66),voicing=(66,67),
            pitch_change=(67,68),energy=(68,69),onset=(69,70),boundary_negative_control=(70,71))
ALPHAS=(0.,.25,.5,.75,1.)


def aligned_arrays(pair):
    a,c=pair['inputs']['A'],pair['inputs']['C']
    m=np.asarray(pair['mask'],bool);e=np.asarray(pair['detuning'])
    if a.shape!=c.shape or a.shape!=(len(m),71) or e.shape!=m.shape:
        raise ValueError('Misaligned diagnostic arrays')
    if not np.isfinite(a).all() or not np.isfinite(c).all():raise ValueError('Nonfinite features')
    p=m&(e!=0);o=m&(e==0)
    if not p.any() or not o.any():raise ValueError('Missing plateau/outside region')
    return a,c,p,o


def feature_stats(record,pair,scale):
    scale=np.asarray(scale,dtype=float)
    if scale.shape!=(71,) or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Invalid feature scales')
    a,c,p,o=aligned_arrays(pair);delta=c[p,64].astype(float)-a[p,64].astype(float)
    expected=float(record['cents']);groups={}
    for name,(lo,hi) in GROUPS.items():
        difference=(c[p,lo:hi].astype(float)-a[p,lo:hi].astype(float))/scale[lo:hi]
        groups[name]=dict(normalized_rms=float(np.sqrt(np.mean(difference**2))),
            normalized_mean_abs=float(abs(difference).mean()),
            exactly_equal_fraction=float(np.mean(c[p,lo:hi]==a[p,lo:hi])))
    return dict(source_id=record['source_id'],dataset=record['dataset'],cents=expected,
        plateau_frames=int(p.sum()),outside_frames=int(o.sum()),
        original_pitch_median_cents=float(np.median(a[p,64])),
        measured_pitch_delta_median_cents=float(np.median(delta)),pitch_delta_mae_cents=float(abs(delta-expected).mean()),
        pitch_delta_p90_error_cents=float(np.quantile(abs(delta-expected),.9)),
        pitch_delta_numerical_zero_fraction=float(np.mean(abs(delta)<1e-6)),
        pitch_delta_correct_direction_fraction=float(np.mean(np.sign(delta)==np.sign(expected))),
        groups=groups)


def group_replacement(a,c,name):
    """Replace one normalized C feature group by A; deliberately off-distribution."""
    if name not in GROUPS:raise ValueError('Unknown feature group')
    result=c.clone();lo,hi=GROUPS[name];result[:,:,lo:hi]=a[:,:,lo:hi]
    return result


def blend(a,c,alpha):
    """Diagnostic interpolation only; endpoint clones preserve exact input tensors."""
    if not 0<=alpha<=1:raise ValueError('Invalid alpha')
    if alpha==0:return a.clone()
    if alpha==1:return c.clone()
    return a*(1-alpha)+c*alpha


def response_row(record,pair,prediction,baseline,original):
    _,_,p,o=aligned_arrays(pair)
    if any(x.shape!=p.shape or not np.isfinite(x).all() for x in [prediction,baseline,original]):
        raise ValueError('Invalid prediction arrays')
    return dict(source_id=record['source_id'],dataset=record['dataset'],cents=float(record['cents']),
        plateau_mean_correction_cents=float(prediction[p].mean()),
        plateau_prediction_change_from_C_cents=float(abs(prediction[p]-baseline[p]).mean()),
        outside_prediction_change_from_C_cents=float(abs(prediction[o]-baseline[o]).mean()),
        response_from_A_cents=float((prediction[p]-original[p]).mean()))


def declare(root):
    prior=Path(json.loads((root/'runs/multisource_control_current.json').read_text())['path'])
    d=json.loads((prior/'declaration.json').read_text());e=json.loads((prior/'evidence.json').read_text())
    verification=json.loads((prior/'independent_verification.json').read_text())
    if digest(prior/'evidence.json')!=verification['evidence_sha256'] or verification['status']!='verified':
        raise ValueError('Previous experiment receipt changed')
    if digest(prior/'declaration.json')!=e['declaration_sha256']:raise ValueError('Previous declaration changed')
    frozen={str(f):digest(f) for f in prior.rglob('*') if f.is_file()}
    # Pin exactly the input artifacts used here; retain the previous full-corpus receipt as historical evidence.
    for record in d['records']:
        if record['split']!='train' or record['training_eligible']:raise ValueError('Wrong partition/status')
        for view in 'ABCD':
            f=Path(record['input_directory'])/(view+'_input.npz')
            if digest(f)!=record['input_sha256'][view]:raise ValueError('Changed input')
            frozen[str(f)]=record['input_sha256'][view]
        for key in ['target_path','audit_path']:
            f=record[key];frozen[f]=digest(f)
        if frozen[record['target_path']]!=record['targets_sha256']:raise ValueError('Changed target')
    frozen[d['normalizer']]=digest(d['normalizer'])
    frozen.update(d['pipeline_files'])
    for f in (root/'src/bpdr_input').glob('*.py'):frozen[str(f)]=digest(f)
    for f,h in frozen.items():
        if digest(f)!=h:raise ValueError('Changed pinned artifact: '+f)
    out=root/'runs/input_response_diagnosis'/uuid.uuid4().hex;out.mkdir(parents=True)
    shutil.copyfile(__file__,out/'input_response_diagnosis_at_declaration.py')
    declaration=dict(version='gate-e-input-response-diagnosis-1',declared_utc=datetime.now(timezone.utc).isoformat(),
        post_hoc=True,selection_use=False,optimizer_updates=0,data_status='exploratory_not_research',
        training_eligible=False,production_eligible=False,research_ready_examples=0,
        previous_run=str(prior),previous_declaration=d,arms=list(ARMS),groups=GROUPS,alphas=ALPHAS,
        scope='Same eight training pairs, all three final diagnostic models. Saved A/B/C/D residuals; sensitivity inference uses nominal-boundary A/C only.',
        measurements='Pitch-shift direction/error/zero fraction; per-group normalized RMS; cross-direction plateau intersection; residual replay; C group replaced by original A; whole-feature A-to-C interpolation.',
        counterfactual_limit='Off-distribution feature diagnostics, not real waveforms, causal attribution, validation accuracy or eligible runtime inputs. Paired originals are unavailable during ordinary inference.',
        no_new_fit=True,no_new_audio_processing=True,no_development_or_reserved_processing=True,
        frozen_inputs=frozen,pipeline_files=d['pipeline_files'])
    write(out/'declaration.json',declaration);write(root/'runs/input_response_diagnosis_current.json',dict(path=str(out)))
    return out


def load(d):
    old=d['previous_declaration'];pairs=[read_pair(r,'engineering') for r in old['records']]
    with np.load(old['normalizer']) as z:
        normalizer=Normalizer(z['mean'],z['scale']);scale=z['scale'].copy()
    return old,pairs,collate(pairs,normalizer),scale


def run(out):
    out=Path(out);d=json.loads((out/'declaration.json').read_text());prior=Path(d['previous_run'])
    if (out/'sensitivity_predictions.npz').exists() or (out/'evidence.json').exists():raise ValueError('Run already started')
    for p,h in d['frozen_inputs'].items():
        if digest(p)!=h:raise ValueError('Changed input: '+p)
    torch.set_num_threads(2);old,pairs,cpu,scale=load(d);batch=move(cpu,'xpu')
    records=old['records'];stats=[feature_stats(r,p,scale) for r,p in zip(records,pairs)]
    cross=[]
    for i in range(0,8,2):
        a,b=pairs[i],pairs[i+1];_,ca,pa,_=aligned_arrays(a);_,cb,pb,_=aligned_arrays(b);m=pa&pb
        if not m.any():raise ValueError('No common cross-direction plateau')
        cross.append(dict(source_id=records[i]['source_id'],frames=int(m.sum()),
            plus_minus_pitch_difference_median_cents=float(np.median(cb[m,64]-ca[m,64])),
            exactly_equal_full_frame_fraction=float(np.mean(np.all(ca[m]==cb[m],axis=1)))))
    protocol=dict(fit=dict(seed=old['seed']),model=old['model']);saved={};results={}
    with torch.inference_mode():
        for arm in ARMS:
            state=torch.load(prior/arm/'diagnostic_state.pt',map_location='cpu',weights_only=True)
            if not state['diagnostic_only'] or state['production_eligible']:raise ValueError('Invalid diagnostic state')
            model=matched_model(protocol,71);model.load_state_dict(state['state']);model.to('xpu').eval().requires_grad_(False)
            with np.load(prior/arm/'final_predictions.npz') as z:baseline={k:z[k].copy() for k in z.files}
            rows=prediction_rows(records,pairs,baseline)
            def evaluate(name,x):
                values=model(x,batch['padding']).cpu().numpy();r=[]
                for i,(record,pair) in enumerate(zip(records,pairs)):
                    p=values[i,:len(pair['mask'])].copy();saved[f'{arm}__{name}__{i}']=p
                    r.append(response_row(record,pair,p,baseline[f'{i}_C'],baseline[f'{i}_A']))
                return r
            interpolation={str(alpha):evaluate('alpha_'+str(alpha),blend(batch['inputs']['A'],batch['inputs']['C'],alpha)) for alpha in ALPHAS}
            groups={name:evaluate('group_'+name,group_replacement(batch['inputs']['A'],batch['inputs']['C'],name)) for name in GROUPS}
            endpoint_max=0.
            for alpha,view in [(0.,'A'),(1.,'C')]:
                for i in range(8):
                    actual=saved[f'{arm}__alpha_{alpha}__{i}'];expected=baseline[f'{i}_{view}']
                    endpoint_max=max(endpoint_max,float(abs(actual-expected).max()))
                    if not np.allclose(actual,expected,atol=1e-5,rtol=1e-5):raise ValueError('Endpoint replay mismatch')
            if not all(torch.equal(t.detach().cpu(),state['state'][k]) for k,t in model.state_dict().items()):
                raise ValueError('Model state changed during inference')
            results[arm]=dict(saved_prediction_metrics=rows,summary=summarize(rows),
                interpolation=interpolation,group_replacement=groups,endpoint_max_difference_cents=endpoint_max,
                parameters_unchanged=True)
            print('INFERENCE_COMPLETE',arm,'endpoint max',endpoint_max,flush=True)
            del model
    np.savez_compressed(out/'sensitivity_predictions.npz',**saved)
    for p,h in d['frozen_inputs'].items():
        if digest(p)!=h:raise ValueError('Changed pinned artifact: '+p)
    write(out/'evidence.json',dict(status='complete',declaration_sha256=digest(out/'declaration.json'),
        post_hoc=True,selection_use=False,optimizer_updates=0,research_ready_examples=0,
        training_eligible=False,production_eligible=False,torch_version=torch.__version__,interpreter=sys.executable,
        feature_statistics=stats,cross_direction=cross,arms=results,prediction_vectors=len(saved),
        artifacts={str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file()}))
    print('COMPLETE',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['declare','run']);p.add_argument('--run',type=Path);a=p.parse_args()
    if a.command=='declare':print(declare(Path(__file__).resolve().parents[2]))
    else:run(a.run)


if __name__=='__main__':main()
