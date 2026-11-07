"""One loss-aggregation contrast on the exact verified V65 uniform trace.

Four frozen uniform parents, four new fits; no new data/labels or promotion.
"""
from pathlib import Path
import argparse,hashlib,inspect,time
import numpy as np
import torch
from torch.nn import functional as F
import train_paired_short_models as paired
import compare_preservation_sampling as previous
from bpdr_input.experiments.learning import tail_preservation as tail

WS=paired.WS
DATA=previous.DATA
ROOT=WS/'results/transformer_usability/2026-11-04_v68_tail_preservation'
ARMS=['frozen_uniform','upper_tail_preservation']
read,write,norm,sha=paired.read,paired.write,paired.norm,paired.sha
model_module=paired.model_module
virtual_pools,draw_virtual=paired.virtual_pools,paired.draw_virtual
OLD_TERM='preservation=.5*p[24:36].square().mean()+.5*p[36:].square().mean()'
NEW_TERM='preservation=tail.upper_tail_preservation(p[24:36],p[36:])'


def declare():
    parent,rows,originals=previous.declare()
    assert inspect.getsource(fit).strip()==inspect.getsource(paired.fit).replace(OLD_TERM,NEW_TERM).strip()
    receipts={}
    for name in ['2026-11-04_v65_preservation_sampling','2026-11-04_v66_preservation_delivery','2026-11-04_v67_preservation_diagnosis']:
        root=ROOT.parent/name;proof=read(root/'integrity_receipt.json');assert proof['results_sha256']==sha(root/'results.json');receipts[str(root/'integrity_receipt.json')]=sha(root/'integrity_receipt.json')
    assert read(previous.ROOT/'integrity_receipt.json')['checkpoints_reloaded']==8
    checkpoints={}
    for fold in range(1,5):
        path=previous.ROOT/f'fold_{fold}/uniform/checkpoint.pt';assert read(path.parent/'fit_receipt.json')['checkpoint_sha256']==sha(path);checkpoints[str(path)]=sha(path)
    base={k:parent[k] for k in ['folds','seed','updates','batch_size','forward_rows_per_update','config','optimizer','virtual_widths_seconds']}
    paths=[Path(__file__),Path(paired.__file__),Path(paired.previous.__file__),Path(model_module.__file__),Path(tail.__file__),WS/'tests/test_tail_preservation.py']
    d=norm(dict(base,version='fixed-upper-third-negative-loss-1',arms=ARMS,new_fits=4,virtual_updates=0,
        source_hashes={str(p.resolve()):sha(p) for p in paths},parent_integrities=receipts,parent_checkpoints=checkpoints,
        parent_declaration_sha256=sha(previous.ROOT/'declaration.json'),data_integrity_sha256=sha(DATA/'integrity_receipt.json'),
        changed_variable='C-zero/generalA loss aggregation only: mean largest4 of12 squared normalized corrections, coefficient.5 per class. No new frames, data, targets or sampling.',
        fixed='Same exact V65 uniform trace/model/init339/1200 updates/48 forward rows/optimizer/pairedA,pairedC,response/activity losses/runtime controller/render/audit.',
        baseline='Four frozen independently verified V65 uniform parents, unchanged held arrays; not mined parents.',
        loss='0.5*(pairedA^2+(C-target)^2)+0.25*(C-pairedA-target)^2+0.5*top4(Czero^2)+0.5*top4(generalA^2)+0.05*BCE(activity). Corrections normalized30c.',
        gate='Existing represented-group raw and actual-WAV original/known-zero/coverage/reduction gates remain. Zero-command collapse is not success; retain all exclusions and unassessed groups.',
        stopping='Exactly4 new fits,no k/seed/epoch/threshold sweep after failure. Independent raw and actual output replay before any adoption.',
        physical_attempts=1152,scoreable_pairs=445,original_attempts=72,original_controls=53,
        scope='Inspected training-source grouped engineering study, not fresh confirmation, musical intention or full BPDR-vs-augmentation/frame-only superiority.',
        nontraining_sources_accessed=False,research_validation_passed=False,production_eligible=False))
    ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/'declaration.json'
    if path.exists():assert read(path)==d
    else:write(path,d)
    return d,rows,originals


def fit(train,arm,trace,d,fold,folder):
    torch.manual_seed(d['seed']);model=model_module.ShortPhraseTransformer(model_module.ShortConfig(**d['config']))
    optimizer=torch.optim.AdamW(model.parameters(),**d['optimizer']);originals={r['source_id']:r for r in train if r['kind']=='original'}
    rng=np.random.default_rng(d['seed']+7919+fold);pools=virtual_pools(train,d);virtual_trace=[];history=[]
    start=time.perf_counter();model.train()
    for step,batch in enumerate(trace):
        arows=[];crows=[];zeros=[];original_rows=[];targets=[]
        if arm=='virtual_then_paired' and step<d['virtual_updates']:
            for _ in range(12):
                a,c,z,target,record=draw_virtual(train,pools,rng,d)
                arows.append(a);crows.append(c);zeros.append(z);targets.append(target)
                virtual_trace.append(dict(update=step,**record))
        else:
            for i,j in batch[:12]:
                c=train[i];a=originals[c['source_id']]
                arows.append((a['short_tokens'][j],a['short_pad'][j]));crows.append((c['short_tokens'][j],c['short_pad'][j]));targets.append(float(c['target'][j]))
            for i,j in batch[12:24]:zeros.append((train[i]['short_tokens'][j],train[i]['short_pad'][j]))
        for i,j in batch[24:]:original_rows.append((train[i]['short_tokens'][j],train[i]['short_pad'][j]))
        packed=arows+crows+zeros+original_rows
        x=torch.from_numpy(np.stack([p[0] for p in packed]));pad=torch.from_numpy(np.stack([p[1] for p in packed]))
        target=torch.tensor(targets,dtype=torch.float32)/30
        pred,logit,_=model(x,pad,True);p=pred/30
        anchor=.5*(p[:12].square().mean()+(p[12:24]-target).square().mean())
        response=.25*(p[12:24]-p[:12]-target).square().mean()
        preservation=tail.upper_tail_preservation(p[24:36],p[36:])
        active=torch.cat([torch.zeros(12),torch.ones(12),torch.zeros(24)])
        loss=anchor+response+preservation+.05*F.binary_cross_entropy_with_logits(logit,active)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%200==0:
            history.append(dict(update=step+1,loss=float(loss.detach())))
            print('PAIRED_SHORT_FIT',fold,arm,step+1,float(loss.detach()),flush=True)
    metadata=dict(arm=arm,training_groups=sorted({r['group'] for r in train}),training_sources=sorted({r['source_id'] for r in train}),
        declaration_sha256=sha(ROOT/'declaration.json'),updates=d['updates'],seed=d['seed'])
    model.eval();model_module.save(folder/'checkpoint.pt',model,metadata)
    write(folder/'virtual_trace.json',virtual_trace);write(folder/'loss_history.json',history)
    return model,dict(elapsed_seconds=time.perf_counter()-start,virtual_examples=len(virtual_trace),
        virtual_trace_sha256=sha(folder/'virtual_trace.json'),parameters=sum(p.numel() for p in model.parameters()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--declare-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);d,rows,originals=declare()
    if args.declare_only:print('TAIL_PRESERVATION_DECLARED',flush=True);return
    data=paired.records(rows,originals);predictions={arm:{} for arm in ARMS}
    with np.load(previous.ROOT/'uniform_all_predictions.npz') as z:baseline={k:z[k].copy() for k in z.files}
    for fold,groups in enumerate(d['folds'],1):
        train=[r for r in data if r['group'] not in groups];test=[r for r in data if r['group'] in groups]
        assert not {r['source_id'] for r in train}&{r['source_id'] for r in test}
        expected,digest=paired.previous.sample_trace(train,d)
        trace=np.load(previous.ROOT/f'fold_{fold}/uniform/physical_trace.npy');np.testing.assert_array_equal(trace,expected)
        assert digest==read(previous.ROOT/f'fold_{fold}/uniform/fit_receipt.json')['physical_trace_sha256']
        folder=ROOT/f'fold_{fold}/upper_tail_preservation';folder.mkdir(parents=True,exist_ok=True);np.save(folder/'physical_trace.npy',trace)
        path=folder/'fit_receipt.json'
        if path.exists():
            receipt=read(path);assert receipt['declaration_sha256']==sha(ROOT/'declaration.json') and receipt['physical_trace_sha256']==digest
            assert receipt['checkpoint_sha256']==sha(folder/'checkpoint.pt');model,_=model_module.load(folder/'checkpoint.pt')
        else:
            model,info=fit(train,'upper_tail_preservation',trace,d,fold,folder)
            receipt=dict(info,declaration_sha256=sha(ROOT/'declaration.json'),physical_trace_sha256=digest,checkpoint_sha256=sha(folder/'checkpoint.pt'),
                training_groups=sorted({r['group'] for r in train}),training_sources=sorted({r['source_id'] for r in train}),heldout_groups=groups,heldout_sources=sorted({r['source_id'] for r in test}))
            write(path,receipt)
        pred={r['id']:paired.previous.predict(model,r,'short_transformer') for r in test};np.savez_compressed(folder/'heldout_predictions.npz',**pred)
        predictions['upper_tail_preservation'].update(pred);predictions['frozen_uniform'].update({r['id']:baseline[r['id']] for r in test})
        print('TAIL_PRESERVATION_FOLD_COMPLETE',fold,flush=True)
    results={}
    for arm,pred in predictions.items():
        np.savez_compressed(ROOT/(arm+'_all_predictions.npz'),**pred)
        results[arm]={name:paired.previous.summarize(subset,pred) for name,subset in [('all',data),('humming',[r for r in data if r['group'].startswith('HumTrans:')]),('vocadito',[r for r in data if not r['group'].startswith('HumTrans:')])]}
    write(ROOT/'results.json',dict(declaration_sha256=sha(ROOT/'declaration.json'),arms=results,physical_attempts=1152,scoreable_pairs=445,original_attempts=72,original_controls=53,
        groups_without_positive_pairs=sorted({r['group'] for r in originals}-{r['group'] for r in rows if 'excluded' not in r}),
        raw_predictions_only=True,delivered_audio_qualification=False,production_eligible=False))
    print('TAIL_PRESERVATION_COMPLETE',{a:s['all']['positive_singer_macro_mae_cents'] for a,s in results.items()},flush=True)


if __name__=='__main__':main()
