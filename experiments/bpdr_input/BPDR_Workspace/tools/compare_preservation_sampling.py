"""One fixed-budget comparison: uniform versus frozen-parent preservation sampling.

Execution refuses unverified V64 data. No development/reserved audio or promotion.
"""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import torch
import train_paired_short_models as paired
import compare_humtrans_short_transfer as parent
from bpdr_input.experiments.learning import preservation_mining as mining
from bpdr_input.repair import short_phrase_policy as policy

WS=paired.WS
DATA=WS/'results/transformer_usability/2026-11-04_v64_reliable_melodic_pairs'
ROOT=WS/'results/transformer_usability/2026-11-04_v65_preservation_sampling'
read,write,norm,sha=paired.read,paired.write,paired.norm,paired.sha
ARMS=['uniform','preservation_mined']


def declare():
    paired.check_lock()
    proof=read(DATA/'integrity_receipt.json')
    assert proof['planned_attempts_retained']==1152 and proof['valid_originals']==53
    for field,name in [('train_pairs_sha256','train_pairs.json'),('train_originals_sha256','train_originals.json'),('input_hashes_sha256','input_hashes.json'),('summary_sha256','summary.json')]:
        assert proof[field]==sha(DATA/name)
    for path,h in read(DATA/'input_hashes.json').items():assert sha(path)==h
    old=read(parent.ROOT/'declaration.json');parent_proof=read(parent.ROOT/'integrity_receipt.json')
    assert parent_proof['checkpoints_reloaded']==8
    checkpoints={}
    for fold in range(1,5):
        path=parent.ROOT/f'fold_{fold}/checkpoint.pt'
        assert read(path.parent/'fit_receipt.json')['checkpoint_sha256']==sha(path)
        checkpoints[str(path)]=sha(path)
    sources=[Path(__file__),Path(paired.__file__),Path(paired.previous.__file__),Path(paired.model_module.__file__),Path(mining.__file__),Path(policy.__file__)]
    base={k:old[k] for k in ('seed','updates','batch_size','forward_rows_per_update','config','optimizer','loss')}
    d=norm(dict(base,folds=old['folds'],version='fixed-parent-preservation-sampling-1',arms=ARMS,new_fits=8,
        source_hashes={str(p.resolve()):sha(p) for p in sources},parent_checkpoints=checkpoints,
        data_integrity_sha256=sha(DATA/'integrity_receipt.json'),
        parent_integrity_sha256=sha(parent.ROOT/'integrity_receipt.json'),parent_declaration_sha256=sha(parent.ROOT/'declaration.json'),
        virtual_widths_seconds=[.24,.28,.36,.50],virtual_updates=0,virtual_pretraining=False,
        gate='Every represented positive group MAE<25c and signed response>=.25; every original group mean<=3c. Delivered original-event,known-zero,benefit and coverage requirements remain separate and unchanged.',
        changed_variable='Negative frame selection only: fixed-parent top20% unprotected known-zero requests >=6c,6 of12 draws per negative class. Preserve class, singer, record and positive draws.',
        fixed='Same verified V64 data,folds,21666-parameter Transformer,paired loss,seed339,1200 updates,optimizer,controller,renderer,audit and old raw/delivered gates. No virtual pretraining.',
        baseline='New uniform-sampling fit on exactly the same V64 pool; historical V51 is the fixed mining teacher, not a same-data control.',
        teacher_scope='Predict training records only. Each frozen V51 teacher excludes the fold held-out people; expanded windows inherit original singer/source roles.',
        stopping='Exactly4 folds x2 arms; no seed/threshold sweep after failure. No activation without independent checkpoint/sampler and actual-WAV replay plus existing qualification gates.',
        physical_attempts=1152,original_attempts=72,original_source_recordings=33,
        fresh_confirmation=False,delivered_audio_qualification=False,production_eligible=False,research_validation_passed=False))
    ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/'declaration.json'
    if path.exists():assert read(path)==d
    else:write(path,d)
    return d,read(DATA/'train_pairs.json'),read(DATA/'train_originals.json')


def mining_evidence(train,teacher):
    predictions=[];protected=[]
    for record in train:
        p=paired.previous.predict(teacher,record,'short_transformer')
        raw=record['raw'];n=len(raw)
        # The policy derives protection exclusively from the supplied raw input.
        # Zero placeholder audio cannot influence this returned protection mask.
        evidence=policy.proposal(np.zeros((n-1)*441+1,np.float32),np.arange(n)*.02,np.stack([raw]*3),np.zeros((3,n)))
        predictions.append(p);protected.append(evidence['protected'])
    return predictions,protected


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--declare-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);d,rows,originals=declare()
    if args.declare_only:print('PRESERVATION_COMPARISON_DECLARED',flush=True);return
    data=paired.records(rows,originals);predictions={arm:{} for arm in ARMS}
    for fold,held in enumerate(d['folds'],1):
        train=[r for r in data if r['group'] not in held];test=[r for r in data if r['group'] in held]
        assert not {r['source_id'] for r in train}&{r['source_id'] for r in test}
        teacher,meta=paired.model_module.load(parent.ROOT/f'fold_{fold}/checkpoint.pt')
        assert not set(meta['training_groups'])&set(held)
        trace,_=paired.previous.sample_trace(train,d)
        p,guards=mining_evidence(train,teacher)
        mined,stats=mining.mined_trace(train,trace,p,guards,d['seed']+fold)
        base=ROOT/f'fold_{fold}';base.mkdir(exist_ok=True)
        evidence={key+str(i):value for i in range(len(train)) for key,value in [('prediction_',p[i]),('protected_',guards[i])]}
        np.savez_compressed(base/'mining_evidence.npz',**evidence)
        write(base/'mining_stats.json',dict(stats,training_record_ids=[r['id'] for r in train]))
        for arm,draws in [('uniform',trace),('preservation_mined',mined)]:
            folder=base/arm;folder.mkdir(exist_ok=True);digest=hashlib.sha256(draws.tobytes()).hexdigest();np.save(folder/'physical_trace.npy',draws)
            path=folder/'fit_receipt.json'
            if path.exists():
                receipt=read(path)
                assert receipt['declaration_sha256']==sha(ROOT/'declaration.json') and receipt['physical_trace_sha256']==digest
                assert receipt['checkpoint_sha256']==sha(folder/'checkpoint.pt')
                model,_=paired.model_module.load(folder/'checkpoint.pt')
            else:
                oldroot=paired.ROOT
                try:
                    paired.ROOT=ROOT
                    model,info=paired.fit(train,'paired_physical',draws,d,fold,folder)
                finally:paired.ROOT=oldroot
                receipt=dict(info,arm=arm,declaration_sha256=sha(ROOT/'declaration.json'),physical_trace_sha256=digest,
                    checkpoint_sha256=sha(folder/'checkpoint.pt'),training_groups=sorted({r['group'] for r in train}),heldout_groups=held,
                    training_sources=sorted({r['source_id'] for r in train}),heldout_sources=sorted({r['source_id'] for r in test}),
                    mining_evidence_sha256=sha(base/'mining_evidence.npz'))
                write(path,receipt)
            pred={r['id']:paired.previous.predict(model,r,'short_transformer') for r in test}
            np.savez_compressed(folder/'heldout_predictions.npz',**pred);predictions[arm].update(pred)
            print('PRESERVATION_SAMPLING_FOLD_COMPLETE',fold,arm,flush=True)
    results={}
    for arm,pred in predictions.items():
        np.savez_compressed(ROOT/(arm+'_all_predictions.npz'),**pred)
        results[arm]={name:paired.previous.summarize(subset,pred) for name,subset in [('all',data),('humming',[r for r in data if r['group'].startswith('HumTrans:')]),('vocadito',[r for r in data if not r['group'].startswith('HumTrans:')])]}
    write(ROOT/'results.json',dict(declaration_sha256=sha(ROOT/'declaration.json'),arms=results,physical_attempts=1152,
        scoreable_pairs=sum('excluded' not in r for r in rows),original_attempts=72,original_controls=53,
        groups_without_positive_pairs=sorted({r['group'] for r in originals}-{r['group'] for r in rows if 'excluded' not in r}),
        raw_predictions_only=True,delivered_audio_qualification=False,production_eligible=False))
    print('PRESERVATION_SAMPLING_COMPLETE',flush=True)


if __name__=='__main__':main()
