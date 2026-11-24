"""Coverage-only original sampler contrast; the verified upper-tail loss stays fixed."""
from pathlib import Path
import argparse,hashlib
import numpy as np
import torch
import compare_tail_preservation as previous
from bpdr_input.experiments.learning import original_coverage as coverage

WS=previous.WS
DATA=previous.DATA
ROOT=WS/'results/transformer_usability/2026-11-04_v70_original_coverage'
ARMS=['frozen_tail','coverage_complete_originals']
paired=previous.paired
model_module=previous.model_module
read,write,norm,sha=previous.read,previous.write,previous.norm,previous.sha


def declare():
    parent,rows,originals=previous.declare();proofs={}
    for root in [previous.ROOT,ROOT.parent/'2026-11-04_v69_tail_delivery']:
        proof=read(root/'integrity_receipt.json');assert proof['results_sha256']==sha(root/'results.json');proofs[str(root/'integrity_receipt.json')]=sha(root/'integrity_receipt.json')
    checkpoints={}
    for fold in range(1,5):
        path=previous.ROOT/f'fold_{fold}/upper_tail_preservation/checkpoint.pt';assert read(path.parent/'fit_receipt.json')['checkpoint_sha256']==sha(path);checkpoints[str(path)]=sha(path)
    base={k:parent[k] for k in ['folds','seed','updates','batch_size','forward_rows_per_update','config','optimizer','virtual_widths_seconds','loss']}
    paths=[Path(__file__),Path(previous.__file__),Path(coverage.__file__),Path(model_module.__file__),Path(previous.tail.__file__),Path(paired.__file__),WS/'tests/test_original_coverage.py']
    d=norm(dict(base,version='complete-original-query-contrast-1',arms=ARMS,new_fits=4,virtual_updates=0,
        source_hashes={str(p.resolve()):sha(p) for p in paths},parent_integrities=proofs,parent_checkpoints=checkpoints,
        parent_declaration_sha256=sha(previous.ROOT/'declaration.json'),data_integrity_sha256=sha(DATA/'integrity_receipt.json'),
        changed_variable='General-original queries only:shuffle every eligible original reliable zero-target frame without replacement;repeat only after a full pass. Positive/pairedA/C-zero draws stay exactly identical.',
        frequency_change='Original record/singer frequencies change from group-balanced replacement to frame-proportional full coverage;all other sampler classes fixed.',
        query_seed='339+fold,not adaptive to predictions',fixed='Same V68 upper-tail loss,model,data,targets,initial339,1200 updates,48 forward rows,optimizer,controller,renderer,audit.',
        stopping='Exactly4 new fits,no seed/epoch/threshold sweep. Refuse if full original pool exceeds14400 original draws. Query coverage is not optimization success or usable accuracy.',
        physical_attempts=1152,scoreable_pairs=445,original_attempts=72,original_controls=53,
        gate='Unchanged raw/original/known-zero/per-group coverage and reduction gates. Keep unassessed/excluded attempts;no zero-command-collapse adoption.',
        scope='Training-source grouped engineering only;no development/reserved audio,guide notes,natural intention or full BPDR methodological superiority.',
        nontraining_sources_accessed=False,production_eligible=False,research_validation_passed=False))
    ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/'declaration.json'
    if path.exists():assert read(path)==d
    else:write(path,d)
    return d,rows,originals


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--declare-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);d,rows,originals=declare()
    if args.declare_only:print('ORIGINAL_COVERAGE_DECLARED',flush=True);return
    data=paired.records(rows,originals);predictions={arm:{} for arm in ARMS}
    with np.load(previous.ROOT/'upper_tail_preservation_all_predictions.npz') as z:baseline={k:z[k].copy() for k in z.files}
    for fold,groups in enumerate(d['folds'],1):
        train=[r for r in data if r['group'] not in groups];test=[r for r in data if r['group'] in groups]
        assert not {r['source_id'] for r in train}&{r['source_id'] for r in test}
        parent_path=previous.ROOT/f'fold_{fold}/upper_tail_preservation/physical_trace.npy';parent_trace=np.load(parent_path)
        assert hashlib.sha256(parent_trace.tobytes()).hexdigest()==read(parent_path.parent/'fit_receipt.json')['physical_trace_sha256']
        trace,stats=coverage.covered_original_trace(train,parent_trace,d['seed']+fold);digest=hashlib.sha256(trace.tobytes()).hexdigest()
        folder=ROOT/f'fold_{fold}/coverage_complete_originals';folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'physical_trace.npy',trace);write(folder/'coverage_stats.json',dict(stats,training_record_ids=[r['id'] for r in train]))
        path=folder/'fit_receipt.json'
        if path.exists():
            receipt=read(path);assert receipt['declaration_sha256']==sha(ROOT/'declaration.json') and receipt['physical_trace_sha256']==digest
            assert receipt['checkpoint_sha256']==sha(folder/'checkpoint.pt');model,_=model_module.load(folder/'checkpoint.pt')
        else:
            prior_root=previous.ROOT
            try:
                previous.ROOT=ROOT
                model,info=previous.fit(train,'coverage_complete_originals',trace,d,fold,folder)
            finally:previous.ROOT=prior_root
            receipt=dict(info,declaration_sha256=sha(ROOT/'declaration.json'),physical_trace_sha256=digest,checkpoint_sha256=sha(folder/'checkpoint.pt'),
                coverage_stats_sha256=sha(folder/'coverage_stats.json'),training_groups=sorted({r['group'] for r in train}),training_sources=sorted({r['source_id'] for r in train}),
                heldout_groups=groups,heldout_sources=sorted({r['source_id'] for r in test}))
            write(path,receipt)
        pred={r['id']:paired.previous.predict(model,r,'short_transformer') for r in test};np.savez_compressed(folder/'heldout_predictions.npz',**pred)
        predictions['coverage_complete_originals'].update(pred);predictions['frozen_tail'].update({r['id']:baseline[r['id']] for r in test})
        print('ORIGINAL_COVERAGE_FOLD_COMPLETE',fold,stats['eligible_original_frames'],stats['unique_original_frames_queried'],flush=True)
    results={}
    for arm,pred in predictions.items():
        np.savez_compressed(ROOT/(arm+'_all_predictions.npz'),**pred)
        results[arm]={name:paired.previous.summarize(subset,pred) for name,subset in [('all',data),('humming',[r for r in data if r['group'].startswith('HumTrans:')]),('vocadito',[r for r in data if not r['group'].startswith('HumTrans:')])]}
    write(ROOT/'results.json',dict(declaration_sha256=sha(ROOT/'declaration.json'),arms=results,physical_attempts=1152,scoreable_pairs=445,original_attempts=72,original_controls=53,
        groups_without_positive_pairs=sorted({r['group'] for r in originals}-{r['group'] for r in rows if 'excluded' not in r}),
        raw_predictions_only=True,delivered_audio_qualification=False,production_eligible=False))
    print('ORIGINAL_COVERAGE_COMPLETE',{a:s['all']['positive_singer_macro_mae_cents'] for a,s in results.items()},flush=True)


if __name__=='__main__':main()
