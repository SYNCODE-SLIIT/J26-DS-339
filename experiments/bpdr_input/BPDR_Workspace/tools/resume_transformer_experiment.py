"""Resume an interrupted declared matrix without changing its fit protocol.

Completed hash-verified fits/evaluations are reused. Independent remaining fits
run in isolated processes, each with the original two Torch threads. The original
runner still performs selection/refits/calibration; its source is not edited.
Stop the original runner before invoking this command. --check is read-only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from bpdr_input.experiments.learning import context_improvement as experiment
from bpdr_input.modeling.context_transformer import load_checkpoint
from bpdr_input.reporting.transformer_accuracy_report import complete_report

_PAIRS=None


class ResumeRoot:
    """Route run allocation without touching UUIDs used by Torch internals."""
    def __init__(self,run,root):
        self.run,self.root=Path(run),Path(root)

    def __truediv__(self,key):
        if key=='latest_experiment.json':
            return self.root/key
        if isinstance(key,str) and len(key)==32 and all(c in '0123456789abcdef' for c in key):
            return self.run
        raise ValueError('Unexpected runner output key: '+str(key))


def fit_complete(folder,declaration_hash):
    receipt=folder/'fit_receipt.json'
    checkpoint=folder/'checkpoint.pt'
    if not receipt.exists() or not checkpoint.exists():
        return False
    r=experiment.read(receipt)
    if r['declaration_sha256']!=declaration_hash or r['checkpoint_sha256']!=experiment.sha(checkpoint):
        raise ValueError('Existing fit receipt/checkpoint integrity mismatch')
    return True


def evaluation_complete(folder):
    path=folder/'accuracy.json'
    if not path.exists() or not (folder/'predictions.npz').exists():
        return False
    record=experiment.read(path)
    if record['predictions_sha256']!=experiment.sha(folder/'predictions.npz'):
        raise ValueError('Existing validation predictions integrity mismatch')
    return True


def initialize_worker(data_run):
    global _PAIRS
    experiment.torch.set_num_threads(2)
    experiment.torch.use_deterministic_algorithms(True)
    _PAIRS=experiment.prepared(experiment.read(Path(data_run)/'train_pairs.json'))


def execute_task(task):
    run,fold_index,seed,arm,updates,declaration_hash=task
    run=Path(run)
    held=set(experiment.FOLDS[fold_index])
    train=[p for p in _PAIRS if experiment.group(p) not in held]
    validation=[p for p in _PAIRS if experiment.group(p) in held]
    experiment.assert_independent({experiment.group(p) for p in train},
        {p['source']['sha256'] for p in train},validation)
    folder=run/'cross_validation'/f'fold_{fold_index+1}'/f'seed_{seed}'/arm
    if fit_complete(folder,declaration_hash):
        model,payload=load_checkpoint(folder/'checkpoint.pt')
    else:
        # A fit is saved only at its fixed endpoint. Do not overwrite a partial
        # checkpoint without a receipt; require explicit recovery inspection.
        if (folder/'checkpoint.pt').exists():
            raise ValueError('Unreceipted checkpoint requires recovery inspection')
        model,payload,_=experiment.fit(train,arm,seed,updates,folder,declaration_hash)
    if not evaluation_complete(folder/'validation'):
        experiment.evaluate(model,payload,validation,folder/'validation',seed=seed)
    return dict(fold=fold_index+1,seed=seed,arm=arm,checkpoint_sha256=experiment.sha(folder/'checkpoint.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve()
    if not run.is_relative_to(experiment.ROOT.resolve()) or not 1<=args.workers<=3:
        raise ValueError('Expected an improvement-run directory and 1–3 workers')
    declaration=experiment.read(run/'declaration.json')
    declaration_hash=experiment.sha(run/'declaration.json')
    for key in ('source_hashes','input_hashes','protected_files'):
        if not all(experiment.sha(p)==digest for p,digest in declaration[key].items()):
            raise ValueError('Declared source/input/protected hashes changed')
    tasks=[]; completed=[]
    for fold in range(len(experiment.FOLDS)):
        for seed in declaration['seeds']:
            for arm in declaration['arms']:
                folder=run/'cross_validation'/f'fold_{fold+1}'/f'seed_{seed}'/arm
                task=(str(run),fold,seed,arm,declaration['updates'],declaration_hash)
                if fit_complete(folder,declaration_hash) and evaluation_complete(folder/'validation'):
                    completed.append((fold,seed,arm))
                else:
                    tasks.append(task)
    print('VERIFIED_COMPLETED',len(completed),'REMAINING',len(tasks),flush=True)
    if args.check:
        return
    amendments=run/'technical_execution'
    amendments.mkdir(exist_ok=True)
    record=dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Resume/orchestrate unchanged independent fits; no objective, split, budget, seed, selection or evaluation changes',
        declaration_sha256=declaration_hash,orchestrator_sha256=experiment.sha(__file__),
        reused_completed=len(completed),remaining=len(tasks),workers=args.workers,torch_threads_per_worker=2,
        timing_note='Use per-fit receipts for fit duration; runner timings during resume include cached reuse/waits')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    experiment.write(amendments/(stamp+'.json'),record)
    original_fit,original_evaluate=experiment.fit,experiment.evaluate
    original_mkdir,original_write=Path.mkdir,experiment.write
    original_root=experiment.ROOT
    def mkdir(path,*a,**kw):
        if path.resolve()==run:
            return None
        return original_mkdir(path,*a,**kw)
    def write(path,value):
        if Path(path).resolve()==run/'declaration.json':
            differences=[k for k in declaration if k!='created_at' and declaration[k]!=value.get(k)]
            if differences:
                raise ValueError('Resume would change declaration: '+str(differences))
            return None
        return original_write(path,value)
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=context,
            initializer=initialize_worker,initargs=(declaration['data_run'],)) as pool:
        pending={(t[1],t[2],t[3]):pool.submit(execute_task,t) for t in tasks}
        def fit(pairs,arm,seed,updates,folder,digest):
            folder=Path(folder)
            if 'cross_validation' not in folder.parts:
                if fit_complete(folder,digest):
                    receipt=experiment.read(folder/'fit_receipt.json')
                    if receipt['arm']!=arm or receipt['seed']!=seed:
                        raise ValueError('Existing final fit belongs to another arm/seed')
                    model,payload=load_checkpoint(folder/'checkpoint.pt')
                    return model,payload,receipt
                return original_fit(pairs,arm,seed,updates,folder,digest)
            fold=int(folder.parent.parent.name.split('_')[1])-1
            future=pending.get((fold,seed,arm))
            if future is not None:
                future.result()
            if not fit_complete(folder,digest):
                raise ValueError('Missing resumed fit')
            model,payload=load_checkpoint(folder/'checkpoint.pt')
            return model,payload,experiment.read(folder/'fit_receipt.json')
        def evaluate(model,payload,pairs,folder,**kwargs):
            folder=Path(folder)
            if evaluation_complete(folder):
                saved=experiment.read(folder/'accuracy.json')
                if saved['rendered_waveforms_evaluated']==bool(kwargs.get('render',False)):
                    return saved
            return original_evaluate(model,payload,pairs,folder,**kwargs)
        experiment.ROOT=ResumeRoot(run,original_root)
        Path.mkdir=mkdir
        experiment.write=write
        experiment.fit=fit
        experiment.evaluate=evaluate
        try:
            result=experiment.run_training(declaration['updates'])
        finally:
            experiment.ROOT=original_root
            Path.mkdir=original_mkdir
            experiment.write=original_write
            experiment.fit=original_fit
            experiment.evaluate=original_evaluate
    complete_report(result)
    for folder in (result/'calibration').glob('seed_*'):
        complete_report(folder)


if __name__=='__main__':
    main()
