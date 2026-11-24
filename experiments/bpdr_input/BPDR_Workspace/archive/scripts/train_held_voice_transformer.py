"""Train the prospectively declared held-voice Transformer, training folds only."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from bpdr_input.experiments.learning.held_voice_training import ROOT,cv_worker,prepared,fit,evaluate
from bpdr_input.experiments.learning import context_improvement as existing
from bpdr_input.modeling.correction_accuracy import summarize

if __name__=='__main__':
    declaration=existing.read(ROOT/'declaration.json')
    source_files=[Path(__file__),existing.WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py',existing.WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py',existing.WORKSPACE/'source/bpdr_input/experiments/learning/held_voice_training.py']
    hashes=ROOT/'training_source_hashes.json'
    if not hashes.exists():existing.write(hashes,{str(p):existing.sha(p) for p in source_files})
    for p,h in existing.read(hashes).items():assert existing.sha(p)==h
    results={arm:[] for arm in ('gated','regression')}
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks={pool.submit(cv_worker,(fold,arm)):(fold,arm) for fold in range(4) for arm in results}
        for task in as_completed(tasks):
            fold,arm=tasks[task];result=task.result();results[arm].extend(result['rows'])
            print('CV_COMPLETE',fold+1,arm,result['summary'],flush=True)
    summaries={arm:summarize(rows) for arm,rows in results.items()}
    qualified=[arm for arm,result in summaries.items() if result['prediction_gate_passed']]
    selected=min(qualified or list(results),key=lambda a:summaries[a]['summary']['raw_mae_cents'] if qualified else summaries[a]['summary']['balanced_error_cents'])
    existing.write(ROOT/'locked_selection.json',dict(arm=selected,prediction_gate_passed=bool(qualified),diagnostic_only=not bool(qualified),results=summaries,default_seed=339,development_used_for_selection=False))
    print('SELECTED',selected,'GATE',bool(qualified),flush=True)
    pairs=prepared(existing.read(ROOT/'train_pairs.json'))
    for seed in (339,340,341):
        folder=ROOT/'final'/f'seed_{seed}'
        if (folder/'fit_receipt.json').exists():continue
        _,_,receipt=fit(pairs,selected,seed,1200,folder,existing.sha(ROOT/'declaration.json'))
        print('FINAL_FIT',seed,receipt['fit_seconds'],flush=True)
    for p,h in existing.read(hashes).items():assert existing.sha(p)==h
    print('MODELS_LOCKED',ROOT,flush=True)
