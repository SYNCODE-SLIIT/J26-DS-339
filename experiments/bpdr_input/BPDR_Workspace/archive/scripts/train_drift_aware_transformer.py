"""Same real long-deviation pairs, drift-aware input, all-original anchors."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from bpdr_input.experiments.learning.held_voice_training import cv_worker,prepared,fit,anchor_pool
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.held_voice_transformer import HeldConfig
from bpdr_input.modeling.correction_accuracy import summarize

ROOT=ex.WORKSPACE/'results/transformer_usability/2026-10-28_v3'
BASE=ROOT.parent/'2026-10-28_long_v2'

if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        declaration=ex.read(BASE/'declaration.json')
        declaration.update(created_at=datetime.now(timezone.utc).isoformat(),version='drift-aware-held-voice-usability-3',
            model_config=dict(radius=150,stride=6,centering='linear_flanks'),intervention_protocol='long_praat_v2',
            prospective_changes_sha256=ex.sha(ROOT/'prospective_changes.json'),
            original_anchor_rule='All training originals, including source recordings with zero scoreable intervention labels. Original-only fitted singers/source hashes included in checkpoint lineage.',
            data_role='Same real longer-deviation training WAV pairs as v2; broader unchanged-audio supervision')
        ex.write(ROOT/'declaration.json',declaration)
        (ROOT/'train_pairs.json').write_bytes((BASE/'train_pairs.json').read_bytes())
        (ROOT/'evaluation_partition_amendment.json').write_bytes((BASE/'evaluation_partition_amendment.json').read_bytes())
        sources=ex.read(BASE/'data_declaration.json')['source_records']
        ex.write(ROOT/'anchor_sources.json',sources)
    declaration=ex.read(ROOT/'declaration.json')
    hashes=ROOT/'training_source_hashes.json'
    source_files=[Path(__file__),ex.WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py',ex.WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py',ex.WORKSPACE/'source/bpdr_input/experiments/learning/held_voice_training.py']
    if not hashes.exists():
        ex.write(hashes,{str(p):ex.sha(p) for p in source_files})
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for p in source_files:(snapshot/p.name).write_bytes(p.read_bytes())
        inputs=ex.input_hashes(ex.read(ROOT/'train_pairs.json'))
        for source in ex.read(ROOT/'anchor_sources.json'):
            if 'working_audio' not in source:continue
            path=Path(source['working_audio'])
            for file in (path,path.parent/'original_features.npz'):inputs[str(file)]=ex.sha(file)
        ex.write(ROOT/'training_input_hashes.json',inputs)
    for p,h in ex.read(hashes).items():assert ex.sha(p)==h
    results={arm:[] for arm in ('gated','regression')}
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks={pool.submit(cv_worker,(str(ROOT),fold,arm)):(fold,arm) for fold in range(4) for arm in results}
        for task in as_completed(tasks):
            fold,arm=tasks[task];result=task.result();results[arm].extend(result['rows'])
            print('DRIFT_CV_COMPLETE',fold+1,arm,result['summary'],flush=True)
    summaries={arm:summarize(rows) for arm,rows in results.items()}
    qualified=[arm for arm,result in summaries.items() if result['prediction_gate_passed']]
    selected=min(qualified or list(results),key=lambda a:summaries[a]['summary']['raw_mae_cents'] if qualified else summaries[a]['summary']['balanced_error_cents'])
    ex.write(ROOT/'locked_selection.json',dict(arm=selected,prediction_gate_passed=bool(qualified),diagnostic_only=not bool(qualified),results=summaries,default_seed=339,development_used_for_selection=False))
    print('DRIFT_SELECTED',selected,'GATE',bool(qualified),flush=True)
    pairs=prepared(ex.read(ROOT/'train_pairs.json'));anchors=anchor_pool(ex.read(ROOT/'anchor_sources.json'))
    for seed in (339,340,341):
        folder=ROOT/f'final/seed_{seed}'
        if (folder/'fit_receipt.json').exists():continue
        _,_,receipt=fit(pairs,selected,seed,1200,folder,ex.sha(ROOT/'declaration.json'),HeldConfig(gated=selected=='gated',radius=150,stride=6,centering='linear_flanks'),anchors)
        print('DRIFT_FINAL_FIT',seed,receipt['fit_seconds'],flush=True)
    for name in ('training_source_hashes','training_input_hashes'):
        for p,h in ex.read(ROOT/f'{name}.json').items():assert ex.sha(p)==h
    print('DRIFT_MODELS_LOCKED',ROOT,flush=True)
