"""Train input-localized context without cross-note normalization saturation."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from bpdr_input.experiments.learning.held_voice_training import cv_worker,prepared,fit,anchor_pool
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.held_voice_transformer import HeldConfig
from bpdr_input.modeling.correction_accuracy import summarize

ROOT=ex.WORKSPACE/'results/transformer_usability/2026-10-28_v4'
BASE=ROOT.parent/'2026-10-28_v3'

if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        declaration=ex.read(BASE/'declaration.json')
        declaration.update(created_at=datetime.now(timezone.utc).isoformat(),version='pitch-neighborhood-held-voice-4',cv_updates=3200,final_updates=4800,
            model_config=dict(radius=150,stride=6,centering='linear_flanks',scope_same_run=True,flank_samples=2),
            prospective_changes_sha256=ex.sha(ROOT/'prospective_changes.json'),
            original_anchor_rule='All reliably voiced training originals; unavailable/unvoiced anchors excluded and kept in source inventory',
            selection='Fixed gated architecture; report all four singer-held-out folds. Default seed339 fixed. No development/final-based family or seed selection.',
            engineering_change='Input-only context localization avoids neighboring-note saturation; two outer flank observations preserve shorter local steps. Longer declared optimization budget; not an isolated feature-only ablation.')
        ex.write(ROOT/'declaration.json',declaration)
        for name in ('train_pairs.json','anchor_sources.json','evaluation_partition_amendment.json'):(ROOT/name).write_bytes((BASE/name).read_bytes())
    declaration=ex.read(ROOT/'declaration.json')
    hashes=ROOT/'training_source_hashes.json'
    source_files=[Path(__file__),ex.WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py',ex.WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py',ex.WORKSPACE/'source/bpdr_input/experiments/learning/held_voice_training.py']
    if not hashes.exists():
        ex.write(hashes,{str(p):ex.sha(p) for p in source_files})
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for p in source_files:(snapshot/p.name).write_bytes(p.read_bytes())
        (ROOT/'training_input_hashes.json').write_bytes((BASE/'training_input_hashes.json').read_bytes())
    for p,h in ex.read(hashes).items():assert ex.sha(p)==h
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks={pool.submit(cv_worker,(str(ROOT),fold,'gated')):fold for fold in range(4)}
        for task in as_completed(tasks):
            fold=tasks[task];result=task.result();rows.extend(result['rows'])
            print('NEIGHBORHOOD_CV_COMPLETE',fold+1,result['summary'],flush=True)
    summary=summarize(rows)
    ex.write(ROOT/'locked_selection.json',dict(arm='gated',prediction_gate_passed=summary['prediction_gate_passed'],diagnostic_only=not summary['prediction_gate_passed'],results=dict(gated=summary),default_seed=339,development_used_for_selection=False))
    pairs=prepared(ex.read(ROOT/'train_pairs.json'));anchors=anchor_pool(ex.read(ROOT/'anchor_sources.json'))
    for seed in (339,340,341):
        folder=ROOT/f'final/seed_{seed}'
        if (folder/'fit_receipt.json').exists():continue
        _,_,receipt=fit(pairs,'gated',seed,4800,folder,ex.sha(ROOT/'declaration.json'),HeldConfig(gated=True,**declaration['model_config']),anchors)
        print('NEIGHBORHOOD_FINAL_FIT',seed,receipt['fit_seconds'],flush=True)
    for name in ('training_source_hashes','training_input_hashes'):
        for p,h in ex.read(ROOT/f'{name}.json').items():assert ex.sha(p)==h
    print('NEIGHBORHOOD_MODELS_LOCKED',ROOT,flush=True)
