"""Train only after the input-only, training-data feature ceiling passes."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from bpdr_input.experiments.learning import step_transformer_training as training
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.correction_accuracy import summarize

ROOT=ex.WORKSPACE/'results/transformer_usability/2026-10-28_v6'

if __name__=='__main__':
    ceiling=ex.read(ROOT/'feature_ceiling/accuracy.json')
    controls=ceiling['original_anchor_controls'];groups=sorted({r['group'] for r in controls})
    original_max=max(float(np.mean([r['raw_original_abs_cents'] for r in controls if r['group']==g])) for g in groups)
    if ceiling['summary']['error_reduction_percent']<25. or ceiling['summary']['response_gain']<.25 or original_max>3.:
        raise ValueError('Training-only feature ceiling failed: no expensive neural fit is unlocked')
    base=ROOT.parent/'2026-10-28_v4'
    if not (ROOT/'declaration.json').exists():
        declaration=ex.read(base/'declaration.json')
        declaration.update(created_at=datetime.now(timezone.utc).isoformat(),version='step-conditioned-transformer-6',model_family='step_context',cv_updates=1000,final_updates=2000,
            intervention_protocol='step_praat_v6',data_declaration_sha256=ex.sha(ROOT/'data_declaration.json'),feature_ceiling_sha256=ex.sha(ROOT/'feature_ceiling/accuracy.json'),
            feature_contract='Three input-only acoustic-step measurements, no target/reference/singer identity or boundary metadata at inference. Genuine two-layer self-attention Transformer calibrates signed correction/activity. Bare-cue versus neural benefit remains a required separate comparison.',
            loss='Original anchor weight4, known-outside2, normalized squared correction, paired response .25, activity BCE .5, positive signed-amplitude auxiliary .25; no proposal-aligned BPDR-methodology superiority claimed',
            practical_scope='Small local step-shaped deviations1.16-1.40s in stable held vocals with measured context on both sides; conditional engineering preview, not natural singing correctness',
            policy_spec=dict(version='held-voice-input-policy-2',region_split_cents=70.),
            selection='Fixed step-conditioned architecture; all four held-out training singer folds; fixed final seeds339/340/341 and default339. Previously inspected development only, reserved final singers remain locked until development passes.',
            additional_source_paths=[str(ex.WORKSPACE/'source/bpdr_input/modeling/step_context.py'),str(ex.WORKSPACE/'source/bpdr_input/modeling/step_transformer.py'),str(ex.WORKSPACE/'source/bpdr_input/experiments/learning/step_transformer_training.py'),str(ex.WORKSPACE/'tools/prepare_recoverable_step_pairs.py')])
        ex.write(ROOT/'declaration.json',declaration)
        for name in ('anchor_sources.json','evaluation_partition_amendment.json','evaluation_vowel_supplement.json'):(ROOT/name).write_bytes((base/name).read_bytes())
    hashes=ROOT/'training_source_hashes.json'
    source_files=[Path(__file__)]+[Path(p) for p in ex.read(ROOT/'declaration.json')['additional_source_paths']]+[ex.WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py']
    if not hashes.exists():
        ex.write(hashes,{str(p):ex.sha(p) for p in source_files});snapshot=ROOT/'executed_training_sources';snapshot.mkdir(exist_ok=True)
        for p in source_files:(snapshot/p.name).write_bytes(p.read_bytes())
        inputs=ex.input_hashes(ex.read(ROOT/'train_pairs.json'))
        for source in ex.read(ROOT/'anchor_sources.json'):
            if 'working_audio' not in source:continue
            path=Path(source['working_audio'])
            for file in (path,path.parent/'original_features.npz'):inputs[str(file)]=ex.sha(file)
        ex.write(ROOT/'training_input_hashes.json',inputs)
    for p,h in ex.read(hashes).items():assert ex.sha(p)==h
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks={pool.submit(training.cv_worker,(str(ROOT),fold)):fold for fold in range(4)}
        for task in as_completed(tasks):
            result=task.result();rows.extend(result['rows']);print('STEP_TRANSFORMER_CV',tasks[task]+1,result['summary'],flush=True)
    result=summarize(rows)
    ex.write(ROOT/'locked_selection.json',dict(arm='step_conditioned',prediction_gate_passed=result['prediction_gate_passed'],diagnostic_only=not result['prediction_gate_passed'],results=dict(step_conditioned=result),default_seed=339,development_used_for_selection=False))
    pairs=training.prepared(ex.read(ROOT/'train_pairs.json'));anchors=training.anchor_pool(ex.read(ROOT/'anchor_sources.json'))
    for seed in (339,340,341):
        folder=ROOT/f'final/seed_{seed}'
        if (folder/'fit_receipt.json').exists():continue
        _,_,receipt=training.fit(pairs,anchors,seed,2000,folder,ex.sha(ROOT/'declaration.json'));print('STEP_TRANSFORMER_REFIT',seed,receipt['fit_seconds'],flush=True)
    for name in ('training_source_hashes','training_input_hashes'):
        for p,h in ex.read(ROOT/f'{name}.json').items():assert ex.sha(p)==h
    print('STEP_TRANSFORMER_LOCKED',ROOT,flush=True)
