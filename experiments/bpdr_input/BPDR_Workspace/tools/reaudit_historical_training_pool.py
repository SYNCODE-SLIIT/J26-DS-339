"""Re-measure historical training variants before allowing fitting."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.repair.interventions import audit_shift
from bpdr_input.repair.model_runtime import runtime_views
from bpdr_input.repair.rule_correction import fine_hz

ROOT=WORKSPACE/'results/transformer_usability/2026-10-29_v17_audio_pool'


def worker(task):
    import torch
    torch.set_num_threads(2)
    index,p=task;folder=ROOT/'acoustic_reaudit';receipt=folder/f'case_{index:03d}.json'
    if receipt.exists():return index,ex.read(receipt)
    a,sr=sf.read(p['path_A'],dtype='float32');c,cr=sf.read(p['path_C'],dtype='float32')
    assert sr==cr==22050 and a.shape==c.shape
    parent=Path(p['path_A']).parent
    with np.load(parent/'original_features.npz') as z:before={k:z[k].copy() for k in z.files}
    with np.load(parent/f"features_{p['cents']:+g}.npz") as z:after={k:z[k].copy() for k in z.files}
    times=before['times_seconds']
    fine_a,fine_c=fine_hz(a,sr,times),fine_hz(c,sr,times)
    _,reliable_a,_,_=runtime_views(before,1200*np.log2(np.maximum(fine_a,1e-6)/440.))
    _,reliable_c,_,_=runtime_views(after,1200*np.log2(np.maximum(fine_c,1e-6)/440.))
    audit,mask,target=audit_shift(before,after,fine_a,fine_c,reliable_a&reliable_c,
                                 p['support_seconds'],p['cents'])
    with np.load(p['target_path']) as z:old_mask=z['mask'].copy();old_target=z['target'].copy()
    removed=int(np.sum(old_mask&~mask));overlap=old_mask&mask
    assert np.array_equal(old_target[overlap],target[overlap])
    base=Path(p['pool_provenance']['manifest']).parent.name=='2026-10-28_v6'
    usable=bool(audit['passed'] and (mask&(target!=0)).any() and (not base or removed==0))
    target_file=None
    if usable:
        target_file=folder/f'targets_{index:03d}_NOT_MODEL_INPUT.npz'
        # Base training/evaluation labels stay unchanged. Extra labels use only
        # the intersection with their freshly verified mask; never add labels.
        known=old_mask if base else old_mask&mask
        np.savez_compressed(target_file,target=old_target.astype(np.float32),mask=known,times=times)
    result=dict(index=index,group=p['group'],source_sha256=p['source']['sha256'],
                original_baseline_case=base,audit=audit,old_known_frames=int(old_mask.sum()),
                removed_known_frames=removed,training_usable=usable,
                target_path=str(target_file) if target_file else None,
                target_sha256=ex.sha(target_file) if target_file else None,
                reason=None if usable else 'ACOUSTIC_REAUDIT_OR_BASE_LABEL_PRESERVATION_FAILED')
    ex.write(receipt,result);return index,result


def main():
    import torch
    torch.set_num_threads(2)
    preflight=ex.read(ROOT/'pool_preflight.json')
    candidates=ex.read(ROOT/'candidate_pairs_NOT_TRAINING_READY.json')
    assert ex.sha(ROOT/'candidate_pairs_NOT_TRAINING_READY.json')==preflight['candidate_manifest_sha256']
    for p,h in preflight['input_hashes'].items():assert ex.sha(p)==h,p
    declaration=ROOT/'acoustic_reaudit_declaration.json'
    files=[Path(__file__).resolve(),WORKSPACE/'source/bpdr_input/repair/interventions.py',
           WORKSPACE/'source/bpdr_input/repair/model_runtime.py',WORKSPACE/'source/bpdr_input/repair/rule_correction.py']
    if not declaration.exists():ex.write(declaration,dict(created_at=datetime.now(timezone.utc).isoformat(),
             preflight_sha256=ex.sha(ROOT/'pool_preflight.json'),candidate_count=len(candidates),
             source_hashes={str(p):ex.sha(p) for p in files},
             protocol='Fresh YIN from actual WAVs, cached original PESTO features, current runtime reliability, unchanged physical intervention audit. Preserve baseline labels exactly; extra labels only intersect fresh known frames. All failures retained. No fitting or evaluation audio access.',
             training_ready=False,research_validation_passed=False))
    frozen=ex.read(declaration)
    for p,h in frozen['source_hashes'].items():assert ex.sha(p)==h,p
    assert frozen['preflight_sha256']==ex.sha(ROOT/'pool_preflight.json')
    (ROOT/'acoustic_reaudit').mkdir(exist_ok=True)
    results={}
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(worker,(i,p)) for i,p in enumerate(candidates)]
        for task in as_completed(tasks):
            i,result=task.result();results[i]=result
            if len(results)%25==0 or len(results)==len(candidates):
                progress=dict(completed=len(results),total=len(candidates),passed=sum(r['training_usable'] for r in results.values()))
                ex.write(ROOT/'acoustic_reaudit_progress.json',progress);print('POOL_REAUDIT',progress,flush=True)
    base=ex.read(ROOT.parent/'2026-10-28_v6/train_pairs.json')
    base_results=[r for r in results.values() if r['original_baseline_case']]
    base_ok=len(base_results)==108 and all(r['training_usable'] for r in base_results)
    fitted=list(base)
    for i,p in enumerate(candidates):
        r=results[i]
        if r['original_baseline_case']:continue
        if r['training_usable']:fitted.append(dict(p,target_path=r['target_path'],fresh_audit_path=str(ROOT/f'acoustic_reaudit/case_{i:03d}.json')))
        else:fitted.append(dict(p,excluded=r['reason'],fresh_audit_path=str(ROOT/f'acoustic_reaudit/case_{i:03d}.json')))
    ready=bool(base_ok and any(r['training_usable'] for r in results.values() if not r['original_baseline_case']))
    path=ROOT/('fit_pairs.json' if ready else 'fit_pairs_NOT_READY.json')
    ex.write(path,fitted)
    for p,h in preflight['input_hashes'].items():assert ex.sha(p)==h,p
    result=dict(training_ready=ready,base108_all_revalidated_without_label_loss=base_ok,
                candidate_variants=len(candidates),variants_revalidated=sum(r['training_usable'] for r in results.values()),
                additional_unique_valid=sum(r['training_usable'] for r in results.values() if not r['original_baseline_case']),
                fit_manifest=str(path),fit_manifest_sha256=ex.sha(path),
                declaration_sha256=ex.sha(declaration),original_singers=10,original_sources=50,
                evaluation_manifest_unchanged=ex.sha(ROOT.parent/'2026-10-28_v6/train_pairs.json'),
                limitation='Fitting readiness only; no model fitted, no new people or independent evaluation. Historical construction failures and repeated manifests retained separately.',
                research_validation_passed=False,demo_activated=False)
    ex.write(ROOT/'acoustic_pool_receipt.json',result);print('POOL_REAUDIT_COMPLETE',result,flush=True)


if __name__=='__main__':main()
