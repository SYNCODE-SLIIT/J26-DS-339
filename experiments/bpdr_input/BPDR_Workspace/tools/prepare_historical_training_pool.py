"""Inventory approved historical audio variants; no fitting or new audio access."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import soundfile as sf

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.repair.interventions import AuditPolicy
from dataclasses import asdict

ROOT=WORKSPACE/'results/transformer_usability/2026-10-29_v17_audio_pool'
RUNS=['2026-10-28_v6','2026-10-28_v1','2026-10-28_long_v2',
      '2026-10-28_v3','2026-10-28_v4','2026-10-28_v5']


def main():
    ROOT.mkdir(exist_ok=True)
    base=ex.read(ROOT.parent/'2026-10-28_v6/train_pairs.json')
    allowed={p['source']['sha256']:p['source']['singer_id'] for p in base}
    manifests=[ROOT.parent/run/'train_pairs.json' for run in RUNS]
    if not (ROOT/'pool_declaration.json').exists():
        ex.write(ROOT/'pool_declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
                 role='Training-source historical variant preflight; NOT a qualified fitting manifest',
                 manifests={str(p):ex.sha(p) for p in manifests},approved_sources=allowed,
                 selection='All six named historical training manifests. No model outcome filtering. Deduplicate identical source/clean/shifted waveform and labels; retain every source attempt and construction exclusion.',
                 evaluation='V6 evaluation200 attempts remains unchanged. Every derived variant inherits original singer/source fold. IDs5/6/8/9 and reserved confirmation sources forbidden.',
                 source_hashes={str(Path(__file__).resolve()):ex.sha(Path(__file__))},
                 next_gate='Fresh acoustic label re-audit required before fitting. Candidate manifest must not be consumed directly by a trainer.',
                 training_ready=False,demo_activated=False,research_validation_passed=False))
    d=ex.read(ROOT/'pool_declaration.json')
    for label in ('manifests','source_hashes'):
        for p,h in d[label].items():assert ex.sha(p)==h,p
    if (ROOT/'pool_preflight.json').exists():
        old=ex.read(ROOT/'pool_preflight.json')
        for p,h in old['input_hashes'].items():assert ex.sha(p)==h,p
        assert ex.sha(ROOT/'candidate_pairs_NOT_TRAINING_READY.json')==old['candidate_manifest_sha256']
        print('POOL_PREFLIGHT_EXISTS',old['summary']);return
    expected_policy=json.loads(json.dumps(asdict(AuditPolicy())))
    seen={};unique=[];attempts=[];counts=Counter();hashes={};audio_cache={}
    def checksum(path):
        path=str(Path(path).resolve())
        if path not in hashes:hashes[path]=ex.sha(path)
        return hashes[path]
    def audio(path):
        if path not in audio_cache:
            wave,sr=sf.read(path,dtype='float32')
            assert sr==22050 and wave.ndim==1 and np.isfinite(wave).all()
            audio_cache[path]=(wave,sr,hashlib.sha256(wave.tobytes()).hexdigest())
        return audio_cache[path]
    for manifest in manifests:
        for i,p in enumerate(ex.read(manifest)):
            source=p['source'];group=source['singer_id'];original=source['sha256']
            if original not in allowed or group!=allowed[original] or p.get('group',group)!=group:
                raise ValueError('Historical row outside approved training source/singer set')
            assert checksum(source['path'])==original
            provenance=dict(manifest=str(manifest),index=i,group=group,source_sha256=original)
            counts['attempts']+=1
            if 'excluded' in p:
                counts['construction_excluded']+=1
                attempts.append(dict(**provenance,disposition='construction_excluded',reason=p['excluded']));continue
            assert p['audit']['passed'] and p['audit']['policy']==expected_policy
            a,sr,clean_hash=audio(p['path_A']);c,cr,shifted_hash=audio(p['path_C'])
            assert a.shape==c.shape and sr==cr
            outside=np.ones(len(a),bool)
            lo,hi=p['support_seconds'];outside[int(np.ceil(lo*sr)):int(np.ceil(hi*sr))]=False
            assert np.array_equal(a[outside],c[outside]),'Historical intervention changed outside samples'
            with np.load(p['target_path']) as z:
                target=z['target'];mask=z['mask'];times=z['times']
                assert mask.dtype==bool and target.shape==mask.shape==times.shape
                assert np.isfinite(target).all() and mask.any() and np.any(mask & (target!=0))
                assert np.all(np.isin(target,[0.,-p['cents']]))
                label=hashlib.sha256(target.astype(np.float32).tobytes()+mask.tobytes()+times.tobytes()).hexdigest()
            for name in ('path_A','path_C','target_path'):checksum(p[name])
            folder=Path(p['path_A']).parent
            for path in (folder/'original_features.npz',folder/f"features_{p['cents']:+g}.npz"):
                checksum(path)
                with np.load(path) as z:np.testing.assert_array_equal(z['times_seconds'],times)
            key=(original,clean_hash,shifted_hash)
            if key in seen:
                previous,old_label=seen[key]
                if old_label!=label:raise ValueError('Identical historical audio has conflicting supervision')
                counts['duplicate_valid']+=1
                attempts.append(dict(**provenance,disposition='duplicate_valid',candidate_index=previous));continue
            index=len(unique);seen[key]=(index,label)
            unique.append(dict(**p,pool_provenance=provenance))
            counts['unique_valid']+=1
            attempts.append(dict(**provenance,disposition='unique_valid',candidate_index=index))
    output=ROOT/'candidate_pairs_NOT_TRAINING_READY.json';ex.write(output,unique)
    ex.write(ROOT/'all_source_attempts.json',attempts)
    summary=dict(counts,original_singers=len(set(allowed.values())),original_sources=len(allowed),
                 baseline_scoreable=108,additional_unique_variants=len(unique)-108)
    ex.write(ROOT/'pool_preflight.json',dict(summary=summary,input_hashes=hashes,
             declaration_sha256=ex.sha(ROOT/'pool_declaration.json'),candidate_manifest_sha256=ex.sha(output),
             attempts_sha256=ex.sha(ROOT/'all_source_attempts.json'),training_ready=False,
             limitation='Historical audit receipts and file/label/sample consistency checked. Fresh acoustic re-audit still required. Variants are not independent singers or fresh evaluation.',
             demo_activated=False,research_validation_passed=False))
    print('POOL_PREFLIGHT_COMPLETE',summary,flush=True)


if __name__=='__main__':main()
