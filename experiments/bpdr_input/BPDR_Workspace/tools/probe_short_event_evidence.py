"""Prospective input-evidence ceiling on all old short-pair attempts and originals."""
from pathlib import Path
from collections import defaultdict
import sys,json
import numpy as np

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
from probe_short_phrase_renderer import sha
from acquire_fresh_confirmation import check_lock
from bpdr_input.modeling import short_event_evidence as feature
from bpdr_input.modeling import step_context as old

DATA=WS/'results/transformer_usability/2026-11-02_v40_short_phrase_data'
ROOT=WS/'results/transformer_usability/2026-11-02_v43_short_event_evidence'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    check_lock();ROOT.mkdir(exist_ok=True)
    rows=read(DATA/'train_pairs.json');originals=read(DATA/'train_originals.json')
    paths=[Path(__file__),Path(feature.__file__),WS/'tests/test_short_event_evidence.py']
    d=dict(version='short-event-evidence-feasibility-1',source_hashes={str(p.resolve()):sha(p) for p in paths},
        data_integrity_sha256=sha(DATA/'integrity_receipt.json'),
        scope='All336 V40 attempts retained;63 scored inputs (19 originals,44 physical C). Input-only cue, not a correction-model or delivered-WAV qualification.',
        inference_input='Only raw pitch64 and reliable-mask66; no labels, reference A, injected supports, IDs, notes, keys or target arrays.',
        planned_comparison='Old long-step evidence vs new anchored short pulse. Same masked raw-gate statistics; report every missing cue and original false event.',
        new_fits=0,production_eligible=False,research_validation_passed=False)
    path=ROOT/'declaration.json'
    if path.exists():assert read(path)==d
    else:write(path,d)
    records=[]
    for s in originals:
        if 'excluded' not in s:records.append(dict(id='A_'+s['source_id'],group=s['group'],kind='original',analysis=s['analysis_A']))
    for i,r in enumerate(rows):
        if 'excluded' not in r:records.append(dict(id='C_'+str(i),group=r['group'],kind='corrupted',analysis=r['analysis_C'],target_path=r['target_path']))
    outcomes=[];measurements=defaultdict(list)
    for record in records:
        with np.load(record['analysis']) as z:raw=z['raw_views'][0].copy();reliable=z['reliable'].copy()
        new,details=feature.evidence(raw,details=True);legacy=old.evidence(raw)
        np.savez_compressed(ROOT/(record['id']+'.npz'),new=new,legacy=legacy)
        result=dict(id=record['id'],group=record['group'],kind=record['kind'],new_fits=details['fits'])
        for name,values in [('legacy',legacy),('short',new)]:
            prediction=values[:,0]*30
            if record['kind']=='original':
                stats=dict(original_mae_cents=float(np.mean(abs(prediction[reliable]))) if reliable.any() else None,
                           active_frames=int(np.sum(abs(prediction[reliable])>=10)))
            else:
                with np.load(record['target_path']) as z:target=z['target'];positive=z['mask']&(target!=0);zero=z['mask']&(target==0)
                stats=dict(positive_mae_cents=float(np.mean(abs(prediction[positive]-target[positive]))),
                    signed_response=float(np.mean(prediction[positive]*target[positive])/np.mean(target[positive]**2)),
                    positive_fraction_active=float(np.mean(abs(prediction[positive])>=10)),
                    known_zero_mae_cents=float(np.mean(abs(prediction[zero]))))
            result[name]=stats;measurements[name].append(dict(group=record['group'],kind=record['kind'],**stats))
        outcomes.append(result)
        print('SHORT_EVIDENCE_INPUT',record['id'],'fits',len(details['fits']),flush=True)
    summaries={}
    for name,values in measurements.items():
        groups=[]
        for group in sorted({v['group'] for v in values}):
            positive=[v for v in values if v['group']==group and v['kind']=='corrupted']
            original=[v for v in values if v['group']==group and v['kind']=='original' and v['original_mae_cents'] is not None]
            groups.append(dict(group=group,scoreable_pairs=len(positive),
                positive_mae_cents=float(np.mean([v['positive_mae_cents'] for v in positive])) if positive else None,
                signed_response=float(np.mean([v['signed_response'] for v in positive])) if positive else None,
                original_mae_cents=float(np.mean([v['original_mae_cents'] for v in original])) if original else None,
                original_active_frames=sum(v['active_frames'] for v in original)))
        scored=[g for g in groups if g['positive_mae_cents'] is not None]
        summaries[name]=dict(groups=groups,singer_macro_positive_mae_cents=float(np.mean([g['positive_mae_cents'] for g in scored])),
            original_files_with_event=sum(v['kind']=='original' and v.get('active_frames',0)>0 for v in values),
            raw_gate_passed=all(g['positive_mae_cents']<25 and g['signed_response']>=.25 for g in scored) and
                            all(g['original_mae_cents']<=3 for g in groups if g['original_mae_cents'] is not None))
    write(ROOT/'results.json',dict(declaration_sha256=sha(path),scored_inputs=len(records),
        physical_attempts=336,scoreable_pairs=44,construction_excluded=292,original_controls=19,
        summaries=summaries,inputs=outcomes,new_model_fits=0,production_eligible=False))
    print('SHORT_EVIDENCE_COMPLETE',{k:(v['singer_macro_positive_mae_cents'],v['original_files_with_event'],v['raw_gate_passed']) for k,v in summaries.items()},flush=True)


if __name__=='__main__':main()
