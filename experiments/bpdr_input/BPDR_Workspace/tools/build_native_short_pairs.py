"""Pure-data staging from replayed short-audit calibration, preserving all attempts."""
from pathlib import Path
import sys,json
import numpy as np

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'tools'))
from probe_short_phrase_renderer import sha

CAL=WS/'results/transformer_usability/2026-11-02_v44_short_audit_calibration'
OLD=WS/'results/transformer_usability/2026-11-02_v40_short_phrase_data'
ROOT=WS/'results/transformer_usability/2026-11-02_v45_native_short_pairs'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def expected(row,source):
    path=Path(row['output']).parent
    with np.load(source['analysis_A']) as z:before_reliable=z['reliable'].copy()
    with np.load(path/'candidate_context_NOT_LABEL.npz') as z:context={k:z[k].copy() for k in z.files}
    with np.load(path/'measurements.npz') as z:
        times=z['times_seconds'].copy();cc=z['cc'];pvalid=z['voiced_mask'];yin=z['yin']
    with np.load(CAL/source['source_id']/'before_measurements.npz') as z:cb=z['cc'];pb=z['voiced_mask'];yb=z['yin']
    start,end=row['support_seconds'];reliable=before_reliable&context['reliable']
    reliable&=pvalid&pb&np.isfinite(yin)&np.isfinite(yb)&(yin>0)&(yb>0)
    plateau=(times>=start+.02+.07)&(times<=end-.02-.07)&reliable
    plateau&=np.isfinite(cb)&np.isfinite(cc)&(cb>0)&(cc>0)
    zero=((times<start-.07)|(times>end+.07))&reliable
    mask=plateau|zero;target=np.where(plateau,-row['cents'],0.).astype(np.float32)
    return dict(**context,times_seconds=times),dict(target=target,mask=mask,plateau=plateau,known_zero=zero,times=times)


def main():
    ROOT.mkdir(exist_ok=True);cal=read(CAL/'results.json');receipt=read(CAL/'integrity_receipt.json')
    assert receipt['results_sha256']==sha(CAL/'results.json') and receipt['negative_pitch_controls_rejected']==512
    assert cal['negative_false_accepts']==0
    originals=read(OLD/'train_originals.json');by_id={s['source_id']:s for s in originals}
    d=dict(version='native-short-physical-pairs-1',script_sha256=sha(__file__),
        calibration_receipt_sha256=sha(CAL/'integrity_receipt.json'),
        calibration_results_sha256=sha(CAL/'results.json'),original_manifest_sha256=sha(OLD/'train_originals.json'),
        required_context='At least3 reliable input-C observations on both sides within source region after70ms guard; source context minimum150ms each side',
        labels='Nominal negative injected cents only on three-estimator common plateau and retained input reliability. Zero on known unchanged outside; ramps/guards excluded.',
        expected_attempts=336,expected_physical_passes=115,annotations_loaded=False,
        new_waveforms_created=False,nontraining_sources_accessed=False,research_training_eligible=False)
    path=ROOT/'declaration.json'
    if path.exists():assert read(path)==d
    else:write(path,d)
    rows=[];hashes={}
    for source in originals:
        if 'excluded' not in source:
            for key in ['path_A','analysis_A']:hashes[source[key]]=sha(source[key])
    for index,row in enumerate(cal['cases']):
        source=by_id[row['source_id']]
        out=dict(source_id=row['source_id'],group=row['group'],width_seconds=row['width_seconds'],cents=row['cents'],split='train',calibration_index=index,research_training_eligible=False)
        if 'excluded' in row:out['excluded']=row['excluded']
        elif not row['three_tracker']['passed']:out['excluded']='THREE_TRACKER_PHYSICAL_AUDIT_FAILED'
        elif not row['observed_context']['both_sides_at_least_three']:out['excluded']='INSUFFICIENT_RETAINED_TWO_SIDED_CONTEXT'
        else:
            context,labels=expected(row,source)
            if labels['plateau'].sum()<3:out['excluded']='INSUFFICIENT_RETAINED_PLATEAU'
            else:
                folder=ROOT/'prepared'/f'{index:03d}';folder.mkdir(parents=True,exist_ok=True)
                ap=folder/'input_context.npz';tp=folder/'targets_NOT_MODEL_INPUT.npz'
                np.savez_compressed(ap,**context);np.savez_compressed(tp,**labels)
                out.update(path_A=source['path_A'],analysis_A=source['analysis_A'],path_C=row['output'],analysis_C=str(ap),
                    target_path=str(tp),support_seconds=row['support_seconds'],audit=row['three_tracker'])
                for p in [row['output'],str(ap),str(tp)]:hashes[p]=sha(p)
        rows.append(out)
    write(ROOT/'train_pairs.json',rows);write(ROOT/'train_originals.json',originals);write(ROOT/'input_hashes.json',hashes)
    valid=[r for r in rows if 'excluded' not in r]
    write(ROOT/'summary.json',dict(attempted=len(rows),valid=len(valid),scoreable_groups=sorted({r['group'] for r in valid}),original_attempts=21,valid_originals=19,research_training_eligible=False))
    print('NATIVE_SHORT_PAIRS_READY',read(ROOT/'summary.json'),flush=True)


if __name__=='__main__':main()
