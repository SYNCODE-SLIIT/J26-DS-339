"""Independent source, decoded-pair, audit and label checks before short-phrase fitting."""
from pathlib import Path
from math import gcd
import json,sys
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
from probe_short_phrase_renderer import sha
from acquire_fresh_confirmation import check_lock
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.repair.rule_correction import fine_hz,segment_notes,RulePolicy
from bpdr_input.repair.centered_precision_audit import coarse_features
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.interventions import AuditPolicy

ROOT=WS/'results/transformer_usability/2026-11-02_v40_short_phrase_data'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def norm(x):return json.loads(json.dumps(x,allow_nan=False))


def check_arrays(path,analysis):
    expected=dict(raw_views=np.stack(analysis['views']),reliable=analysis['reliable'],fine=analysis['fine'],**analysis['features'])
    with np.load(path) as z:
        assert set(z.files)==set(expected)
        for key,value in expected.items():np.testing.assert_array_equal(z[key],value,err_msg=str(path)+' '+key)
    return len(expected)


def main():
    torch.set_num_threads(2);check_lock();d=read(ROOT/'preparation_declaration.json')
    assert d['script_sha256']==sha(WS/'tools/prepare_short_phrase_training.py')
    assert d['source_reservation_sha256']==sha(ROOT/'source_reservation.json')
    for p,h in read(ROOT/'input_hashes.json').items():assert sha(p)==h,p
    rows=read(ROOT/'train_pairs.json');originals=read(ROOT/'train_originals.json')
    assert len(rows)==336 and len(originals)==21
    policy=AuditPolicy(**d['audit']);cache={};array_count=0;decoded_audits=0;labels=0;geometry=0;invalid=[]
    recreated=ROOT/'verification_recreated';recreated.mkdir(exist_ok=True)
    for original in originals:
        s=original['source'];assert s['split']=='train' and sha(s['audio_path'])==s['audio_sha256']
        raw,sr=sf.read(s['audio_path'],dtype='float32',always_2d=True)
        channel=int(np.argmax(np.mean(raw.astype(float)**2,axis=0)))
        y=raw[:min(len(raw),10*sr),channel];div=gcd(sr,22050)
        if sr!=22050:y=resample_poly(y,22050//div,sr//div).astype(np.float32)
        valid=len(y)>=5*22050 and np.isfinite(y).all() and np.max(abs(y))<1
        if not valid:
            assert original['excluded']=='INVALID_SOURCE_DURATION_OR_SAMPLES'
            invalid.append(dict(source_id=s['source_id'],duration_seconds=len(y)/22050,
                                resampled_peak=float(np.max(abs(y))),finite=bool(np.isfinite(y).all())))
            continue
        p=recreated/(s['source_id']+'.wav');sf.write(p,y,22050,subtype='PCM_24')
        decoded,_=sf.read(p,dtype='float32');a,srate=sf.read(original['path_A'],dtype='float32')
        assert srate==22050 and np.array_equal(a,decoded)
        analysis=recovery.measured(a);array_count+=check_arrays(original['analysis_A'],analysis)
        regions=segment_notes(analysis['views'][0][:,64],analysis['reliable'],RulePolicy(note_split_cents=70.))
        region=max(regions,key=lambda r:r[1]-r[0]) if regions else None
        assert norm(region)==original['longest_region_frames']
        times=analysis['features']['times_seconds']
        cache[s['source_id']]=(a,analysis,region,coarse_features(a,22050,WS/'checkpoints'),fine_hz(a,22050,times))
    for row in rows:
        if row['source_id'] not in cache:
            assert row['excluded']=='INVALID_SOURCE_DURATION_OR_SAMPLES';continue
        audio,a,region,bf,yb=cache[row['source_id']];width=row['width_seconds']
        eligible=region is not None and (region[1]-region[0])*.02>=width+.14-1e-12
        if not eligible:
            assert row['excluded']=='NO_SINGLE_RELIABLE_REGION_WITH_70MS_CONTEXT';geometry+=1;continue
        candidate,sr=sf.read(row['path_C'],dtype='float32');assert sr==22050 and candidate.shape==audio.shape
        assert np.isfinite(candidate).all() and abs(candidate).max()<1
        times=a['features']['times_seconds'];start,end=row['support_seconds']
        center=float(np.mean(region)*.02);assert row['support_seconds']==[center-width/2,center+width/2]
        st=np.arange(len(audio))/22050;outside_samples=(st<=start)|(st>=end)
        assert np.array_equal(audio[outside_samples],candidate[outside_samples])
        cf=coarse_features(candidate,22050,WS/'checkpoints');yc=fine_hz(candidate,22050,times)
        curve=np.where((times>=start)&(times<end),row['cents'],0.)
        audit=audit_curve(bf,cf,yb,yc,a['reliable'],[(start,end)],curve,policy)
        assert norm(audit)==row['audit']==read(row['audit_path']);decoded_audits+=1
        if not audit['passed']:
            assert row['excluded']=='PHYSICAL_INTERVENTION_AUDIT_FAILED';continue
        assert 'excluded' not in row
        c=recovery.measured(candidate);array_count+=check_arrays(row['analysis_C'],c)
        with np.load(row['target_path']) as z:
            assert np.array_equal(z['times'],times)
            common=a['reliable']&c['reliable']&bf['voiced_mask']&cf['voiced_mask']&np.isfinite(yb)&np.isfinite(yc)
            plateau=(times>=start+.09)&(times<=end-.09)&common
            outside=((times<start-.07)|(times>end+.07))&common
            np.testing.assert_array_equal(z['plateau'],plateau)
            np.testing.assert_array_equal(z['mask'],plateau|outside)
            np.testing.assert_array_equal(z['target'],np.where(plateau,-row['cents'],0.).astype(np.float32))
            assert plateau.sum()>=10
            assert np.all(~plateau|((times>=start+policy.ramp_seconds+policy.analysis_guard_seconds)&(times<=end-policy.ramp_seconds-policy.analysis_guard_seconds)))
        labels+=1
    result=dict(source_records_checked=21,decoded_sources_recreated=len(cache),invalid_source_details=invalid,
        original_and_corrupted_feature_arrays_reinferred=array_count,decoded_audits_replayed=decoded_audits,
        supervised_label_files_verified=labels,geometry_exclusions_replayed=geometry,
        all336_attempts_retained=True,training_singers_only=True,annotation_arrays_loaded=False,
        verifier_source_sha256=sha(__file__),preparation_declaration_sha256=sha(ROOT/'preparation_declaration.json'),
        train_pairs_sha256=sha(ROOT/'train_pairs.json'),train_originals_sha256=sha(ROOT/'train_originals.json'),
        input_hashes_sha256=sha(ROOT/'input_hashes.json'),research_training_eligible=False,production_eligible=False)
    (ROOT/'integrity_receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('SHORT_TRAINING_DATA_REPLAY_PASS',labels,decoded_audits,array_count,flush=True)


if __name__=='__main__':main()
