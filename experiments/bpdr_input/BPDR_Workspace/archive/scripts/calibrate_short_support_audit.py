"""Prospective three-tracker short-audit calibration; never trains or activates a model."""
from pathlib import Path
from dataclasses import asdict,replace
from collections import Counter
import sys,json
import numpy as np
import soundfile as sf
import torch,parselmouth

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
from probe_short_phrase_renderer import sha
from acquire_fresh_confirmation import check_lock
from bpdr_input.repair import experimental_short_audit as short
from bpdr_input.repair.framewise_renderer import RendererPolicy,render_candidate
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.centered_precision_audit import coarse_features
from bpdr_input.repair.rule_correction import fine_hz
from bpdr_input.repair.learned_runtime import audit_delivered_wav
from bpdr_input.experiments.learning import model_recovery as recovery

DATA=WS/'results/transformer_usability/2026-11-02_v40_short_phrase_data'
ROOT=WS/'results/transformer_usability/2026-11-02_v44_short_audit_calibration'
RENDER=replace(RendererPolicy(),version='short-three-tracker-calibration-render-1',ramp_seconds=.02,min_support_seconds=.24)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def norm(x):return json.loads(json.dumps(x,allow_nan=False))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def declare():
    check_lock();ROOT.mkdir(exist_ok=True)
    paths=[Path(__file__),Path(short.__file__),WS/'tests/test_experimental_short_audit.py']
    d=norm(dict(version='short-native-frame-three-tracker-calibration-1',
        source_hashes={str(p.resolve()):sha(p) for p in paths},
        original_manifest_sha256=sha(DATA/'train_originals.json'),data_integrity_sha256=sha(DATA/'integrity_receipt.json'),
        originals=read(DATA/'train_originals.json'),planned_attempts=336,
        widths_seconds=[.24,.28,.36,.50],shifts_cents=[-30.,-20.,20.,30.],
        required_original_context_seconds=.15,renderer=asdict(RENDER),audit=asdict(short.POLICY),
        comparison='Same native20ms grid, same error/direction/coverage/outside limits. Compare10-frame two-tracker,3-frame two-tracker and3-frame three-tracker. This is a changed measurement protocol, not a gain under unchanged old rules.',
        negative_controls=['unchanged waveform','opposite requested sign','requested magnitude one-third of realized intervention','requested support displaced+1s','single outside-support PCM sample tamper'],
        cross_correlation=dict(parselmouth_version=parselmouth.__version__,praat_version=parselmouth.PRAAT_VERSION,
            step_seconds=.01,floor_hz=50.,ceiling_hz=1500.,very_accurate=True,
            alignment='Nearest native10ms estimate to20ms audit grid, max5ms+1e-12s; no unvoiced interpolation'),
        source_reference='https://raw.githubusercontent.com/praat/praat/v6.1.38/fon/Sound_to_Pitch.cpp',
        expected_negatives='All five controls must be rejected; keep failures and unassessed controls visible. Zero false accepts on this finite set does not prove broad reliability.',
        caveats='Three overlapping observations are not independent trials or a confidence interval. CC shares Praat library with renderer and has global path finding; no fully independent ground-truth claim. No musical-intention validation.',
        annotations_loaded=False,nontraining_sources_accessed=False,new_model_fits=0,production_eligible=False))
    p=ROOT/'declaration.json'
    if p.exists():assert read(p)==d
    else:write(p,d)
    return d


def main():
    torch.set_num_threads(2);d=declare();rows=[]
    baseline=replace(short.POLICY,version='retained-ten-frame-calibration-baseline',min_audit_frames=10)
    for source in d['originals']:
        sid=source['source_id'];folder=ROOT/sid;folder.mkdir(exist_ok=True)
        source_rows=[]
        if 'excluded' not in source:
            audio,sr=sf.read(source['path_A'],dtype='float32');assert sr==22050
            with np.load(source['analysis_A']) as z:times=z['times_seconds'];reliable=z['reliable']
            bf=coarse_features(audio,sr,WS/'checkpoints');yb=fine_hz(audio,sr,times)
            cb=short.cross_correlation_pitch(audio,sr,times);region=source['longest_region_frames']
            np.savez_compressed(folder/'before_measurements.npz',**bf,yin=yb,cc=cb,reliable=reliable)
        for width in d['widths_seconds']:
            for cents in d['shifts_cents']:
                base=dict(source_id=sid,group=source['group'],width_seconds=width,cents=cents)
                if 'excluded' in source:
                    source_rows.append(dict(**base,excluded='SOURCE_PREPROCESSING_EXCLUDED'));continue
                if region is None or (region[1]-region[0])*.02<width+.30-1e-12:
                    source_rows.append(dict(**base,excluded='NO_SINGLE_RELIABLE_REGION_WITH_150MS_CONTEXT'));continue
                center=float(np.mean(region)*.02);support=(center-width/2,center+width/2)
                curve=np.where((times>=support[0])&(times<support[1]),cents,0.)
                case=folder/f'{width:.2f}_{cents:+g}';case.mkdir(exist_ok=True)
                output,_=render_candidate(audio,sr,times,curve,[support],RENDER)
                path=case/'intervention.wav';sf.write(path,output,sr,subtype='PCM_24');decoded,rate=sf.read(path,dtype='float32')
                assert rate==sr and decoded.shape==audio.shape and np.isfinite(decoded).all() and abs(decoded).max()<1
                st=np.arange(len(audio))/sr;outside=(st<=support[0])|(st>=support[1])
                assert np.array_equal(audio[outside],decoded[outside])
                af=coarse_features(decoded,sr,WS/'checkpoints');ya=fine_hz(decoded,sr,times)
                ca=short.cross_correlation_pitch(decoded,sr,times)
                ten=audit_curve(bf,af,yb,ya,reliable,[support],curve,baseline)
                three=audit_curve(bf,af,yb,ya,reliable,[support],curve,short.POLICY)
                corroborated=short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],curve)
                controls={}
                controls['unchanged']=short.audit_short_curve(bf,bf,yb,yb,cb,cb,reliable,[support],curve)
                controls['opposite']=short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],-curve)
                controls['wrong_magnitude']=short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],curve/3)
                displaced=(support[0]+1.,support[1]+1.)
                dc=np.where((times>=displaced[0])&(times<displaced[1]),cents,0.)
                controls['wrong_location']=short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[displaced],dc)
                tampered=decoded.copy();sample=int(np.flatnonzero(outside)[0]);tampered[sample]+=np.float32(2**-15)
                bad=case/'tampered_outside.wav';sf.write(bad,tampered,sr,subtype='PCM_24')
                def never_called(*args):raise AssertionError('Sample tamper reached acoustic audit')
                try:audit_delivered_wav(bad,audio,sr,[support],never_called)
                except ValueError as exc:
                    assert str(exc)=='ENCODED_CANDIDATE_OUTSIDE_SUPPORT_CHANGED'
                    tamper=dict(rejected=True,reason=str(exc),sha256=sha(bad))
                else:raise AssertionError('Outside sample tamper was accepted')
                np.savez_compressed(case/'measurements.npz',**af,yin=ya,cc=ca,requested=curve)
                row=dict(**base,support_seconds=support,output=str(path),output_sha256=sha(path),
                    ten_frame=ten,three_frame=three,three_tracker=corroborated,controls=controls,tamper=tamper)
                if corroborated['passed']:
                    analysis=recovery.measured(decoded)
                    np.savez_compressed(case/'candidate_context_NOT_LABEL.npz',raw_views=np.stack(analysis['views']),reliable=analysis['reliable'],fine=analysis['fine'])
                    a,b=region;left=(times>=a*.02)&(times<support[0]-.07)&analysis['reliable']
                    right=(times>support[1]+.07)&(times<b*.02)&analysis['reliable']
                    row['observed_context']=dict(left_anchor_frames=int(left.sum()),right_anchor_frames=int(right.sum()),
                        both_sides_at_least_three=bool(left.sum()>=3 and right.sum()>=3))
                source_rows.append(row);write(case/'result.json',row)
        rows.extend(source_rows);write(folder/'results.json',source_rows)
        print('SHORT_AUDIT_SOURCE',sid,'attempts',len(rows),'three_tracker_pass',sum(r.get('three_tracker',{}).get('passed',False) for r in rows),flush=True)
    assert len(rows)==336
    summaries=[]
    for width in d['widths_seconds']:
        group=[r for r in rows if r['width_seconds']==width]
        summaries.append(dict(width_seconds=width,attempted=len(group),rendered=sum('excluded' not in r for r in group),
            ten_frame_pass=sum(r.get('ten_frame',{}).get('passed',False) for r in group),
            three_frame_pass=sum(r.get('three_frame',{}).get('passed',False) for r in group),
            three_tracker_pass=sum(r.get('three_tracker',{}).get('passed',False) for r in group)))
    generated=[r for r in rows if 'excluded' not in r]
    accepted_negative=sum(c['passed'] for r in generated for c in r['controls'].values())
    write(ROOT/'results.json',dict(declaration_sha256=sha(ROOT/'declaration.json'),cases=rows,summary=summaries,
        negative_controls=len(generated)*4,negative_false_accepts=accepted_negative,
        outside_sample_tamper_rejections=sum(r['tamper']['rejected'] for r in generated),
        new_model_fits=0,production_eligible=False))
    print('SHORT_AUDIT_CALIBRATION_COMPLETE',summaries,'false_accepts',accepted_negative,flush=True)


if __name__=='__main__':main()
