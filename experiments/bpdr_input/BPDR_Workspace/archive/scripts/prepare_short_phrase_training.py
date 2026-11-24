"""Training-only physical short deviations with complete exclusions and input-only views."""
from pathlib import Path
from dataclasses import asdict,replace
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
from bpdr_input.repair.framewise_renderer import RendererPolicy,render_candidate
from bpdr_input.repair.interventions import AuditPolicy
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.centered_precision_audit import coarse_features
from bpdr_input.repair.rule_correction import fine_hz,segment_notes,RulePolicy

ROOT=WS/'results/transformer_usability/2026-11-02_v40_short_phrase_data'
RENDER=replace(RendererPolicy(),version='short-training-diagnostic-render-1',ramp_seconds=.02,min_support_seconds=.24)
AUDIT=replace(AuditPolicy(),version='short-training-diagnostic-audit-1',ramp_seconds=.02,analysis_guard_seconds=.07)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def norm(x):return json.loads(json.dumps(x,allow_nan=False))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def declare():
    check_lock();reservation=read(ROOT/'source_reservation.json')
    timing=ROOT.parent/'2026-11-02_v39_short_timing/integrity_receipt.json'
    receipt=read(timing);assert receipt['audits_replayed']==240 and receipt['opposite_request_rejections']==240
    sources=[s for s in reservation['sources'] if s['split']=='train']
    assert len(sources)==21 and len({s['singer_id'] for s in sources})==14
    for s in sources:assert sha(s['audio_path'])==s['audio_sha256']
    d=norm(dict(version='short-phrase-physical-training-1',script_sha256=sha(__file__),
        source_reservation_sha256=sha(ROOT/'source_reservation.json'),timing_integrity_sha256=sha(timing),
        source_records=sources,planned_attempts=336,widths_seconds=[.24,.36,.50,.70],cents=[-30.,-20.,20.,30.],
        crop_policy=reservation['planned_input'],support_policy='Center of longest source-only reliable acoustic region split70c, >=70ms original context on each side; no quality-outcome search or replacement',
        renderer=asdict(RENDER),audit=asdict(AUDIT),
        labels='Nominal negative injected cents on common audited plateau; zero on reliable unchanged outside; ramps/guards excluded. Clean originals have zero labels on their reliable frames.',
        input_contract='Only acoustic raw_views; target arrays, reference A, injected cents, source IDs and intervention interval never passed to inference.',
        heldout_sources_accessed=False,annotation_arrays_loaded=False,
        intended_pitch_truth=False,research_training_eligible=False,production_eligible=False))
    path=ROOT/'preparation_declaration.json'
    if path.exists():assert read(path)==d
    else:write(path,d)
    return d,sources


def save_analysis(path,a):
    np.savez_compressed(path,raw_views=np.stack(a['views']),reliable=a['reliable'],fine=a['fine'],
                        **a['features'])


def main():
    torch.set_num_threads(2);d,sources=declare();rows=[];originals=[]
    for index,s in enumerate(sources,1):
        folder=ROOT/'training'/s['source_id'];folder.mkdir(parents=True,exist_ok=True)
        result_path=folder/'source_result.json'
        if result_path.exists():
            saved=read(result_path)
            assert saved['declaration_sha256']==sha(ROOT/'preparation_declaration.json')
            for p,h in saved['output_hashes'].items():assert sha(p)==h,p
            rows.extend(saved['pairs']);originals.append(saved['original'])
            print('SHORT_TRAIN_REUSED',s['source_id'],flush=True);continue
        raw,sr=sf.read(s['audio_path'],dtype='float32',always_2d=True)
        channel=int(np.argmax(np.mean(raw.astype(np.float64)**2,axis=0)))
        y=raw[:min(len(raw),10*sr),channel]
        div=gcd(sr,22050)
        if sr!=22050:y=resample_poly(y,22050//div,sr//div).astype(np.float32)
        original=dict(source_id=s['source_id'],group=s['singer_id'],source=s,
            crop_offset_seconds=0.,source_channel=channel,crop_native_samples=min(len(raw),10*sr),
            sample_rate_hz=22050,duration_seconds=len(y)/22050)
        pairs=[];paths=[]
        if len(y)<5*22050 or not np.isfinite(y).all() or np.max(abs(y))>=1:
            original['excluded']='INVALID_SOURCE_DURATION_OR_SAMPLES'
        else:
            path=folder/'original_working.wav';sf.write(path,y,22050,subtype='PCM_24')
            y,_=sf.read(path,dtype='float32');a=recovery.measured(y);times=a['features']['times_seconds']
            analysis_path=folder/'original_analysis.npz';save_analysis(analysis_path,a)
            paths.extend([path,analysis_path]);original.update(path_A=str(path),analysis_A=str(analysis_path))
            regions=segment_notes(a['views'][0][:,64],a['reliable'],RulePolicy(note_split_cents=70.))
            region=max(regions,key=lambda r:r[1]-r[0]) if regions else None
            original['longest_region_frames']=region
            bf=coarse_features(y,22050,WS/'checkpoints');yb=fine_hz(y,22050,times)
        for width in d['widths_seconds']:
            for cents in d['cents']:
                base=dict(source_id=s['source_id'],group=s['singer_id'],width_seconds=width,cents=cents,
                    source=s,split='train',research_training_eligible=False)
                if 'excluded' in original:
                    pairs.append(dict(**base,excluded=original['excluded']));continue
                if region is None or (region[1]-region[0])*.02<width+.14-1e-12:
                    pairs.append(dict(**base,excluded='NO_SINGLE_RELIABLE_REGION_WITH_70MS_CONTEXT'));continue
                center=float(np.mean(region)*.02);support=(center-width/2,center+width/2)
                curve=np.where((times>=support[0])&(times<support[1]),cents,0.)
                base.update(path_A=str(path),analysis_A=str(analysis_path),support_seconds=support)
                try:
                    candidate,render_receipt=render_candidate(y,22050,times,curve,[support],RENDER)
                    cpath=folder/f'detuned_{width:.2f}_{cents:+g}.wav'
                    sf.write(cpath,candidate,22050,subtype='PCM_24');candidate,rate=sf.read(cpath,dtype='float32')
                    assert rate==22050 and candidate.shape==y.shape
                    sample_times=np.arange(len(y))/22050
                    outside_samples=(sample_times<=support[0])|(sample_times>=support[1])
                    assert np.array_equal(y[outside_samples],candidate[outside_samples])
                    cf=coarse_features(candidate,22050,WS/'checkpoints');yc=fine_hz(candidate,22050,times)
                    audit=audit_curve(bf,cf,yb,yc,a['reliable'],[support],curve,AUDIT)
                    apath=folder/f'audit_{width:.2f}_{cents:+g}.json';write(apath,audit)
                    paths.extend([cpath,apath]);base.update(path_C=str(cpath),audit_path=str(apath),audit=audit)
                    if not audit['passed']:
                        pairs.append(dict(**base,excluded='PHYSICAL_INTERVENTION_AUDIT_FAILED'));continue
                    c=recovery.measured(candidate)
                    cp=folder/f'analysis_{width:.2f}_{cents:+g}.npz';save_analysis(cp,c)
                    common=a['reliable']&c['reliable']&bf['voiced_mask']&cf['voiced_mask']&np.isfinite(yb)&np.isfinite(yc)
                    plateau=(times>=support[0]+.09)&(times<=support[1]-.09)&common
                    outside=((times<support[0]-.07)|(times>support[1]+.07))&common
                    assert plateau.sum()>=10,'Runtime reliability removed label observations'
                    target=np.where(plateau,-cents,0.).astype(np.float32);mask=plateau|outside
                    tp=folder/f'targets_{width:.2f}_{cents:+g}_NOT_MODEL_INPUT.npz'
                    np.savez_compressed(tp,target=target,mask=mask,plateau=plateau,times=times)
                    paths.extend([cp,tp]);pairs.append(dict(**base,analysis_C=str(cp),target_path=str(tp),
                        positive_frames=int(plateau.sum()),zero_frames=int(outside.sum()),render_receipt=render_receipt))
                except (ValueError,RuntimeError) as exc:
                    pairs.append(dict(**base,excluded='RENDER_OR_MEASUREMENT_FAILED',detail=str(exc)))
        saved=dict(declaration_sha256=sha(ROOT/'preparation_declaration.json'),original=original,pairs=pairs,
                   output_hashes={str(p):sha(p) for p in paths})
        write(result_path,saved);rows.extend(pairs);originals.append(original)
        write(ROOT/'preparation_progress.json',dict(completed_sources=index,planned_sources=21,
            attempted=len(rows),valid=sum('excluded' not in r for r in rows)))
        print('SHORT_TRAIN_SOURCE',index,'/21',s['source_id'],'valid',sum('excluded' not in r for r in rows),'/',len(rows),flush=True)
    assert len(rows)==336
    write(ROOT/'train_pairs.json',rows);write(ROOT/'train_originals.json',originals)
    hashes={}
    for s in sources:hashes.update(read(ROOT/'training'/s['source_id']/'source_result.json')['output_hashes'])
    write(ROOT/'input_hashes.json',hashes)
    write(ROOT/'preparation_summary.json',dict(attempted=len(rows),valid=sum('excluded' not in r for r in rows),
        source_count=len(originals),attempted_singers=len({s['group'] for s in originals}),
        scoreable_singers=sorted({r['group'] for r in rows if 'excluded' not in r}),
        heldout_sources_accessed=False,model_training_started=False,research_training_eligible=False))
    print('SHORT_TRAINING_PREPARATION_COMPLETE',read(ROOT/'preparation_summary.json'),flush=True)


if __name__=='__main__':main()
