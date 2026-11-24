"""Prospective timing-only renderer/audit comparison; no model training or activation."""
from pathlib import Path
from dataclasses import asdict,replace
import json,sys
import numpy as np
import soundfile as sf
import torch

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
from bpdr_input.audio.analysis import _pitch_model
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.repair.framewise_renderer import RendererPolicy,render_candidate
from bpdr_input.repair.interventions import AuditPolicy
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.centered_precision_audit import coarse_features
from bpdr_input.repair.rule_correction import fine_hz,segment_notes,RulePolicy
from probe_short_phrase_renderer import phrase,sha
from acquire_fresh_confirmation import check_lock

ROOT=WS/'results/transformer_usability/2026-11-02_v39_short_timing'
PREV=WS/'results/transformer_usability/2026-11-01_v37_fresh_confirmation/preparation'
ARMS={'baseline':(.08,.10),'window_guard':(.08,.07),'short_ramp':(.02,.07)}
WIDTHS=[.24,.36,.50,.70]


def norm(x):return json.loads(json.dumps(x,allow_nan=False))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def footprint():
    model=_pitch_model(str((WS/'checkpoints').resolve()))
    kernels=model.preprocessor.hcqt_kernels.cqt_kernels
    offsets=[];details=[]
    for kernel in kernels:
        weight=kernel.conv.weight.detach().numpy()
        active=np.flatnonzero(np.any(weight!=0,axis=(0,1)))
        center=kernel.conv.padding[0]
        left,right=int(active[0]-center),int(active[-1]-center)
        offsets.extend([abs(left)/22050,abs(right)/22050])
        details.append(dict(kernel_width=kernel.kernel_width,nonzero_left_samples=left,
                            nonzero_right_samples=right,centered=kernel.center))
    assert max(offsets)<.07 and 1024/22050<.07
    assert model.hop_size==20.
    return dict(kernels=details,max_nonzero_half_support_seconds=max(offsets),
        yin_frame_samples=2048,yin_half_support_seconds=1024/22050,
        guard_seconds=.07,hop_seconds=.02,
        qualification='Finite nonzero acoustic-kernel support plus YIN window. Installed code treats time as batch; no temporal neural convolution. This is a window-containment argument, not evidence of estimator accuracy or independent frames.')


def declare():
    check_lock();ROOT.mkdir(parents=True,exist_ok=True)
    sources=read(PREV/'original_sources.json')
    for s in sources:assert sha(s['working_audio'])==s['working_sha256']
    paths=[Path(__file__),Path(__file__).with_name('probe_short_phrase_renderer.py'),
           WS.parent/'.venv/Lib/site-packages/pesto/data.py',
           WS.parent/'.venv/Lib/site-packages/pesto/model.py',
           WS.parent/'.venv/Lib/site-packages/pesto/utils/hcqt.py',
           WS.parent/'.venv/Lib/site-packages/pesto/weights/mir-1k_g7.ckpt']
    d=norm(dict(version='short-timing-diagnostic-1',source_hashes={str(p.resolve()):sha(p) for p in paths},
        previous_manifest_sha256=sha(PREV/'original_sources.json'),source_records=sources,
        acoustic_footprint=footprint(),arms=ARMS,widths_seconds=WIDTHS,shifts_cents=[-30.,30.],
        synthetic_registers_hz=[120,240],synthetic_start_offsets_seconds=[0.,.01],
        real_crop='First min(10 s, working recording duration), no silence removal or outcome replacement',
        real_support='Center of longest current reliable acoustic region split at70 c; require70 ms original context on each side; same source-only location across arms and directions',
        planned_real_attempts=96,planned_synthetic_attempts=32,planned_arm_attempts=384,
        renderer_base=asdict(RendererPolicy()),audit_base=asdict(AuditPolicy()),
        runtime_unchanged=True,controller_bypassed_for_user_export=False,
        v37_usage='Previously inspected sources, now diagnostic only. No training use or fresh-confirmation claim. All source restrictions remain in force.',
        model_training=False,correction_model_inference=False,production_eligible=False,
        research_validation_passed=False,accepted_audio=None,
        hypothesis='A70 ms guard contains both estimator windows; shorter ramps may expose an interior for short requests without loosening error/direction/coverage/frame-count gates. Retain every failed/excluded case.'))
    p=ROOT/'declaration.json'
    if p.exists():assert read(p)==d,'Immutable declaration mismatch'
    else:write(p,d)
    return d,sources


def sources_and_supports(sources):
    for base in [120,240]:
        for width in WIDTHS:
            for phase in [0.,.01]:
                y,support=phrase(base,width)
                yield dict(id=f'synthetic_{base}_{width:.2f}_{phase:.2f}',kind='synthetic',
                    group=f'synthetic_{base}',audio=y,widths=[width],
                    support_center=np.mean(support)+phase,region=None)
    for i,s in enumerate(sources,1):
        y,sr=sf.read(s['working_audio'],dtype='float32');assert sr==22050
        y=y[:10*sr]
        m=recovery.measured(y);raw=m['views'][0]
        regions=segment_notes(raw[:,64],m['reliable'],RulePolicy(note_split_cents=70.))
        region=max(regions,key=lambda r:r[1]-r[0]) if regions else None
        center=float(np.mean(region)*.02) if region else None
        yield dict(id=f'real_{i:02d}',kind='real_voice',group=s['source']['singer_id'],
            audio=y,widths=WIDTHS,support_center=center,region=region,
            source_working_sha256=s['working_sha256'])


def main():
    torch.set_num_threads(2);d,sources=declare();rows=[];originals=[]
    for source in sources_and_supports(sources):
        folder=ROOT/source['id'];folder.mkdir(exist_ok=True)
        original=folder/'original.wav';sf.write(original,source['audio'],22050,subtype='PCM_24')
        audio,sr=sf.read(original,dtype='float32');times=np.arange(len(audio)//441+1)*.02
        before=coarse_features(audio,sr,WS/'checkpoints');yb=fine_hz(audio,sr,times)
        reliable=before['voiced_mask']&np.isfinite(yb)&(yb>0)
        identity,_=render_candidate(audio,sr,times,np.zeros(len(times)),[],RendererPolicy())
        assert np.array_equal(identity,audio)
        originals.append(dict(id=source['id'],kind=source['kind'],group=source['group'],
            working_sha256=sha(original),identity_exact=True,sample_count=len(audio),
            longest_region_frames=source['region'],source_working_sha256=source.get('source_working_sha256')))
        for width in source['widths']:
            center=source['support_center'];region=source['region']
            valid=center is not None and (source['kind']=='synthetic' or
                (region[1]-region[0])*.02>=width+.14-1e-12)
            support=(float(center-width/2),float(center+width/2)) if center is not None else None
            for cents in [-30.,30.]:
                base=dict(id=source['id'],kind=source['kind'],group=source['group'],
                          width_seconds=width,cents=cents,support=support)
                if not valid:
                    rows.extend(dict(**base,arm=arm,excluded='NO_SINGLE_RELIABLE_REGION_WITH_70MS_CONTEXT') for arm in ARMS)
                    continue
                curve=np.where((times>=support[0])&(times<support[1]),cents,0.)
                cache={}
                for arm,(ramp,guard) in ARMS.items():
                    renderer=replace(RendererPolicy(),version='timing-diagnostic-1',ramp_seconds=ramp,min_support_seconds=.24)
                    policy=replace(AuditPolicy(),version='timing-diagnostic-audit-1',ramp_seconds=ramp,analysis_guard_seconds=guard)
                    row=dict(**base,arm=arm,renderer_policy=asdict(renderer),audit_policy=asdict(policy))
                    try:
                        if ramp not in cache:
                            output,render_receipt=render_candidate(audio,sr,times,curve,[support],renderer)
                            path=folder/f'{width:.2f}_{cents:+g}_ramp{ramp:.2f}.wav'
                            sf.write(path,output,sr,subtype='PCM_24');decoded,rate=sf.read(path,dtype='float32')
                            assert rate==sr and decoded.shape==audio.shape and np.isfinite(decoded).all() and abs(decoded).max()<1
                            t=np.arange(len(audio))/sr;outside=(t<=support[0])|(t>=support[1])
                            assert np.array_equal(decoded[outside],audio[outside])
                            after=coarse_features(decoded,sr,WS/'checkpoints');ya=fine_hz(decoded,sr,times)
                            cache[ramp]=(path,after,ya,render_receipt)
                        path,after,ya,render_receipt=cache[ramp]
                        audit=audit_curve(before,after,yb,ya,reliable,[support],curve,policy)
                        wrong=audit_curve(before,after,yb,ya,reliable,[support],-curve,policy)
                        assert not wrong['passed'],'Opposite-request control unexpectedly passed'
                        row.update(audit=audit,opposite_request_audit=wrong,
                            output=str(path),output_sha256=sha(path),outside_samples_exact=True,
                            render_receipt=render_receipt)
                    except (ValueError,RuntimeError) as exc:
                        row.update(excluded='RENDER_OR_MEASUREMENT_FAILED',detail=str(exc))
                    rows.append(row)
        write(folder/'results.json',[r for r in rows if r['id']==source['id']])
        print('SHORT_TIMING_SOURCE',source['id'],'arm_attempts',len(rows),
              'passed',sum(r.get('audit',{}).get('passed',False) for r in rows),flush=True)
    assert len(rows)==d['planned_arm_attempts']
    summary=[]
    for kind in ['synthetic','real_voice']:
        for arm in ARMS:
            for width in WIDTHS:
                group=[r for r in rows if r['kind']==kind and r['arm']==arm and r['width_seconds']==width]
                summary.append(dict(kind=kind,arm=arm,width_seconds=width,attempted=len(group),
                    rendered=sum('audit' in r for r in group),passed=sum(r.get('audit',{}).get('passed',False) for r in group)))
    write(ROOT/'results.json',dict(declaration_sha256=sha(ROOT/'declaration.json'),
        cases=rows,originals=originals,summary=summary,model_accuracy_assessed=False,
        correction_model_inference=False,production_eligible=False))
    print('SHORT_TIMING_COMPLETE',summary,flush=True)


if __name__=='__main__':main()
