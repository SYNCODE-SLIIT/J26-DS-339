"""Bounded source-eligibility pilot: reliable melody versus steady-pitch regions.

No model fit or runtime-policy change. All old decoded-waveform audit limits stay.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch

import prepare_full_training_windows as inputs
import calibrate_short_support_audit as cal

WS=inputs.WS
ROOT=WS/'results/transformer_usability/2026-11-04_v60_melodic_construction'
GEOMETRY=ROOT.parent/'2026-11-04_v59_melodic_geometry'
BASE=ROOT.parent/'2026-11-04_v58_full_window_short_pairs'
read,write,norm,sha=inputs.read,inputs.write,inputs.norm,inputs.sha


def declare():
    inputs.common.check_lock()
    geometry=read(GEOMETRY/'results.json');receipt=read(BASE/'integrity_receipt.json')
    assert receipt['staged_pairs_verified']==349
    new_groups=set(geometry['additional_potential_groups'])
    selected=[r for r in geometry['rows'] if r['group'] in new_groups and not r['current_widths'] and r['continuous_reliability_widths']]
    assert len(selected)==4 and new_groups=={'S10','S15','S20'}
    originals={r['source_id']:r for r in read(inputs.ROOT/'train_originals.json')}
    baseline=read(BASE/'train_pairs.json')
    assert all('excluded' in r for r in baseline if r['source_id'] in {s['source_id'] for s in selected})
    sources=[]
    for row in selected:
        source=originals[row['source_id']]
        assert 'excluded' not in source and source['split']=='train'
        sources.append(dict(original=source,reliable_region=row['reliable_run']))
    ROOT.mkdir(parents=True,exist_ok=True)
    d=norm(dict(version='melodic-source-eligibility-pilot-1',source_sha256=sha(__file__),
        geometry_sha256=sha(GEOMETRY/'results.json'),baseline_integrity_sha256=sha(BASE/'integrity_receipt.json'),
        input_integrity_sha256=sha(inputs.ROOT/'integrity_receipt.json'),sources=sources,
        source_hashes={str(p):sha(p) for s in sources for p in [Path(s['original']['path_A']),Path(s['original']['analysis_A'])]},
        planned_attempts=64,widths_seconds=[.24,.28,.36,.50],cents=[-30.,-20.,20.,30.],
        baseline='All64 selected-window attempts lack a steady-pitch source region in V58; source-eligibility baseline0 scoreable cases',
        changed_variable='Longest continuous source-reliable span replaces70c steady-pitch segmentation; no joining missing frames',
        fixed='Same source PCM,150ms context each side,center rule,renderer,3-tracker errors/direction/common-count/coverage and exact-outside samples',
        renderer=asdict(cal.RENDER),audit=asdict(cal.short.POLICY),
        controls=['unchanged','opposite','one-third requested magnitude','displaced requested support+1s if it fits else-1s','outside PCM sample tamper'],
        stopping_rule='At most64 attempts,0 fits; on any accepted negative control mark FAILED and stop. Preserve declaration/unexecuted count. No labels admitted before independent replay.',
        timing_scope='Offline construction timings on this machine; not live repair or end-to-end app latency',
        nontraining_audio_accessed=False,guide_notes_loaded=False,new_model_fits=0,
        research_training_eligible=False,production_eligible=False))
    p=ROOT/'declaration.json'
    if p.exists():assert read(p)==d
    else:write(p,d)
    return d


def main():
    torch.set_num_threads(2);d=declare();rows=[];started=time.perf_counter()
    for source in d['sources']:
        original=source['original'];sid=original['source_id'];region=source['reliable_region']
        folder=ROOT/sid;folder.mkdir(exist_ok=True)
        if (folder/'results.json').exists():
            saved=read(folder/'results.json');assert saved['declaration_sha256']==sha(ROOT/'declaration.json')
            for path,expected in saved['output_hashes'].items():assert sha(path)==expected
            rows.extend(saved['cases']);continue
        evidence_start=time.perf_counter();audio,sr=sf.read(original['path_A'],dtype='float32');assert sr==22050
        with np.load(original['analysis_A']) as z:times=z['times_seconds'].copy();reliable=z['reliable'].copy()
        bf=cal.coarse_features(audio,sr,WS/'checkpoints');yb=cal.fine_hz(audio,sr,times)
        cb=cal.short.cross_correlation_pitch(audio,sr,times)
        evidence_elapsed=time.perf_counter()-evidence_start
        before=folder/'before_measurements.npz';np.savez_compressed(before,**bf,yin=yb,cc=cb,reliable=reliable)
        paths=[before];source_rows=[]
        for width in d['widths_seconds']:
            for cents in d['cents']:
                case_start=time.perf_counter()
                row=dict(source_id=sid,group=original['group'],width_seconds=width,cents=cents)
                if (region[1]-region[0])*.02<width+.30-1e-12:
                    source_rows.append(dict(**row,excluded='NO_CONTINUOUS_RELIABLE_SPAN_WITH_150MS_CONTEXT'));continue
                center=float(np.mean(region)*.02);support=(center-width/2,center+width/2)
                requested=np.where((times>=support[0])&(times<support[1]),cents,0.)
                case=folder/f'{width:.2f}_{cents:+g}';case.mkdir(exist_ok=True)
                try:output,render_receipt=cal.render_candidate(audio,sr,times,requested,[support],cal.RENDER)
                except ValueError as error:
                    source_rows.append(dict(**row,excluded='RENDER_ERROR',detail=str(error)));continue
                if not np.isfinite(output).all() or abs(output).max()>=1:
                    source_rows.append(dict(**row,excluded='RENDER_SAMPLE_FAILURE'));continue
                path=case/'intervention.wav';sf.write(path,output,sr,subtype='PCM_24')
                decoded,rate=sf.read(path,dtype='float32');assert rate==sr and decoded.shape==audio.shape
                st=np.arange(len(audio))/sr;outside=(st<=support[0])|(st>=support[1])
                np.testing.assert_array_equal(decoded[outside],audio[outside])
                af=cal.coarse_features(decoded,sr,WS/'checkpoints');ya=cal.fine_hz(decoded,sr,times)
                ca=cal.short.cross_correlation_pitch(decoded,sr,times)
                audit=cal.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],requested)
                shift=1. if support[1]+1.<=len(audio)/sr else -1.
                displaced=(support[0]+shift,support[1]+shift)
                wrong_location=np.where((times>=displaced[0])&(times<displaced[1]),cents,0.)
                controls=dict(
                    unchanged=cal.short.audit_short_curve(bf,bf,yb,yb,cb,cb,reliable,[support],requested),
                    opposite=cal.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],-requested),
                    wrong_magnitude=cal.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],requested/3),
                    wrong_location=cal.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[displaced],wrong_location))
                tampered=decoded.copy();tampered[int(np.flatnonzero(outside)[0])]+=np.float32(2**-15)
                bad=case/'outside_tamper.wav';sf.write(bad,tampered,sr,subtype='PCM_24')
                def never_called(*args):raise AssertionError('Outside tamper reached acoustic audit')
                try:cal.audit_delivered_wav(bad,audio,sr,[support],never_called)
                except ValueError as error:
                    assert str(error)=='ENCODED_CANDIDATE_OUTSIDE_SUPPORT_CHANGED'
                    tamper=dict(rejected=True,reason=str(error))
                else:raise AssertionError('Outside tamper accepted')
                measured=case/'measurements.npz';np.savez_compressed(measured,**af,yin=ya,cc=ca,requested=requested)
                paths.extend([path,bad,measured])
                row.update(support_seconds=support,output=str(path),output_sha256=sha(path),audit=audit,
                    controls=controls,displaced_support_seconds=displaced,tamper=tamper,render_receipt=render_receipt)
                if audit['passed']:
                    c=cal.recovery.measured(decoded)
                    context=case/'input_context.npz';np.savez_compressed(context,raw_views=np.stack(c['views']),reliable=c['reliable'],fine=c['fine'],times_seconds=times)
                    paths.append(context)
                    left=(times>=region[0]*.02)&(times<support[0]-.07)&c['reliable']
                    right=(times>support[1]+.07)&(times<region[1]*.02)&c['reliable']
                    row['retained_context']=dict(left=int(left.sum()),right=int(right.sum()),both_sides_at_least_three=bool(min(left.sum(),right.sum())>=3))
                row['construction_elapsed_seconds']=time.perf_counter()-case_start
                source_rows.append(row);write(case/'result.json',row)
                if any(control['passed'] for control in controls.values()):
                    write(ROOT/'failure.json',dict(status='FAILED_NEGATIVE_CONTROL',completed_cases=len(rows)+len(source_rows),unexecuted=64-len(rows)-len(source_rows),case=row))
                    raise RuntimeError('Negative-control false acceptance: pilot stopped')
        saved=dict(declaration_sha256=sha(ROOT/'declaration.json'),cases=source_rows,source_evidence_elapsed_seconds=evidence_elapsed,output_hashes={str(p):sha(p) for p in paths})
        write(folder/'results.json',saved);rows.extend(source_rows)
        print('MELODIC_CONSTRUCTION_SOURCE',sid,sum(r.get('audit',{}).get('passed',False) for r in source_rows),'/',len(source_rows),flush=True)
    assert len(rows)==64
    rendered=[r for r in rows if 'excluded' not in r]
    passes=[r for r in rendered if r['audit']['passed']]
    controls=[control for r in rendered for control in r['controls'].values()]
    assessable=[c for c in controls if c['three_estimator_common_frames']>=3 and c['plateau_coverage_fraction']>=.75 and c['cc']['coverage']>=.75]
    durations=[r['construction_elapsed_seconds'] for r in rendered]
    summary=dict(attempted=64,baseline_scoreable=0,rendered=len(rendered),physical_audit_passes=len(passes),
        passes_with_retained_context=sum(r['retained_context']['both_sides_at_least_three'] for r in passes),
        passes_by_group=dict(Counter(r['group'] for r in passes)),negative_controls=len(controls),
        negative_false_accepts=sum(c['passed'] for c in controls),negative_controls_with_adequate_common_coverage=len(assessable),
        outside_sample_tamper_rejections=len(rendered),
        construction_median_p95_seconds=np.quantile(durations,[.5,.95]).tolist() if durations else None,
        invocation_elapsed_seconds=time.perf_counter()-started,new_model_fits=0,production_eligible=False)
    write(ROOT/'results.json',dict(declaration_sha256=sha(ROOT/'declaration.json'),cases=rows,summary=summary))
    print('MELODIC_CONSTRUCTION_PILOT_COMPLETE',summary,flush=True)


if __name__=='__main__':main()
