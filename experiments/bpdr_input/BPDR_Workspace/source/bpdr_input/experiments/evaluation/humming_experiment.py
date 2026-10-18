"""HumTrans audio intake, alignment diagnostics and local-detuning transfer probe.

Reference MIDI is diagnostic only. Audio-derived stable regions define injected
errors; recovery targets undo those known injections, not a guessed ideal tune.
"""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import json
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.experiments.evaluation.transfer_study import COMPONENT, read, write, sha, score_predictions, summaries

ROOT=COMPONENT/'runs/humming_experiment'


def midi_notes(path):
    import mido
    now=0.;active=defaultdict(list);notes=[]
    for event in mido.MidiFile(path):
        now+=event.time
        if event.type=='note_on' and event.velocity:
            active[(event.channel,event.note)].append(now)
        elif event.type=='note_off' or (event.type=='note_on' and not event.velocity):
            key=(event.channel,event.note)
            if active[key]:notes.append((active[key].pop(0),now,event.note))
    if any(active.values()):raise ValueError('MIDI contains unfinished notes')
    return sorted(n for n in notes if n[1]>n[0])


def alignment(notes,times,fine,reliable):
    """Report lag/octave sensitivity without changing labels or model features."""
    raw=[]
    for lag in np.arange(-1.,1.001,.02):
        reference=np.full(len(times),np.nan)
        for a,b,p in notes:reference[(times>=a+lag)&(times<b+lag)]=(p-69)*100.
        m=reliable & np.isfinite(reference)
        if m.sum()<10:continue
        for octave in [-24,-12,0,12,24]:
            difference=fine[m]-(reference[m]+100*octave)
            raw.append(dict(lag_seconds=float(round(lag,3)),octave_semitones=octave,frames=int(m.sum()),
                            within_50_cents=float((np.abs(difference)<=50).mean()),median_absolute_cents=float(np.median(np.abs(difference)))))
    nominal=next((r for r in raw if abs(r['lag_seconds'])<.001 and r['octave_semitones']==0),None)
    best=max(raw,key=lambda r:(r['within_50_cents'],-abs(r['lag_seconds']),-abs(r['octave_semitones']))) if raw else None
    return dict(nominal=nominal,best_diagnostic_only=best,
        limitation='Lag/octave search is a diagnostic with performed-pitch estimates, not a corrected label, independent transcription score, musical judgement or model input')


def fine_features(features,audio,sr):
    from bpdr_input.repair.rule_correction import fine_pitch, reliable_frames
    from bpdr_input.preparation.pairs.shared_pair_audit import game_boundary_track
    from scipy.signal import find_peaks
    fine=fine_pitch(audio,sr,features['times_seconds']);reliable=reliable_frames(features,fine)
    x=features['boundary_features_71'].copy();x[:,64]=np.where(reliable,fine,0.);x[:,66]=reliable;x[:,67]=0.
    x[1:,67]=np.where(reliable[1:] & reliable[:-1],np.diff(fine),0.)
    peaks,_=find_peaks(features['provisional_boundary_score'],height=.5,distance=5)
    edges=features['times_seconds'][peaks];x[:,70]=game_boundary_track(features['times_seconds'],edges)
    return x,fine,reliable,edges


def prepare(intake_path=None):
    import torch
    from scipy.signal import resample_poly
    from math import gcd
    from bpdr_input.audio.analysis import extract
    from bpdr_input.repair.rule_correction import segment_notes
    from bpdr_input.repair.interventions import render_local, envelope
    from bpdr_input.preparation.pairs.shared_pair_audit import game_boundary_track
    torch.set_num_threads(2)
    intake=Path(intake_path) if intake_path else COMPONENT/'data/humtrans/intake_receipt.json';receipt=read(intake)
    out=ROOT/uuid.uuid4().hex;out.mkdir(parents=True)
    declaration=dict(version='humming-pairs-1',created_utc=datetime.now(timezone.utc).isoformat(),intake_sha256=sha(intake),source_sha256=sha(__file__),
        crop='First min(duration,20) seconds, offset zero; preserve source; mono strongest channel, polyphase resample to 22050 Hz',
        support='Longest audio-derived reliable note span >=1.10 seconds; 0.10-second inside margins; no score-derived selection',
        shifts=[-30.,-20.,-10.,10.,20.,30.],boundary_corruption='Nominal heuristic boundary times versus +40 ms jitter',
        audit='At least 10 plateau/outside common reliable frames; >=75% plateau coverage; median shift error <=5c, p90 <=10c; outside median <=5c',
        model_features='Fine YIN/PESTO agreement and explicit voicing; score reference, target and intervention metadata excluded',
        limitations=['Small deterministic subset, filename-person split; not representative novice sample','Score timing/absolute tuning are not ideal-repair labels'],
        research_ready_examples=0,final_test_audio_accessed=False,review_status='pending')
    write(out/'declaration.json',declaration);(out/'executed_humming_experiment.py').write_bytes(Path(__file__).read_bytes())
    source_rows=[];records=[]
    for source in receipt['records']:
        if sha(source['path'])!=source['sha256']:raise ValueError('HumTrans source changed')
        folder=out/source['key'];folder.mkdir()
        original,sr=sf.read(source['path'],dtype='float32')
        if original.ndim==2:original=original[:,np.argmax(np.mean(original.astype(np.float64)**2,axis=0))]
        original=original[:20*sr];divisor=gcd(sr,22050)
        audio=resample_poly(original,22050//divisor,sr//divisor).astype(np.float32) if sr!=22050 else original
        rate=22050;sf.write(folder/'A.wav',audio,rate,subtype='PCM_24');audio,_=sf.read(folder/'A.wav',dtype='float32')
        features,_=extract(audio,rate,COMPONENT/'checkpoints');np.savez_compressed(folder/'baseline_features.npz',**features)
        times=features['times_seconds'];a,fine,reliable,edges=fine_features(features,audio,rate)
        notes=midi_notes(source['midi_path']);diag=alignment(notes,times,fine,reliable)
        spans=segment_notes(fine,reliable)
        spans=[(lo,hi) for lo,hi in spans if times[hi-1]-times[lo]>=1.10]
        source_row=dict(source,working_path=str(folder/'A.wav'),working_sha256=sha(folder/'A.wav'),duration_seconds=len(audio)/rate,
                        voiced_fraction=float(reliable.mean()),score_alignment=diag,eligible_note_spans=len(spans),variants=[])
        if not spans:
            source_row['status']='NO_QUALIFYING_STABLE_NOTE';source_rows.append(source_row);print(source['key'],source_row['status'],flush=True);continue
        lo,hi=max(spans,key=lambda s:s[1]-s[0]);support=(float(times[lo]+.1),float(times[hi-1]-.1))
        alternate=a.copy();alternate[:,70]=game_boundary_track(times,np.clip(edges+.04,0,times[-1]))
        for cents in declaration['shifts']:
            variant=folder/f'shift_{cents:+g}';variant.mkdir()
            candidate,_=render_local(audio,rate,cents,support)
            sf.write(variant/'C.wav',candidate,rate,subtype='PCM_24');candidate,_=sf.read(variant/'C.wav',dtype='float32')
            feat,_=extract(candidate,rate,COMPONENT/'checkpoints');c,cfine,creliable,_=fine_features(feat,candidate,rate)
            c[:,70]=a[:,70];d=c.copy();d[:,70]=alternate[:,70]
            plateau=(times>=support[0]+.18)&(times<=support[1]-.18)
            outside=(times<support[0]-.1)|(times>support[1]+.1)
            common=reliable & creliable;pm=plateau & common;om=outside & common
            errors=np.abs(cfine[pm]-fine[pm]-cents);drift=np.abs(cfine[om]-fine[om])
            audit=dict(plateau_frames=int(pm.sum()),outside_frames=int(om.sum()),coverage=float(pm.sum()/max(1,plateau.sum())),
                median_error_cents=float(np.median(errors)) if len(errors) else None,p90_error_cents=float(np.percentile(errors,90)) if len(errors) else None,
                outside_median_cents=float(np.median(drift)) if len(drift) else None)
            passed=pm.sum()>=10 and om.sum()>=10 and audit['coverage']>=.75 and audit['median_error_cents']<=5 and audit['p90_error_cents']<=10 and audit['outside_median_cents']<=5
            audit['passes']=bool(passed);write(variant/'audit.json',audit);source_row['variants'].append(dict(cents=cents,**audit))
            if not passed:continue
            hashes={}
            for v,x in dict(A=a,B=alternate,C=c,D=d).items():
                p=variant/(v+'_input.npz');np.savez_compressed(p,times_seconds=times,boundary_features_71=x,frame_features_70=x[:,:70]);hashes[v]=sha(p)
            mask=pm|om;det=cents*envelope(times,support)
            target=variant/'targets_NOT_MODEL_INPUT.npz'
            np.savez_compressed(target,times_seconds=times,common_reliable_mask=mask,correction_target_cents=-det)
            records.append(dict(dataset='humtrans',phrase_id=source['key'],source_id=source['key'],singer_id=source['person_id'],split=source['split'],
                group_id='humtrans:'+source['person_id'],reference_type='audio_estimate_known_injected_shift',cents=cents,corruption='jitter',
                input_directory=str(variant),input_sha256=hashes,target_path=str(target),targets_sha256=sha(target),audit_passed=True,
                training_eligible=False,review_status='pending',data_status='exploratory_not_research',support_seconds=support,
                waveform_paths=dict(A=str(folder/'A.wav'),B=str(folder/'A.wav'),C=str(variant/'C.wav'),D=str(variant/'C.wav'))))
        source_row['status']='PAIRS_AVAILABLE' if any(r['phrase_id']==source['key'] for r in records) else 'ALL_INTERVENTIONS_WITHHELD'
        source_rows.append(source_row);print(source['key'],source_row['status'],'passing',sum(r['phrase_id']==source['key'] for r in records),flush=True)
    for split in ['train','development']:write(out/(split+'_catalog.json'),dict(records=[r for r in records if r['split']==split]))
    summary=dict(recordings=len(source_rows),usable_phrases=sum(r['status']=='PAIRS_AVAILABLE' for r in source_rows),
        paired_variants=len(records),train_variants=sum(r['split']=='train' for r in records),development_variants=sum(r['split']=='development' for r in records))
    write(out/'evidence.json',dict(declaration_sha256=sha(out/'declaration.json'),summary=summary,sources=source_rows,research_ready_examples=0,
        interpretation='Known local-shift recovery on real humming; no musical approval or novice performance claim'))
    write(ROOT/'current.json',dict(path=str(out)));print('HUMMING_PREPARED',json.dumps(summary),out,flush=True)


def evaluate():
    import torch
    from bpdr_input.repair.model_runtime import default_checkpoint
    from bpdr_input.reporting.bpdr_showcase import load_model
    from bpdr_input.experiments.learning.real_learning_v2 import pair
    from bpdr_input.modeling.examples import collate
    torch.set_num_threads(2)
    base=Path(read(ROOT/'current.json')['path']);out=base/'frozen_evaluation';out.mkdir()
    cp=default_checkpoint();write(out/'declaration.json',dict(checkpoint=str(cp),sha256=sha(cp),optimizer_updates=0,split='development',selection='all available fixed development pairs',final_test_audio_accessed=False))
    model,norm,width=load_model(cp);rows=[]
    with torch.inference_mode():
        for r in read(base/'development_catalog.json')['records']:
            p=pair(r);b=collate([p],norm,width)
            predictions={v:model(x,b['padding'])[0].numpy() for v,x in b['inputs'].items()}
            rows.append(dict(score_predictions(p,predictions),dataset='humtrans',group=r['singer_id'],phrase_id=r['phrase_id'],cents=r['cents'],corruption=r['corruption']))
    write(out/'rows.json',rows);write(out/'evidence.json',dict(results=summaries(rows) if rows else None,variants=len(rows),checkpoint_unchanged=sha(cp)==read(out/'declaration.json')['sha256'],model_adopted=False))
    print('HUMMING_EVALUATED',len(rows),flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare(sys.argv[2] if len(sys.argv)>2 else None)
    else:evaluate()
