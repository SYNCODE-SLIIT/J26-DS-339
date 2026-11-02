"""Experimental coarse-auditor precision variant; no changed gate or model input."""
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from bpdr_input.audio.analysis import _pitch_model
from bpdr_input.modeling.pesto_precision import decode_float64
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.rule_correction import fine_hz
from bpdr_input.repair.repair_controller import transact_fixture,waveform_hash
from bpdr_input.repair.framewise_renderer import render_candidate
from bpdr_input.repair.learned_runtime import audit_delivered_wav


def coarse_features(audio,sr,cache_dir):
    if sr!=22050:raise ValueError('Fixed audit sample rate required')
    model=_pitch_model(str(Path(cache_dir).resolve()))
    assert model.reduction=='alwa' and model.bins_per_semitone==3
    with torch.inference_mode():
        _,confidence,_,activations=model(torch.as_tensor(audio,dtype=torch.float32),sr,convert_to_freq=True,return_activations=True)
    native=decode_float64(activations.numpy().reshape(-1,384));conf=confidence.numpy().reshape(-1)
    hop=round(model.hop_size*sr/1000.)/sr
    nt=np.arange(len(native))*hop;times=np.arange(len(audio)//441+1)*441/sr
    index=np.clip(np.searchsorted(nt,times),0,len(nt)-1);previous=np.maximum(index-1,0)
    index=np.where(abs(nt[previous]-times)<abs(nt[index]-times),previous,index)
    voiced=(conf[index]>=.8)&(native[index]>0)
    return dict(times_seconds=times,f0_hz=np.where(voiced,native[index],np.nan),voiced_mask=voiced,confidence=conf[index],native_pitch_times_seconds=nt,native_pitch_hz=native,native_pitch_confidence=conf)


def auditor_for(pair,view,curve,cache_dir,cache):
    analysis=pair[view];times=analysis['features']['times_seconds'];reliable=analysis['reliable']
    def measured(audio):
        key=waveform_hash(audio)
        if key not in cache:cache[key]=(coarse_features(audio,22050,cache_dir),fine_hz(audio,22050,times))
        return cache[key]
    def auditor(before,after,sr,intervals):
        if sr!=22050:raise ValueError('Fixed audit sample rate required')
        bf,yb=measured(before);af,ya=measured(after)
        assert np.array_equal(bf['times_seconds'],times)
        result=audit_curve(bf,af,yb,ya,reliable,intervals,curve)
        result['coarse_measurement']='Frozen float32 PESTO encoder; unchanged ALWA window/weights; float64 pitch/frequency decoding; unchanged native alignment and voicedness mask'
        return result
    return auditor


def rendered_output(pair,view,curve,plan,folder,cache_dir,cache):
    audio=pair['audio_'+view];times=pair[view]['features']['times_seconds']
    auditor=auditor_for(pair,view,curve,cache_dir,cache)
    renderer=lambda a,sr,intervals:render_candidate(a,sr,times,curve,intervals)[0]
    output,transaction=transact_fixture(audio,22050,plan,renderer,auditor)
    if transaction['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
        folder.mkdir(parents=True,exist_ok=True);path=folder/(view+'_candidate.wav')
        sf.write(path,output,22050,subtype='PCM_24')
        try:
            transaction['delivered_wav_audit']=audit_delivered_wav(path,audio,22050,plan['intervals_seconds'],auditor)
            output,_=sf.read(path,dtype='float32')
        except Exception as exc:
            transaction.update(outcome='ROLLED_BACK',rollback=True,delivered_error=str(exc));output=audio.copy()
    return output,transaction
