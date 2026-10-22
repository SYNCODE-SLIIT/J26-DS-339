"""Run manifest always distinguishes real inference, assumptions and failures."""
import json
import time
import uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import soundfile as sf
from pp1_demo.config import ROOT,RUNTIME,PASAN,RAHUL,VOCAB
from pp1_demo.contracts import sha,beat_grid,validate_notes
from pp1_demo.adapters import vishwa,pasan,rahul
from pp1_demo import rendering


def save(run,folder):
    (folder/'manifest.json').write_text(json.dumps(run,indent=2,allow_nan=False))


def run_pipeline(path,settings,progress=lambda stage,message:None):
    settings.validate()
    path=Path(path)
    info=sf.info(path)
    if not .25<=info.duration<=20: raise ValueError('Use a 0.25–20 second solo vocal clip for this demonstration; no silent cropping is applied.')
    folder=RUNTIME/'runs'/uuid.uuid4().hex[:12];folder.mkdir(parents=True)
    source=folder/('input'+path.suffix.lower());source.write_bytes(path.read_bytes())
    run={'id':folder.name,'created_utc':datetime.now(timezone.utc).isoformat(),'source_name':path.name,
         'audio_sha256':sha(source),'input_file':source.name,'duration_s':info.duration,
         'settings':settings.validate(),'provenance':{'key':'user_supplied_major_reference','beat_grid':'user_configured','genre':'user_supplied','rahul_mode':'demo_constant'},
         'stages':{},'artifacts':{},'warnings':[], 'adapter_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'pp1_demo').rglob('*.py')) if 'runtime' not in p.parts},
         'vocabulary_hashes':{p.name:sha(p) for p in [VOCAB/'feature_vocabularies.json',VOCAB/'target_vocabulary.json']}}
    save(run,folder)
    def stage(name,fn):
        progress(name,'Running');t=time.perf_counter()
        try:
            result=fn();run['stages'][name]={'status':'Complete','seconds':round(time.perf_counter()-t,3)}
            save(run,folder);progress(name,'Complete');return result
        except Exception as exc:
            run['stages'][name]={'status':'Failed','error':f'{type(exc).__name__}: {exc}','seconds':round(time.perf_counter()-t,3)}
            save(run,folder);progress(name,'Failed');raise
    run['stages']['Praveen']={'status':'Bypassed','reason':'Required model, VocalSet manifest and audited training assets are absent; original audio used unchanged.'}
    progress('Praveen','Bypassed')
    if not settings.bpdr_bypass:
        run['stages']['Praveen']['status']='Unavailable';save(run,folder)
        raise ValueError('BPDR requires the complete owner-exported local assets; enable explicit bypass to demonstrate other stages.')
    try:
        result,features,audio=stage('Vishwa',lambda:vishwa.analyze(source))
        run['vishwa']=result
        sf.write(folder/'prepared.wav',audio.samples,audio.sample_rate)
        np.savez_compressed(folder/'features.npz',**features)
        run['artifacts'].update(prepared='prepared.wav',features='features.npz')
        try: run['pilot_models']=vishwa.pilot_predictions(features)
        except Exception as exc: run['warnings'].append('Pilot comparison unavailable: '+str(exc))
        validate_notes(result['notes'],info.duration)
        melody,midi=rendering.melody(result['notes'],info.duration)
        sf.write(folder/'melody.wav',melody,rendering.SR);midi.write(str(folder/'melody.mid'))
        run['artifacts'].update(melody='melody.wav',melody_midi='melody.mid')
        beats=beat_grid(info.duration,settings);run['beats']=beats
        run['warnings'].append(f"User-configured grid covers {beats[0]['start_s']:.3f}–{beats[-1]['end_s']:.3f}s; uncovered edges have no chord predictions.")
        run['pasan']=stage('Pasan',lambda:pasan.predict(result['notes'],beats,settings))
        run['rahul']=stage('Rahul',lambda:rahul.predict(result['notes'],run['pasan']['chords'],settings))
        for name,key in [('basic','pasan'),('refined','rahul')]:
            rendered,midi,unsupported=rendering.chords(run[key]['chords'],info.duration)
            sf.write(folder/(name+'.wav'),rendered,rendering.SR);midi.write(str(folder/(name+'.mid')))
            run['artifacts'][name]=name+'.wav';run['artifacts'][name+'_midi']=name+'.mid'
            if unsupported: run['warnings'].append(f'{name}: unrenderable chord labels kept in predictions but silent in audio: {unsupported}')
            if name=='refined':
                sf.write(folder/'combined.wav',rendering.mix(audio.samples,rendered),rendering.SR)
                run['artifacts']['combined']='combined.wav'
    except Exception as exc:
        run['error']=f'{type(exc).__name__}: {exc}'
    run['status']='Complete with BPDR bypass' if 'combined' in run['artifacts'] else 'Partial / blocked'
    for artifact in run['artifacts'].values():
        if not (folder/artifact).is_file():raise ValueError('Missing generated artifact '+artifact)
    run['artifact_hashes']={name:sha(folder/name) for name in run['artifacts'].values()}
    save(run,folder)
    return folder,run
