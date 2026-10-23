"""An inspectable, sequential pipeline using real models and saved feature context."""
import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from pp1_demo.config import RUNTIME, PASAN, RAHUL, VOCAB
from pp1_demo.contracts import Settings, sha
from pp1_demo.dataset import context, crop_rows, verify_audio
from pp1_demo.adapters import bpdr, pasan, rahul
from pp1_demo import rendering

STAGES = ['praveen', 'context', 'pasan', 'rahul', 'render']


def save(run, folder):
    temporary = folder / 'manifest.tmp'
    temporary.write_text(json.dumps(run, indent=2, allow_nan=False))
    # Windows readers can briefly prevent replacement while polling the manifest.
    # Keep the atomic write and allow the reader to close its file handle.
    for attempt in range(20):
        try:
            temporary.replace(folder / 'manifest.json')
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.025)


def new_run(track_id, genre=None):
    c = context(track_id)
    folder = RUNTIME / 'runs' / ('r' + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True)
    run = {'id':folder.name, 'track_id':track_id, 'created_utc':datetime.now(timezone.utc).isoformat(),
           'status':'queued', 'active_stage':None, 'duration_s':c['duration_s'], 'progress':0,
           'stages':{k:{'status':'pending'} for k in STAGES}, 'events':[], 'artifacts':{},
           'warnings':[], 'settings':{'genre':genre or c['genre'], 'rahul_mode':'major',
                                     'mode_source':'demo_constant','reference_key':c['reference_key']},
           'provenance':{'vishwa':'deferred; saved dataset features used', 'genre':c['genre_source'] if not genre or genre==c['genre'] else 'demo_selection',
                         'melody':'saved_extraction','beats':'saved_extraction','key':'saved_melody_key',
                         'bpdr':'live_model','pasan':'live_model','rahul':'live_model'},
           'models':{'pasan':PASAN.name,'rahul':RAHUL.name}, 'audio_sha256':c['audio_sha256']}
    save(run, folder)
    return folder, run


def execute(folder, run, progress=lambda run:None, handoff_pause_s=0):
    folder=Path(folder)
    started=time.perf_counter()
    c=context(run['track_id'])
    def stage(name, title, fn):
        run['active_stage']=name
        run['stages'][name]={'status':'running','started_at':time.time(),'title':title}
        run['events'].append({'stage':name,'text':title,'state':'running','elapsed_s':round(time.perf_counter()-started,2)})
        save(run, folder);progress(run)
        tick=time.perf_counter()
        try:
            value=fn()
            run[name]=value
            run['stages'][name].update(status='complete',seconds=round(time.perf_counter()-tick,3))
            run['progress']=round(100*sum(x['status']=='complete' for x in run['stages'].values())/len(STAGES))
            run['events'].append({'stage':name,'text':title,'state':'complete','elapsed_s':round(time.perf_counter()-started,2)})
            save(run, folder);progress(run)
            # Briefly present completed handoffs in the website. Model timings
            # above exclude this display hold; no simulated computation/results.
            if handoff_pause_s and name not in ('praveen','render'):
                time.sleep(handoff_pause_s)
            return value
        except Exception as exc:
            run['stages'][name].update(status='failed',error=f'{type(exc).__name__}: {exc}')
            raise
    run['status']='running'
    try:
        verify_audio(c)
        shutil.copy2(c['audio_path'],folder/'input.wav')
        run['artifacts']['input']='input.wav'
        def assess():
            result=bpdr.predict(folder/'input.wav',folder)
            shutil.copy2(result['accepted_audio'],folder/'accepted.wav')
            run['artifacts']['accepted']='accepted.wav'
            if result.get('preview'):
                shutil.copy2(result['preview'],folder/'repair_preview.wav')
                run['artifacts']['repair_preview']='repair_preview.wav'
            return result
        result=stage('praveen','Praveen · analyse the vocal and run learned repair',assess)
        if result.get('reason'):
            run['warnings'].append('BPDR kept the accepted original; optional repair preview: '+result['reason'])
        def feature_bridge():
            hashes={name:sha(path) for name,path in c['files'].items()}
            return {'source':'Saved extraction · Vishwa stage deferred',
                    'notes':c['display_notes'],'beats':c['display_beats'],'key':c['key'],
                    'reference_key':c['reference_key'],'genre':run['settings']['genre'],
                    'genre_source':run['provenance']['genre'],'rahul_mode':'major',
                    'crop_offset_s':c['crop_offset_s'],'context_start_s':c['context_start_s'],
                    'context_end_s':c['context_end_s'],'context_notes':len(c['notes']),
                    'context_beats':len(c['beats']),'context_bars':16,'source_hashes':hashes,
                    'note':'Full sixteen-bar saved context is used for inference. Timelines and playback show only this vocal excerpt.'}
        stage('context','Dataset bridge · align saved notes, key and beats',feature_bridge)
        settings=Settings(tonic_pc=c['reference_pc'],genre=run['settings']['genre'],bpdr_bypass=False)
        def align(result):
            result['context_chords']=result['chords']
            result['chords']=crop_rows(result['context_chords'],c['visible_start_s'],c['duration_s'])
            return result
        full_pasan=stage('pasan','Pasan · Temporal Transformer harmonic planning',lambda:align(pasan.predict(c['notes'],c['beats'],settings)))
        full_rahul=stage('rahul','Rahul · multi-task chord refinement',lambda:align(rahul.predict(c['notes'],full_pasan['context_chords'],settings)))
        def render():
            audio,sr=sf.read(folder/'accepted.wav',dtype='float32')
            if sr!=rendering.SR:raise ValueError('Unexpected accepted audio sample rate')
            melody,midi=rendering.melody(c['display_notes'],c['duration_s'])
            sf.write(folder/'melody.wav',melody,rendering.SR);midi.write(str(folder/'melody.mid'))
            run['artifacts'].update(melody='melody.wav',melody_midi='melody.mid')
            unsupported={}
            for name,owner in [('basic','pasan'),('refined','rahul')]:
                y,midi,labels=rendering.chords(run[owner]['chords'],c['duration_s'])
                sf.write(folder/(name+'.wav'),y,rendering.SR);midi.write(str(folder/(name+'.mid')))
                run['artifacts'][name]=name+'.wav';run['artifacts'][name+'_midi']=name+'.mid'
                unsupported[name]=labels
                if labels:run['warnings'].append('Unrendered '+name+' labels remain visible: '+', '.join(labels))
                if name=='refined':
                    sf.write(folder/'combined.wav',rendering.mix(audio,y),rendering.SR)
                    run['artifacts']['combined']='combined.wav'
            return {'source':'Synthesized instrument playback, mixed with accepted voice',
                    'unsupported_labels':unsupported,'melody_notes':len(c['display_notes']),
                    'basic_chords':len(run['pasan']['chords']),'refined_chords':len(run['rahul']['chords'])}
        stage('render','Render · build listening previews and MIDI',render)
        run['status']='complete';run['active_stage']=None
        run['artifact_hashes']={name:sha(folder/path) for name,path in run['artifacts'].items()}
    except Exception as exc:
        run['status']='failed';run['error']=f'{type(exc).__name__}: {exc}'
    run['seconds']=round(time.perf_counter()-started,3)
    save(run, folder);progress(run)
    return run
