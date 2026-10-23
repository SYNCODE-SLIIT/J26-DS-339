"""Verify the running local website's model API, media ranges and artifact boundary."""
import argparse,json,time,urllib.request,urllib.error

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8765)
args=parser.parse_args();base=f'http://127.0.0.1:{args.port}'

def call(path,data=None,headers=None):
    request=urllib.request.Request(base+path,data=json.dumps(data).encode() if data is not None else None,
                                  headers={'Content-Type':'application/json',**(headers or {})})
    try:
        with urllib.request.urlopen(request,timeout=10) as r:return r.status,r.headers,r.read()
    except urllib.error.HTTPError as e:return e.code,e.headers,e.read()

assert call('/api/health')[0]==200
status,_,body=call('/api/catalog');library=json.loads(body)
assert status==200 and len(library['tracks'])>0
track=library['tracks'][0]
assert 'waveform' not in track
status,_,body=call('/api/tracks/'+track['id'])
preview=json.loads(body)
assert status==200 and preview['waveform'] and preview['notes']
assert call('/api/tracks/not-a-recording')[0]==404
status,headers,body=call(track['audio_url'],headers={'Range':'bytes=0-63'})
assert status==206 and len(body)==64 and body[:4]==b'RIFF'
assert headers['Content-Range'].startswith('bytes 0-63/')
assert call('/api/runs',data=[])[0]==400
status,_,body=call('/api/runs',data={'track_id':track['id']})
assert status==202,body
run_id=json.loads(body)['id']
assert call('/api/runs',data={'track_id':track['id']})[0]==409
seen=set()
for attempt in range(720):
    status,_,body=call('/api/runs/'+run_id);run=json.loads(body)
    seen.add(run.get('active_stage'))
    if run['status'] in ('complete','failed'):break
    time.sleep(.25)
assert run['status']=='complete',run.get('error')
assert run['praveen']['chart']['model_ran'] is True
assert run['settings']['rahul_mode']=='major'
assert run['provenance']['genre']=='dataset_metadata'
assert run['pasan']['metadata']['epoch']==9
assert run['pasan']['window_bars']==16
assert len(run['pasan']['chords'])==len(run['rahul']['chords'])>0
assert {'vishwa','context','pasan','rahul'}<=seen,seen
assert run['stages']['vishwa']['status']=='complete'
assert set(run['vishwa']['pilot_models'])=={'cnn','cnn_transformer'}
assert run['vishwa']['input_audio_sha256']==run['artifact_hashes']['accepted']
assert run['context']['notes']==run['vishwa']['notes']
assert any(n['context_provenance']=='live_accepted_audio' for n in run['context']['inference_notes'])
assert run['pasan']['input_summary']['notes']==len(run['context']['inference_notes'])
for name in run['artifacts'].values():
    status,_,body=call('/api/artifacts/'+run_id+'/'+name)
    assert status==200 and len(body)>0,(name,status)
assert call('/api/artifacts/'+run_id+'/../../config.py')[0] in (403,404)
assert call('/api/runs',data={'track_id':'../../file'})[0]==400
print(json.dumps({'status':'PASS','run_id':run_id,'pipeline_seconds':run['seconds'],
                  'visible_handoffs':sorted(s for s in seen if s),'artifacts':list(run['artifacts']),
                  'model_stages':{k:run['stages'][k]['status'] for k in ['praveen','vishwa','pasan','rahul']}},indent=2))
