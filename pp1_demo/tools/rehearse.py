"""Run real audio through the demo; results stay ignored and local."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pp1_demo.orchestrator import run_pipeline
from pp1_demo.contracts import Settings
path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'datasets/humtrans/downloads/wav_data_sync_with_midi/F01_0001_0001_1_D.wav'
folder,run=run_pipeline(path,Settings(),lambda name,status:print(name,status,flush=True))
print(json.dumps({'folder':str(folder),'status':run['status'],'stages':run['stages'],
    'notes':len(run.get('vishwa',{}).get('notes',[])),
    'basic_chords':len(run.get('pasan',{}).get('chords',[])),
    'refined_chords':len(run.get('rahul',{}).get('chords',[])),
    'warnings':run['warnings'],'error':run.get('error')},indent=2))
if run['status']=='Partial / blocked':raise SystemExit(1)
