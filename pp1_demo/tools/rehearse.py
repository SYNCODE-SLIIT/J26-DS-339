"""Rehearse a selected local vocal or the complete library without the website."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pp1_demo.config import configure_runtime
configure_runtime()
import argparse
import json
from pp1_demo.dataset import catalogue
from pp1_demo.orchestrator import new_run,execute

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--track',default='track_0000382')
parser.add_argument('--all',action='store_true')
args=parser.parse_args()
tracks=[x['id'] for x in catalogue()] if args.all else [args.track]
failed=False
for track in tracks:
    folder,run=new_run(track)
    execute(folder,run,lambda r:print(r['active_stage'],r['status'],r['progress'],flush=True))
    print(json.dumps({'folder':str(folder),'status':run['status'],'seconds':run['seconds'],
                      'bpdr':run.get('praveen',{}).get('outcome'),
                      'chords':len(run.get('rahul',{}).get('chords',[])),
                      'error':run.get('error')},indent=2))
    failed |= run['status']!='complete'
raise SystemExit(1 if failed else 0)
