"""Portable real-audio inference check; no dataset or other component required."""
import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pp1_demo.config import configure_runtime,RUNTIME
configure_runtime()
from pp1_demo.adapters.vishwa import predict
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audio',type=Path,required=True);args=p.parse_args()
folder=RUNTIME/'vishwa_standalone';folder.mkdir(exist_ok=True)
r=predict(args.audio,folder)
(folder/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False))
print(json.dumps({'result':str(folder/'result.json'),'notes':len(r['notes']),'checkpoints':list(r['pilot_models']),'downstream_source':r['downstream_source']},indent=2))
