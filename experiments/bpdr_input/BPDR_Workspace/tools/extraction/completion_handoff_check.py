"""One fixed-GAME diagnostic of the real-humming noise preview, CPU only.

Comparison with GAME on the pre-noise baseline is extractor consistency, not
human transcription accuracy or evidence for all novices. No shared pipeline.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_handoff import agreement,read_game

COMPONENT=next(p for p in Path(__file__).resolve().parents if p.name == 'bpdr_input' and (p/'pyproject.toml').is_file())
ASSETS=COMPONENT/'checkpoints/melody-extraction'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def main():
    live=json.loads((COMPONENT/'runs/humming_noise_live_check.json').read_text());report=live['report']
    assert report['outcome']=='NOISE_PREVIEW_PASSED' and report['handoff']['status']=='ACCEPT_UNCHANGED'
    case=Path(live['source_study_case']['directory'])
    out=COMPONENT/'runs/completion_handoff_check'/uuid.uuid4().hex;inputs=out/'inputs';inputs.mkdir(parents=True)
    sources={'baseline':case/'baseline.wav','baseline_repeat':case/'baseline.wav','noisy':case/'noisy.wav',
             'prepared_preview':Path(report['noise_candidate']['path'])}
    for name,p in sources.items():shutil.copy2(p,inputs/(name+'.wav'))
    model=ASSETS/'models/GAME-1.0-medium/model.pt';tool=ASSETS/'tools/GAME/infer.py'
    declaration=dict(version='fixed-game-humming-noise-check-1',sources={k:dict(path=str(p),sha256=sha(p)) for k,p in sources.items()},
        checkpoint_sha256=sha(model),tool_sha256=sha(tool),seed=339,device='CPU; two threads',
        interpretation='One training-source controlled mixture; GAME consistency against baseline, not human label accuracy',
        preview_not_delivered_downstream=True,shared_pipeline_execution=False,model_training=False,final_test_audio_accessed=False)
    write(out/'declaration.json',declaration)
    (out/'executed_completion_handoff_check.py').write_bytes(Path(__file__).read_bytes())
    code="import sys,random,numpy as np,torch,runpy;torch.set_num_threads(2);torch.manual_seed(339);np.random.seed(339);random.seed(339);sys.argv=sys.argv[1:];runpy.run_path('infer.py',run_name='__main__')"
    command=[str(ASSETS/'.venv-game/Scripts/python.exe'),'-c',code,'infer.py','extract',str(inputs),'-m',str(model),
             '--glob','*.wav','--output-formats','mid,csv','--pitch-format','number','--output-dir',str(out/'notes')]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    print('GAME_CHECK',out,flush=True);started=time.monotonic()
    result=subprocess.run(command,cwd=tool.parent,capture_output=True,text=True,env=env)
    (out/'stdout.log').write_text(result.stdout,encoding='utf-8');(out/'stderr.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Local GAME failed; inspect the saved stderr log')
    notes={k:read_game(out/'notes'/(k+'.csv')) for k in sources}
    scores={k:agreement(notes['baseline'],notes[k]) for k in ['baseline_repeat','noisy','prepared_preview']}
    assert sha(model)==declaration['checkpoint_sha256'] and sha(tool)==declaration['tool_sha256']
    write(out/'evidence.json',dict(scores=scores,elapsed_seconds=time.monotonic()-started,checkpoint_unchanged=True,
        limitation='Single correlated diagnostic with stochastic boundary extraction; no general downstream or musical-quality claim'))
    write(out.parent/'current.json',dict(path=str(out)));print('GAME_CHECK_COMPLETE',json.dumps(scores),flush=True)


if __name__=='__main__':main()
