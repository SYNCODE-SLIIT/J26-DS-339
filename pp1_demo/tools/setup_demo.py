"""Explicit uv setup for the existing demo, without root uv sync or dataset downloads."""
import argparse,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--install',action='store_true')
args=parser.parse_args()
uv=shutil.which('uv') or str(Path.home()/'.local/bin/uv')
relative='Scripts/python.exe' if os.name=='nt' else 'bin/python'
main=ROOT/'.venv'/relative
bpdr=ROOT/'pp1_demo/runtime/bpdr_env'/relative
if args.install:
    if not main.is_file():subprocess.run([uv,'venv','--python','3.12',str(ROOT/'.venv')],check=True)
    subprocess.run([uv,'pip','install','--python',str(main),'-r',str(ROOT/'pp1_demo/requirements-inference.txt')],check=True)
    if not bpdr.is_file():subprocess.run([uv,'venv','--python',str(main),'--system-site-packages',str(bpdr.parents[1])],check=True)
    subprocess.run([uv,'pip','install','--python',str(bpdr),'-r',str(ROOT/'pp1_demo/requirements-bpdr.txt')],check=True)
print('Main runtime:',main,'present:',main.is_file())
print('BPDR runtime:',bpdr,'present:',bpdr.is_file())
print('Next: uv run --no-sync python -B -m pp1_demo.app')
print('The library UI also requires locally extracted Jamendo audio/context; no dataset files are downloaded here.')
