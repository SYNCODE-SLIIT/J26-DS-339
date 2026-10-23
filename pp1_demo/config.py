"""Read-only model locations; every writable path belongs to this application."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'pp1_demo'
RUNTIME = APP / 'runtime'
PASAN = ROOT / 'experiments/pasan/checkpoints/phase3_temporal_transformer_weighted_latest.pt'
PASAN_NOTEBOOK = ROOT / 'experiments/pasan/notebooks/03_phase3_transformer_development.ipynb'
RAHUL = ROOT / 'experiments/rahul/outputs/multitask_transformer/multitask_20261020_164527_051309/best_multitask.pt'
VOCAB = ROOT / 'experiments/rahul/outputs/model_preprocessing/final_all_tracks_quality_v6_multibar'
BPDR = ROOT / 'experiments/bpdr_input/BPDR_Workspace'
BPDR_EXPORT = BPDR / 'exports/selected_run_inference'
DATASET = ROOT / 'datasets/mtg-jamendo'
EXTRACTIONS = DATASET / 'melody-chord-extraction-pipeline/outputs'
_legacy_bpdr = DATASET / 'melody-chord-extraction-pipeline/.venv-bpdr/bin/python'
_app_bpdr = RUNTIME / 'bpdr_env' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
BPDR_PYTHON = Path(os.environ.get('PP1_BPDR_PYTHON', str(_legacy_bpdr if _legacy_bpdr.is_file() else _app_bpdr)))
PC_NAMES = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']


def configure_runtime():
    """Call before scientific imports, including when a CLI is used directly."""
    import sys
    sys.dont_write_bytecode = True
    RUNTIME.mkdir(exist_ok=True)
    for name, relative in {'MPLCONFIGDIR':'cache/matplotlib', 'NUMBA_CACHE_DIR':'cache/numba',
                           'XDG_CACHE_HOME':'cache', 'TORCH_HOME':'cache/torch',
                           'PYTHONPYCACHEPREFIX':'cache/python'}.items():
        path = RUNTIME / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
