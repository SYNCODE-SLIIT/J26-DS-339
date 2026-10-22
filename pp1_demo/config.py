from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'pp1_demo'
RUNTIME = APP / 'runtime'
PASAN = ROOT / 'experiments/pasan/checkpoints/phase3_temporal_transformer_weighted_best.pt'
PASAN_NOTEBOOK = ROOT / 'experiments/pasan/notebooks/03_phase3_transformer_development.ipynb'
RAHUL = ROOT / 'experiments/rahul/outputs/multitask_transformer/multitask_20261020_164527_051309/best_multitask.pt'
VOCAB = ROOT / 'experiments/rahul/outputs/model_preprocessing/final_all_tracks_quality_v6_multibar'
BPDR = ROOT / 'experiments/bpdr_input/BPDR_Workspace'
PC_NAMES = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
