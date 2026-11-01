"""Same locked WAV evaluation, with the step-conditioned Transformer family."""
import argparse
from pathlib import Path
import sys
import torch
WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'));sys.path.insert(0,str(WORKSPACE/'tools'))
import evaluate_held_voice_transformer as workflow
from bpdr_input.experiments.learning import step_transformer_training
from bpdr_input.modeling.step_transformer import load
from bpdr_input.reporting.represented_singer_gates import assess

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['select-policy','development','final'])
    parser.add_argument('--run-root',type=Path,default=WORKSPACE/'results/transformer_usability/2026-10-28_v6')
    args=parser.parse_args();torch.set_num_threads(2)
    workflow.ROOT=args.run_root.resolve();workflow.training=step_transformer_training;workflow.load=load
    if (workflow.ROOT/'gate_protocol_amendment.json').exists():workflow.gates=assess
    if args.stage=='select-policy':workflow.choose_policy()
    else:workflow.evaluate_partition('final_reserved' if args.stage=='final' else 'development')
