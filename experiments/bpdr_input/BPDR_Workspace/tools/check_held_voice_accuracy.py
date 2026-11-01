"""Repeat accuracy of a locked held-voice checkpoint without refitting."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import uuid
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
sys.path.insert(0,str(WORKSPACE/'tools'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.experiments.learning import held_voice_training as training
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.modeling.held_voice_transformer import load
from bpdr_input.repair.held_voice_policy import HeldPolicy
from bpdr_input.repair.held_voice_runtime import load_model
from evaluate_held_voice_transformer import original_controls,gates,complete_trace


def check(root=None,*,latest=False,render=False,output_root=None,partition='final_evaluation'):
    torch.set_num_threads(2)
    if latest:
        model,payload,card=load_model();root=Path(card['run_root']);policy=HeldPolicy(**card['policy'])
    else:
        root=Path(root).resolve()
        if ex.read(root/'declaration.json').get('model_family')=='step_context':
            from bpdr_input.modeling.step_transformer import load as load_step
            model,payload=load_step(root/'final/seed_339/checkpoint.pt')
        else:model,payload=load(root/'final/seed_339/checkpoint.pt')
        policy=HeldPolicy(**ex.read(root/'locked_policy.json')['selected']['policy'])
    evaluator=training
    from bpdr_input.experiments.learning import step_transformer_training
    if ex.read(root/'declaration.json').get('model_family')=='step_context':evaluator=step_transformer_training
    source_folder=root/partition;manifest=source_folder/'pairs.json';rows=ex.read(manifest)
    assert_independent(payload['training_groups'],payload['training_sources'],rows)
    hashes=ex.input_hashes(rows);checkpoint=root/'final/seed_339/checkpoint.pt';before=ex.sha(checkpoint)
    output=Path(output_root or WORKSPACE/'results/transformer_usability/accuracy_checks')/uuid.uuid4().hex
    output.mkdir(parents=True)
    ex.write(output/'evaluation_declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),checkpoint=str(checkpoint),checkpoint_sha256=before,pairs_manifest=str(manifest),
        dataset_role='Repeat of an already inspected evaluation partition; not a new final test',source_partition=str(source_folder),render=render,policy=policy.__dict__,input_hashes=hashes))
    prepared=evaluator.prepared(rows)
    result=evaluator.evaluate(model,payload,prepared,output,policy,render)
    complete_trace(result,prepared,output,policy)
    if render:
        import evaluate_held_voice_transformer as evaluation_module
        evaluation_module.training=evaluator
        sources=[ex.read(p)['source'] for p in sorted(source_folder.glob('source_*.json'))]
        controls=original_controls(model,sources,policy,output)
        assess=gates
        if (root/'gate_protocol_amendment.json').exists():
            from bpdr_input.reporting.represented_singer_gates import assess
        ex.write(output/'gates.json',assess(result,controls))
    assert ex.sha(checkpoint)==before
    assert all(ex.sha(p)==h for p,h in hashes.items())
    print('HELD_VOICE_ACCURACY_REPORT',output,flush=True)
    print('RAW_MAE_CENTS',result['summary']['raw_mae_cents'],flush=True)
    if render:print('DELIVERED_PRACTICAL_GATES',ex.read(output/'gates.json'),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--latest',action='store_true');group.add_argument('--run-root',type=Path)
    parser.add_argument('--render',action='store_true');parser.add_argument('--partition',choices=['development','final_evaluation'],default='final_evaluation')
    args=parser.parse_args();check(args.run_root,latest=args.latest,render=args.render,partition=args.partition)
