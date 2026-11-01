"""Actual-WAV replay on historical development; never accesses reserved audio."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.experiments.learning import joint_transformer_training as training
from bpdr_input.modeling.joint_transformer import load
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair.held_voice_policy import HeldPolicy
from bpdr_input.reporting.represented_singer_gates import assess
import evaluate_held_voice_transformer as workflow


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=WORKSPACE/'results/transformer_usability/2026-10-28_v8_joint')
    args = parser.parse_args()
    root = args.run_root.resolve()
    checkpoint = root/'final/seed_339/checkpoint.pt'
    selection = ex.read(root/'locked_selection.json')
    if not selection['prediction_gate_passed']:
        raise ValueError('Development replay waits for the declared prediction gate')
    source = root.parent/'2026-10-28_v7_guard/development'
    rows = ex.read(source/'pairs.json')
    sources = [ex.read(p)['source'] for p in sorted(source.glob('source_*.json'))]
    model, payload = load(checkpoint)
    assert payload['arm'] == selection['arm']
    assert_independent(payload['training_groups'], payload['training_sources'], rows)
    folder = root/'development'
    folder.mkdir(exist_ok=True)
    if (folder/'evidence.json').exists():
        raise ValueError('Completed evaluation exists; preserve it and use an explicitly declared replay folder')
    policy = HeldPolicy(version='held-voice-input-policy-2', activation_cents=8.,
                        region_split_cents=70., minimum_step_fit_gain=.75)
    hashes = ex.input_hashes(rows)
    for p in sources:
        if 'working_audio' in p:
            path = Path(p['working_audio'])
            for f in (path, path.parent/'original_features.npz'):
                hashes[str(f)] = ex.sha(f)
    ex.write(folder/'evaluation_declaration.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
             role='Previously inspected development, not fresh final confirmation',
             checkpoint_sha256=ex.sha(checkpoint), pairs_manifest=str(source/'pairs.json'),
             policy=policy.__dict__, policy_selection='Fixed preceding V7 policy; not retuned on new predictions',
             input_hashes=hashes, evaluator_source_sha256=ex.sha(Path(__file__))))
    ex.write(folder/'pairs.json', rows)
    pairs = held.prepared(rows)
    result = training.evaluate(model, payload, pairs, folder, policy, True)
    workflow.complete_trace(result, pairs, folder, policy)
    workflow.training = training
    controls = workflow.original_controls(model, sources, policy, folder)
    gates = assess(result, controls)
    ex.write(folder/'gates.json', gates)
    assert all(ex.sha(p) == h for p, h in hashes.items())
    assert ex.sha(checkpoint) == ex.read(folder/'evaluation_declaration.json')['checkpoint_sha256']
    ex.write(folder/'evidence.json', dict(checkpoint_sha256=ex.sha(checkpoint),
             accuracy_sha256=ex.sha(folder/'accuracy.json'), controls_sha256=ex.sha(folder/'original_controls.json'),
             gates=gates, historical_development=True, fresh_confirmation=False,
             demo_activated=False, research_validation_passed=False))
    print('JOINT_DEVELOPMENT', gates, flush=True)


if __name__ == '__main__':
    main()
