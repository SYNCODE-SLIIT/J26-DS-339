"""Replay saved out-of-fold predictions under one fixed policy, without refits."""
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.repair.held_voice_policy import HeldPolicy


if __name__ == '__main__':
    torch.set_num_threads(2)
    root = WORKSPACE/'results/transformer_usability/2026-10-28_v8_joint'
    pairs = held.prepared(ex.read(root/'train_pairs.json'))
    policy = HeldPolicy(version='held-voice-input-policy-2', activation_cents=8.,
                        region_split_cents=70., minimum_step_fit_gain=.75)
    for name, run, arm in [('previous_step', root.parent/'2026-10-28_v6', 'step_conditioned'),
                           ('joint', root, 'joint'), ('joint_augmented', root, 'joint_augmented')]:
        folder = root/'common_policy'/name
        folder.mkdir(parents=True, exist_ok=True)
        cache = {}
        decl = ex.read(run/'declaration.json')
        for fold, groups in enumerate(decl['folds'], 1):
            subset = [p for p in pairs if ex.group(p) in groups]
            with np.load(run/f'cross_validation/fold_{fold}/{arm}/validation/predictions.npz') as arrays:
                for index, p in enumerate(subset):
                    if 'excluded' not in p:
                        for view in ('A', 'C'):
                            cache[p['path_'+view]] = arrays[f'{index}_{view}'].copy()
        ex.write(folder/'evaluation_declaration.json', dict(
                 role='Previously saved training-side out-of-fold predictions; not a new test',
                 pairs_manifest=str(root/'train_pairs.json'),
                 policy=policy.__dict__, model_run=str(run), arm=arm,
                 raw_selection_unchanged=True, render=False,
                 caveat='Historical step fit budget 1000 updates; joint arms 1200. No isolated architecture causality claim.'))
        previous = held.predictions
        held.predictions = lambda model, p, view: cache[p['path_'+view]]
        try:
            result = held.evaluate(None, dict(seed=339), pairs, folder, policy, False)
        finally:
            held.predictions = previous
        print('COMMON_POLICY', name, result['summary']['permitted_mae_cents'], flush=True)
