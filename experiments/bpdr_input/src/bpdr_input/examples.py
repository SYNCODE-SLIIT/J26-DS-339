"""Phase 3 loader: four aligned views per audited pair, with research and engineering data kept apart."""
import json
from pathlib import Path
import numpy as np
import torch
from .laboratory import require_training_example, validate_pair_record

VIEWS = 'ABCD'
MODES = ('research', 'engineering')
EXPLORATORY = 'exploratory_not_research'
VOICED_COLUMN = 66


def data_status(mode):
    return 'reviewed_research' if mode == 'research' else EXPLORATORY


def read_pair(record, mode):
    directory = Path(record['input_directory'])
    inputs = {}
    for view in VIEWS:
        # Only the runtime-like input archives are read here; references never enter the features.
        with np.load(directory/(view+'_input.npz')) as archive:
            inputs[view] = archive['boundary_features_71'].astype(np.float32)
    with np.load(record['target_path']) as archive:
        mask = archive['common_reliable_mask'].astype(bool)
        detuning = -archive['correction_target_cents'].astype(np.float32)
    return dict(pair_id=f"{record['phrase_id']}|{record['cents']:+g}|{record['corruption']}", phrase_id=record['phrase_id'],
                singer_id=record['singer_id'], cents=float(record['cents']), corruption=record['corruption'], inputs=inputs,
                mask=mask, detuning=detuning, input_sha256=record['input_sha256'], data_status=data_status(mode))


def load_pairs(run_directory, mode):
    """research: every pair must pass the reviewed-and-audited gate. engineering: audited training pairs for debugging only."""
    if mode not in MODES:
        raise ValueError(f'Unknown mode: {mode}')
    records = json.loads((Path(run_directory)/'summary.json').read_text())['examples']
    pairs = []
    for record in records:
        if mode == 'research':
            require_training_example(record)
        elif record.get('split') != 'train' or not record.get('audit_passed'):
            continue
        else:
            validate_pair_record(record)
        pairs.append(read_pair(record, mode))
    if not pairs:
        raise ValueError(f'No usable {mode} pairs in {run_directory}')
    return pairs


class Normalizer:
    """Per-feature standardization; fit it on training pairs only."""
    def __init__(self, mean, scale):
        self.mean, self.scale = np.asarray(mean, dtype=np.float32), np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, pairs):
        frames = np.concatenate([pair['inputs'][view] for pair in pairs for view in VIEWS])
        scale = frames.std(axis=0)
        scale[scale < 1e-6] = 1.
        return cls(frames.mean(axis=0), scale)

    def __call__(self, features):
        width = features.shape[-1]
        return (features-self.mean[:width])/self.scale[:width]

    def state(self):
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist())


def collate(pairs, normalizer, features=71):
    """Pad to one length. The 70-feature control simply never sees the boundary column."""
    if features not in (70, 71):
        raise ValueError('The models take 70 or 71 input features')
    length = max(len(pair['mask']) for pair in pairs)

    def pad(array):
        return np.concatenate([array, np.zeros((length-len(array), *array.shape[1:]), dtype=array.dtype)])
    padding = np.stack([np.arange(length) >= len(pair['mask']) for pair in pairs])
    return dict(
        inputs={view: torch.from_numpy(np.stack([pad(normalizer(pair['inputs'][view][:, :features])) for pair in pairs])) for view in VIEWS},
        padding=torch.from_numpy(padding),
        mask=torch.from_numpy(np.stack([pad(pair['mask']) for pair in pairs])),
        detuning=torch.from_numpy(np.stack([pad(pair['detuning']) for pair in pairs])),
        # A/B share one waveform and C/D the other, so two voicing masks cover the four views.
        voiced={view: torch.from_numpy(np.stack([pad(pair['inputs'][view][:, VOICED_COLUMN] > .5) for pair in pairs])) for view in 'AC'},
        pair_ids=[pair['pair_id'] for pair in pairs], singer_ids=[pair['singer_id'] for pair in pairs])
