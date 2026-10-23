"""Real-data forward/backward/ONE-step demonstration; no training loop or metrics."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from experiments.rahul import config as paths
from experiments.rahul.src.transformer_dataset import SavedSequenceDataset, make_dataloader, collate_sequences
from experiments.rahul.src.model import DatasetSpec, ModelConfig, ChordRefinementTransformer, masked_cross_entropy

DEFAULT_DATASET = paths.COMPONENT_DIR/'outputs/model_preprocessing/final_all_tracks_quality_v6_multibar'


def mixed_length_examples(folder: Path, lengths=(12, 16, 20)):
    """Find actual saved examples from offsets, without manufacturing beat rows."""
    index = json.loads((folder/'train_sequences.json').read_text(encoding='utf-8'))
    found = {}
    global_offset = 0
    for entry in index['shards']:
        path = folder/entry['path']
        with np.load(path, allow_pickle=False) as shard:
            sizes = np.diff(shard['sequence_offsets'])
        needed = [length for length in lengths if length not in found and (sizes == length).any()]
        if needed:
            dataset = SavedSequenceDataset(path)
            for length in needed:
                local = int(np.flatnonzero(sizes == length)[0])
                item = dataset[local]
                item['metadata']['sequence_index'] = global_offset + local
                found[length] = item
        if len(found) == len(lengths):
            return [found[length] for length in lengths]
        global_offset += entry['sequences']
    raise ValueError(f'Requested real sequence lengths not found: {set(lengths)-set(found)}')


def run_demo(folder=DEFAULT_DATASET, device='cpu', seed=42):
    folder = Path(folder)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    spec = DatasetSpec.from_folder(folder)
    settings = ModelConfig()
    model = ChordRefinementTransformer(spec, settings).to(device)
    dataset = SavedSequenceDataset(folder/'train_sequences.json')
    batch = next(iter(make_dataloader(dataset, batch_size=4, shuffle=False, seed=seed)))
    inputs = {k: v.to(device) for k, v in batch['inputs'].items() if k != 'melody_source_ids'}
    targets = {k: v.to(device) for k, v in batch['targets'].items()}
    model.eval()
    with torch.no_grad():
        logits, details = model(inputs, return_details=True)
        loss = masked_cross_entropy(logits, targets, inputs['beat_attention_mask'])
        qkv = model.layers[0].attention.project_qkv(model.layers[0].norm1(details['fused']))
    predictions = model.predict_topk(inputs)
    valid = targets['target_loss_mask'] & inputs['beat_attention_mask']
    chosen = torch.nonzero(valid)[:3]
    samples = [{'batch': int(b), 'position': int(t),
                'track_id': batch['metadata'][int(b)]['track_id'],
                'target_id': int(targets['target_chord_ids'][b, t]),
                'target_label': spec.target_labels[int(targets['target_chord_ids'][b, t])],
                'first_eight_logits': logits[b, t, :8].cpu().tolist(),
                'untrained_top3': predictions[int(b)][int(t)]} for b, t in chosen]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step_logits = model(inputs)
    step_loss = masked_cross_entropy(step_logits, targets, inputs['beat_attention_mask'])
    step_loss.backward()
    gradients = {k: float(p.grad.norm()) if p.grad is not None else None for k, p in model.named_parameters()}
    all_gradients_finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    if not all_gradients_finite:
        raise AssertionError('Missing or non-finite parameter gradients')
    optimizer.step()
    changed = [k for k, p in model.named_parameters() if not torch.equal(before[k], p)]
    if not changed:
        raise AssertionError('Optimizer updated no parameters')
    mixed = collate_sequences(mixed_length_examples(folder))
    model.eval()
    with torch.no_grad():
        mixed_logits = model({k: v.to(device) for k, v in mixed['inputs'].items() if k != 'melody_source_ids'})
    report = {
        'purpose': 'Untrained architecture debugging; exactly one optimizer step, no accuracy evaluation',
        'torch_version': str(torch.__version__), 'device': str(device), 'seed': seed,
        'config': asdict(settings), 'vocabulary_hashes': spec.vocabulary_hashes,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'target_classes': spec.num_target_classes, 'genre_dim': spec.genre_dim,
        'note_capacity': spec.note_capacity,
        'input_shapes': {k: list(v.shape) for k, v in inputs.items()},
        'input_dtypes': {k: str(v.dtype) for k, v in inputs.items()},
        'target_shapes': {k: list(v.shape) for k, v in targets.items()},
        'feature_shapes': {k: list(v.shape) for k, v in details['features'].items()},
        'fused_shape': list(details['fused'].shape), 'qkv_shapes': [list(v.shape) for v in qkv],
        'attention_shape': list(details['attention'][0].shape), 'logits_shape': list(logits.shape),
        'real_lengths': inputs['beat_attention_mask'].sum(-1).cpu().tolist(),
        'valid_targets_in_batch': int(valid.sum()), 'initial_eval_loss': float(loss),
        'dropout_enabled_step_loss': float(step_loss.detach()),
        'all_gradients_finite': all_gradients_finite, 'gradient_norms': gradients,
        'changed_parameter_tensors': len(changed), 'optimizer_steps': 1,
        'all_logits_finite': bool(torch.isfinite(logits).all()),
        'mixed_lengths': mixed['inputs']['beat_attention_mask'].sum(-1).tolist(),
        'mixed_logits_shape': list(mixed_logits.shape),
        'mixed_padded_positions': int((~mixed['inputs']['beat_attention_mask']).sum()),
        'sample_untrained_predictions_before_step': samples,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, default=paths.COMPONENT_DIR/'outputs/transformer_architecture/smoke_report.json')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    # Debug outputs must never be written into the immutable dataset snapshot.
    if args.output.resolve().is_relative_to(args.dataset.resolve()):
        parser.error('Choose a report output outside the dataset snapshot')
    report = run_demo(args.dataset, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('gradient_norms', 'input_dtypes')}, indent=2))
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
