"""Portable weights plus architecture and saved-vocabulary identity."""
from dataclasses import asdict
from pathlib import Path
import torch
from .config import DatasetSpec, ModelConfig
from .chord_refinement_transformer import ChordRefinementTransformer


def save_checkpoint(path: str | Path, model: ChordRefinementTransformer) -> None:
    torch.save({'config': asdict(model.config), 'spec': asdict(model.spec),
                'state_dict': model.state_dict()}, path)


def load_checkpoint(path: str | Path, dataset_folder: str | Path, device='cpu') -> ChordRefinementTransformer:
    saved = torch.load(path, map_location=device, weights_only=True)
    spec = DatasetSpec.from_folder(dataset_folder)
    if saved['spec'] != asdict(spec):
        raise ValueError('Checkpoint and dataset vocabularies differ')
    model = ChordRefinementTransformer(spec, ModelConfig(**saved['config'])).to(device)
    model.load_state_dict(saved['state_dict'])
    return model.eval()
