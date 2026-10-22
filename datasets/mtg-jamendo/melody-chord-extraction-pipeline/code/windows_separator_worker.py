"""Isolated PyTorch BS-RoFormer vocal separator for Windows (CPU or supported GPU)."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    request = json.load(sys.stdin)
    from audio_separator.separator import Separator
    import torch

    model = Path(request['model'])
    source = Path(request['source'])
    destination = Path(request['destination'])
    if not model.is_file() or not source.is_file():
        raise FileNotFoundError('Windows separator input or pinned checkpoint missing')
    destination.parent.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    separator = Separator(output_dir=str(destination.parent), model_file_dir=str(model.parent),
                          output_format='WAV', output_single_stem='Vocals')
    separator.load_model(model.name)
    found = separator.separate(str(source), {'Vocals': destination.stem})
    candidates = [Path(path) if Path(path).is_absolute() else destination.parent / path
                  for path in found]
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one vocal stem, got {found}')
    if candidates[0] != destination:
        candidates[0].replace(destination)
    print(json.dumps({'backend': 'audio-separator 0.47.0 PyTorch', 'device': device,
                      'precision': 'package_default', 'output': str(destination)}))


if __name__ == '__main__':
    main()
