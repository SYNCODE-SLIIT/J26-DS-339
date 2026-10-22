"""Architecture settings and the immutable saved-dataset vocabulary contract."""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from experiments.rahul.src.encode_features import CATEGORICAL, FeatureEncoder


@dataclass(frozen=True)
class DatasetSpec:
    vocabulary_sizes: dict[str, int]
    target_labels: tuple[str, ...]
    genre_dim: int
    note_capacity: int
    pitch_vocab_size: int
    vocabulary_hashes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_folder(cls, folder: str | Path) -> 'DatasetSpec':
        folder = Path(folder)
        encoder = FeatureEncoder.load(folder)  # Validates both saved mappings.
        sizes = {key: len(encoder.vocabs[name].tokens) for key, name in CATEGORICAL.items()}
        # encode_features: round(MIDI in [0,127]) + 2; PAD=0, UNK=1.
        return cls(sizes, tuple(encoder.targets.tokens), len(encoder.genres),
                   encoder.max_notes, 128 + 2,
                   {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                    for name in ('feature_vocabularies.json', 'target_vocabulary.json')})

    @property
    def num_target_classes(self) -> int:
        return len(self.target_labels)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    chord_dim: int = 32
    melody_dim: int = 32
    harmonic_dim: int = 32
    genre_temporal_dim: int = 32
    use_melody: bool = True
    use_key_mode: bool = True
    use_harmonic: bool = True
    use_genre: bool = True
    use_temporal: bool = True
    use_absolute_timing: bool = False
    use_root_quality: bool = False
    use_neighbor_chords: bool = False
    use_pitch_class: bool = False
    use_continuous_pitch: bool = False
    positional_encoding: bool = True

    def __post_init__(self):
        dimensions = (self.d_model, self.num_heads, self.num_layers, self.d_ff,
                      self.chord_dim, self.melody_dim, self.harmonic_dim, self.genre_temporal_dim)
        if any(v < 1 for v in dimensions) or self.d_model % self.num_heads:
            raise ValueError('Positive dimensions required; d_model must divide evenly into heads')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0,1)')

    @classmethod
    def experiment(cls, name: str, **overrides) -> 'ModelConfig':
        features = {
            'baseline': (False, False, False, False, False),
            'A': (True, False, False, False, False),
            'B': (True, True, False, False, False),
            'C': (True, True, True, False, False),
            'full': (True, True, True, True, True),
        }
        if name not in features:
            raise ValueError(f'Unknown experiment: {name}')
        values = dict(zip(('use_melody', 'use_key_mode', 'use_harmonic', 'use_genre', 'use_temporal'), features[name]))
        return cls(**(values | overrides))
