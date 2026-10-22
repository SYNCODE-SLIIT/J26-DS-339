"""Read-only checkpoint comparison for notebook 11; no fitting or training."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from experiments.rahul.src.canonicalize_chords import ROOT_TO_PITCH_CLASS
from experiments.rahul.src.normalize_chords import parse_chord_label
from experiments.rahul.src.build_synthetic_dataset import context, simplify
from experiments.rahul.src.create_sequences import validate_splits
from experiments.rahul.src.encode_features import FeatureEncoder
from experiments.rahul.src.transformer_dataset import SavedSequenceDataset, make_dataloader
from experiments.rahul.src.model.config import DatasetSpec
from experiments.rahul.src.model.checkpoint import load_checkpoint
from experiments.rahul.src.model.multitask_transformer import load_multitask_checkpoint
from experiments.rahul.src.weighted_training import supervised_mask, metrics_from_confusion

MODELS = ('baseline', 'weighted', 'multitask')
COLORS = {'baseline': '#2878B5', 'weighted': '#E58B20', 'multitask': '#27916B', 'auxiliary': '#9562B0'}
DISPLAY = {'baseline': 'Baseline', 'weighted': 'Weighted', 'multitask': 'Multi-task', 'auxiliary': 'Auxiliary quality'}
INPUT_KEYS = ('basic_chord_ids', 'melody_pitch_ids', 'melody_timing', 'melody_note_mask', 'melody_silent',
    'key_ids', 'mode_ids', 'roman_ids', 'harmonic_function_ids', 'genre_features', 'beat_index',
    'bar_number', 'beat_position', 'event_duration', 'relative_bar_number', 'beat_timing_known_mask', 'beat_attention_mask')
TARGET_KEYS = ('target_chord_ids', 'target_loss_mask')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, Path): return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value): raise ValueError('Non-finite exported metric')
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def save_table(folder, name, rows):
    """Typed JSON plus numeric CSV; empty cells mean unavailable, never false zeros."""
    rows = clean(list(rows))
    save_json(Path(folder) / (name + '.json'), rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(folder) / (name + '.csv')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, separators=(',', ':'), ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)
    with path.open(encoding='utf-8', newline='') as stream:
        restored = list(csv.DictReader(stream))
    assert len(restored) == len(rows)
    for a, b in zip(rows, restored):
        for key, value in a.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert float(b[key]) == value, (name, key)
    return pd.DataFrame(rows)


def label_info(label):
    """Reuse the project parser and canonical root pitch classes; do not collapse qualities."""
    if label == 'N': return None, 'N', None
    parsed = parse_chord_label(label)
    if not parsed.parsed or not parsed.normalized: return None, label, None
    return ROOT_TO_PITCH_CLASS.get(parsed.root), parsed.quality, parsed.bass_degree


# Explicit diagnostic chord-tone dictionary, not a replacement label parser.
INTERVALS = {'maj': (0, 4, 7), 'min': (0, 3, 7), '7': (0, 4, 7, 10), 'maj7': (0, 4, 7, 11),
    'min7': (0, 3, 7, 10), 'dim': (0, 3, 6), 'dim7': (0, 3, 6, 9), 'hdim7': (0, 3, 6, 10),
    'aug': (0, 4, 8), 'sus2': (0, 2, 7), 'sus4': (0, 5, 7), '6': (0, 4, 7, 9),
    'min6': (0, 3, 7, 9), '9': (0, 2, 4, 7, 10), 'maj9': (0, 2, 4, 7, 11),
    'min9': (0, 2, 3, 7, 10), '11': (0, 2, 4, 5, 7, 10), '13': (0, 2, 4, 5, 7, 9, 10), 'add9': (0, 2, 4, 7)}


def chord_tones(label):
    root, quality, bass = label_info(label)
    if root is None or quality not in INTERVALS: return None
    tones = {(root + offset) % 12 for offset in INTERVALS[quality]}
    if bass:
        degree = int(bass.lstrip('#b'))
        if degree < 1: return None
        alteration = 1 if bass.startswith('#') else -1 if bass.startswith('b') else 0
        tones.add((root + (0, 2, 4, 5, 7, 9, 11)[(degree - 1) % 7] + alteration) % 12)
    return tones


def compatible_duration(tones, pitch_classes, overlaps):
    if tones is None or not overlaps or sum(overlaps) <= 0: return None
    return sum(w for pc, w in zip(pitch_classes, overlaps) if pc in tones) / sum(overlaps)


def scores(confusion, labels, reserved=(0, 1)):
    confusion = np.asarray(confusion, dtype=np.int64)
    summary, rows = metrics_from_confusion(torch.from_numpy(confusion), labels, reserved)
    support = confusion.sum(1); predicted = confusion.sum(0); tp = confusion.diagonal()
    eligible = support > 0
    for i in reserved: eligible[i] = False
    p = np.divide(tp, predicted, out=np.zeros(len(tp), dtype=float), where=predicted > 0)
    r = np.divide(tp, support, out=np.zeros(len(tp), dtype=float), where=support > 0)
    n = support[eligible].sum()
    summary.update(macro_precision=float(p[eligible].mean()), macro_recall=float(r[eligible].mean()),
        weighted_precision=float((p[eligible] * support[eligible]).sum() / n),
        weighted_recall=float((r[eligible] * support[eligible]).sum() / n),
        micro_f1=float(tp.sum() / confusion.sum()))
    return summary, rows


def confusion(y, p, size):
    return np.bincount(np.asarray(y) * size + np.asarray(p), minlength=size * size).reshape(size, size)


class CachedEvaluationDataset:
    """Official saved loader, with the notebook-06 memory cache pattern; no data writes."""
    def __init__(self, path):
        original = SavedSequenceDataset(path)
        self.shards = [SavedSequenceDataset(p) for p in original._shards] if original._shards is not None else [original]
        self.ends = list(original._ends) if original._shards is not None else [len(original)]
        keep = {'inputs__' + k for k in INPUT_KEYS} | {'targets__' + k for k in TARGET_KEYS}
        for shard in self.shards:
            shard.arrays = {k: v for k, v in shard.arrays.items()
                if not k.startswith(('inputs__', 'targets__')) or k in keep}

    def __len__(self): return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self): raise IndexError(index)
        shard = bisect.bisect_right(self.ends, index)
        item = self.shards[shard][index - (self.ends[shard - 1] if shard else 0)]
        item['metadata']['sequence_index'] = index
        return item


def discover_runs(root):
    base = Path(root) / 'experiments/rahul/outputs'
    result = {}
    for name, folder, checkpoint in [('baseline', 'transformer_training', 'best.pt'),
            ('weighted', 'weighted_transformer', 'best_weighted.pt'), ('multitask', 'multitask_transformer', 'best_multitask.pt')]:
        candidates = [p.parent for p in (base / folder).glob('*/results.json')
                      if read_json(p).get('status') == 'complete' and (p.parent / checkpoint).is_file()]
        if len(candidates) != 1:
            raise ValueError(f'{name}: expected one completed run, found {len(candidates)}. Supply explicit runs; do not silently choose.')
        result[name] = {'folder': candidates[0], 'checkpoint': candidates[0] / checkpoint}
    return result


class EvaluationStudy:
    def __init__(self, root, output, dataset=None, runs=None, batch_size=32, device='cpu', threads=4, seed=42):
        self.root = Path(root).resolve(); self.base = self.root / 'experiments/rahul'
        self.output = Path(output).resolve()
        allowed = (self.base / 'outputs/comprehensive_model_evaluation').resolve()
        if not self.output.is_relative_to(allowed): raise ValueError('Output must stay inside separate comprehensive evaluation directory')
        if self.output.exists(): raise FileExistsError('Use a fresh output folder; old evaluation outputs are protected')
        self.output.mkdir(parents=True); self.tables = self.output / 'tables'; self.figures = self.output / 'figures'
        self.tables.mkdir(); self.figures.mkdir()
        self.dataset = Path(dataset) if dataset else self.base / 'outputs/model_preprocessing/final_all_tracks_quality_v6_multibar'
        self.runs = runs or discover_runs(self.root)
        self.batch_size, self.device, self.threads, self.seed = batch_size, torch.device(device), threads, seed
        torch.set_num_threads(threads); torch.manual_seed(seed); np.random.seed(seed)
        self.spec = DatasetSpec.from_folder(self.dataset); self.labels = self.spec.target_labels; self.C = len(self.labels)
        self.encoder = FeatureEncoder.load(self.dataset)
        self.parts = [label_info(label) for label in self.labels]
        self.qualities = sorted({p[1] for label, p in zip(self.labels, self.parts) if label not in ('PAD', 'UNK')})
        self.quality_labels = self.qualities + ['INVALID']
        self.qlookup = np.array([self.quality_labels.index(q) if label not in ('PAD', 'UNK') else len(self.qualities)
                                for label, (_, q, _) in zip(self.labels, self.parts)])
        self.root_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N', 'INVALID']
        self.rlookup = np.array([r if r is not None else 12 if label == 'N' else 13 for label, (r, _, _) in zip(self.labels, self.parts)])
        self.configs = {name: read_json(run['folder'] / ('training_config.json' if name == 'baseline' else 'experiment_config.json'))
                        for name, run in self.runs.items()}
        self.previous = {name: read_json(run['folder'] / 'results.json') for name, run in self.runs.items()}
        self.histories = {name: read_json(run['folder'] / 'history.json') for name, run in self.runs.items()}
        self.protected = {}
        for folder in ['transformer_training', 'weighted_transformer', 'multitask_transformer', 'maj7_failure_analysis']:
            for p in (self.base / 'outputs' / folder).rglob('*'):
                if p.is_file(): self.protected[p.relative_to(self.root).as_posix()] = sha(p)
        for p in list((self.base / 'src').rglob('*.py')) + list((self.base / 'notebooks').glob('*.ipynb')):
            if p.name != '11_comprehensive_model_evaluation.ipynb': self.protected[p.relative_to(self.root).as_posix()] = sha(p)
        self.ready = read_json(self.dataset / 'FINAL_DATASET_READY.json')
        for name, expected in self.ready['artifacts'].items():
            if sha(self.dataset / name) != expected: raise ValueError(f'Dataset artifact changed before evaluation: {name}')
        save_json(self.output / 'protected_artifacts_before.json', self.protected)
        self.models = {}
        for name, run in self.runs.items():
            cfg = self.configs[name]
            if set(cfg['input_columns']) != {'inputs__' + k for k in INPUT_KEYS}: raise ValueError('Input contract differs')
            if cfg['model'] != self.configs['baseline']['model']: raise ValueError('Shared model configuration differs')
            loader = load_multitask_checkpoint if name == 'multitask' else load_checkpoint
            self.models[name] = loader(run['checkpoint'], self.dataset, self.device)
            if asdict(self.models[name].spec) != asdict(self.spec): raise ValueError('Incompatible class-index mapping')
        manifest = read_json(self.dataset / 'split_manifest.json')
        validate_splits(manifest, [t for ids in manifest['splits'].values() for t in ids])
        self.manifest = manifest
        self.data = CachedEvaluationDataset(self.dataset / 'test_sequences.json')
        self.loader = make_dataloader(self.data, batch_size=batch_size, shuffle=False, seed=seed)
        # Every batch is materialized once; timing later excludes decompression and disk I/O.
        self.batches = list(self.loader)
        self.events = self._events()
        self.y = np.array([e['target_id'] for e in self.events], dtype=np.int64)
        self.track_ids = sorted({e['track_id'] for e in self.events})
        self.track_index = np.array([self.track_ids.index(e['track_id']) for e in self.events])
        self.identity_hash = sha_bytes(json.dumps([(e['track_id'], e['sequence_id'], e['beat_index'], e['target_id'])
                                                   for e in self.events], separators=(',', ':')).encode())
        self.results = {}; self.predictions = {}; self.training_counts = self._training_counts()
        self.rare_ids = set(np.flatnonzero((self.training_counts > 0) & (self.training_counts < 100)).tolist())
        save_table(self.tables, 'evaluation_configuration', [self._configuration(name) for name in MODELS])
        save_json(self.output / 'protocol.json', {'device': str(self.device), 'batch_size': batch_size, 'threads': threads,
            'seed': seed, 'test_sequences': len(self.data), 'valid_test_targets': len(self.y), 'test_tracks': len(self.track_ids),
            'candidate_policy': 'All raw saved output coordinates, including PAD/UNK. No filtering, renormalization or auxiliary re-ranking.',
            'tie_policy': 'Stable descending logits, lower saved vocabulary ID first on exact ties.',
            'macro_policy': 'Positive reference-support classes in the evaluated population; PAD/UNK excluded; supervised N included. Undefined precision contributes zero.',
            'zero_division': 0, 'paired_event_identity_sha256': self.identity_hash,
            'rare_definition': 'Complete target chord class with 1-99 valid training beats; selected before test comparisons.',
            'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count()}})

    def relative(self, path): return Path(path).resolve().relative_to(self.root).as_posix()

    def _configuration(self, name):
        m = self.models[name]
        return {'model': name, 'checkpoint': self.relative(self.runs[name]['checkpoint']),
            'checkpoint_sha256': sha(self.runs[name]['checkpoint']), 'dataset': self.relative(self.dataset),
            'device': str(self.device), 'batch_size': self.batch_size, 'threads': self.threads,
            'sequences': len(self.data), 'valid_targets': len(self.y), 'test_tracks': len(self.track_ids),
            'classes': self.C, 'd_model': m.config.d_model, 'layers': m.config.num_layers,
            'parameters': sum(p.numel() for p in m.parameters()),
            'trainable_parameters': sum(p.numel() for p in m.parameters() if p.requires_grad),
            'quality_categories': len(m.quality_mapping.labels) if name == 'multitask' else None,
            'checkpoint_bytes': self.runs[name]['checkpoint'].stat().st_size,
            'target_vocabulary_sha256': self.spec.vocabulary_hashes['target_vocabulary.json'],
            'event_identity_sha256': self.identity_hash}

    def _training_counts(self):
        index = read_json(self.dataset / 'train_sequences.json'); counts = np.zeros(self.C, dtype=np.int64)
        for shard in index['shards']:
            with np.load(self.dataset / shard['path'], allow_pickle=False) as z:
                ids = z['targets__target_chord_ids']; mask = z['targets__target_loss_mask'] & z['inputs__beat_attention_mask'] & (ids >= 2)
                counts += np.bincount(ids[mask], minlength=self.C)
        return counts

    def _events(self):
        genre_path = self.root / 'datasets/mtg-jamendo/annotations/autotagging_genre.tsv'
        genres = {}
        if genre_path.exists():
            with genre_path.open(encoding='utf-8', newline='') as stream:
                for r in csv.reader(stream, delimiter='\t'):
                    if r and r[0].startswith('track_'):
                        genres[str(int(r[0][6:]))] = [g.removeprefix('genre---') for g in r[5:]]
        def decode(key, value): return self.encoder.vocabs[key].decode(int(value))
        events = []
        for batch in self.batches:
            x, targets = batch['inputs'], batch['targets']
            mask = supervised_mask(targets, x['beat_attention_mask'], self.C)
            for b, t in mask.nonzero().tolist():
                meta = batch['metadata'][b]; note_mask = x['melody_note_mask'][b, t]
                tokens = x['melody_pitch_ids'][b, t][note_mask].tolist()
                timings = x['melody_timing'][b, t][note_mask].tolist()
                duration = float(x['event_duration'][b, t])
                assert all(token >= 2 for token in tokens)
                assert int(meta['beat_indices'][t]) == int(x['beat_index'][b, t])
                track = str(meta['track_id'])
                active_genres = [g for g, flag in zip(self.encoder.genres, x['genre_features'][b]) if flag]
                events.append({'track_id': track, 'sequence_id': int(meta['sequence_index']), 'position_in_sequence': t,
                    'beat_index': int(meta['beat_indices'][t]), 'bar_number': int(x['bar_number'][b, t]),
                    'beat_position': int(x['beat_position'][b, t]), 'event_duration': duration,
                    'target_id': int(targets['target_chord_ids'][b, t]),
                    'reference_chord': self.labels[int(targets['target_chord_ids'][b, t])],
                    'basic_chord': decode('basic_chord_canonical', x['basic_chord_ids'][b, t]),
                    'key': decode('key', x['key_ids'][b, t]), 'mode': decode('mode', x['mode_ids'][b, t]),
                    'roman': decode('roman_numeral', x['roman_ids'][b, t]),
                    'harmonic_function': decode('harmonic_function', x['harmonic_function_ids'][b, t]),
                    'genre': genres.get(track, active_genres) or ['UNANNOTATED'], 'saved_genre_coordinates': active_genres,
                    'melody_pitch_classes': [(token - 2) % 12 for token in tokens],
                    'melody_overlap_seconds': [timing[2] * duration for timing in timings]})
        assert len({(e['track_id'], e['beat_index']) for e in events}) == len(events)
        assert {e['track_id'] for e in events} <= set(self.manifest['splits']['test'])
        return events


def sha_bytes(value): return hashlib.sha256(value).hexdigest()


def evaluate(study):
    """One immutable event order, common metric implementation and raw candidates."""
    for name in MODELS:
        model = study.models[name].eval()
        predictions, ranks, true_probability, top_ids, top_probability, auxiliary, observed_targets = [], [], [], [], [], [], []
        ce_sum = 0.0
        with torch.inference_mode():
            for batch in study.batches:
                x = {key: batch['inputs'][key].to(study.device) for key in INPUT_KEYS}
                targets = {key: batch['targets'][key].to(study.device) for key in TARGET_KEYS}
                mask = supervised_mask(targets, x['beat_attention_mask'], study.C)
                result = model(x); logits = result['chord_logits'] if isinstance(result, dict) else result
                selected = logits[mask]; y = targets['target_chord_ids'][mask]
                assert torch.isfinite(selected).all()
                order = selected.argsort(dim=-1, descending=True, stable=True)
                rank = (order == y[:, None]).to(torch.int64).argmax(-1) + 1
                probability = selected.softmax(-1)
                predictions.extend(order[:, 0].cpu().tolist()); ranks.extend(rank.cpu().tolist())
                true_probability.extend(probability.gather(1, y[:, None]).squeeze(1).cpu().tolist())
                top_ids.extend(order[:, :5].cpu().tolist())
                top_probability.extend(probability.gather(1, order[:, :5]).cpu().tolist())
                observed_targets.extend(y.cpu().tolist())
                ce_sum += float(F.cross_entropy(selected, y, reduction='sum'))
                if name == 'multitask':
                    qlogits = result['quality_logits'][mask]
                    assert torch.isfinite(qlogits).all()
                    local = qlogits.argmax(-1).cpu().tolist()
                    auxiliary.extend(study.quality_labels.index(model.quality_mapping.labels[i]) for i in local)
        np.testing.assert_array_equal(observed_targets, study.y)
        p = np.asarray(predictions, dtype=np.int64); rank = np.asarray(ranks, dtype=np.int64)
        cm = confusion(study.y, p, study.C)
        metrics, classes = scores(cm, study.labels)
        metrics.update(model=name, cross_entropy=ce_sum / len(p), top1_accuracy=metrics['accuracy'],
            top3_accuracy=float((rank <= 3).mean()), top5_accuracy=float((rank <= 5).mean()),
            mean_reciprocal_rank=float((1 / rank).mean()), mean_reference_probability=float(np.mean(true_probability)),
            top1_to_top3_gain=float((rank <= 3).mean() - (rank == 1).mean()))
        qtrue, qpred = study.qlookup[study.y], study.qlookup[p]
        qcm = confusion(qtrue, qpred, len(study.quality_labels))
        quality_summary, quality_rows = scores(qcm, study.quality_labels, reserved=())
        for row in quality_rows:
            q = row['target_id']; member = qtrue == q
            row.update(model=name, head='full_chord_implied', quality=row['reference_chord'],
                quality_accuracy=row['recall'],
                exact_full_chord_accuracy=float((p[member] == study.y[member]).mean()) if member.any() else None,
                top3_exact_chord_accuracy=float((rank[member] <= 3).mean()) if member.any() else None)
        study.predictions[name] = {'predicted_ids': p, 'ranks': rank, 'reference_probability': np.asarray(true_probability),
            'top5_ids': np.asarray(top_ids), 'top5_probability': np.asarray(top_probability), 'auxiliary_ids': np.asarray(auxiliary)}
        study.results[name] = {'metrics': metrics, 'confusion': cm, 'quality_confusion': qcm,
            'quality_summary': quality_summary, 'quality_rows': quality_rows}
        if name == 'multitask':
            aq = np.asarray(auxiliary, dtype=np.int64); acm = confusion(qtrue, aq, len(study.quality_labels))
            am, ar = scores(acm, study.quality_labels, reserved=())
            for row in ar:
                row.update(model='multitask', head='auxiliary', quality=row['reference_chord'], quality_accuracy=row['recall'])
            study.auxiliary = {'metrics': am, 'rows': ar, 'confusion': acm}
            save_table(study.tables, 'multitask_auxiliary_quality', ar)
            save_json(study.tables / 'multitask_auxiliary_metrics.json', am)
        save_table(study.tables, name + '_per_chord', classes)
        save_table(study.tables, name + '_per_quality', quality_rows)
        # Save only target/top-five probabilities, not an N x C probability table.
        rows = []
        for i, event in enumerate(study.events):
            rows.append({**{k: event[k] for k in ('track_id', 'sequence_id', 'position_in_sequence', 'beat_index', 'bar_number', 'beat_position', 'target_id', 'reference_chord', 'basic_chord')},
                'predicted_id': int(p[i]), 'predicted_chord': study.labels[p[i]], 'true_rank': int(rank[i]),
                'reference_probability': true_probability[i],
                'top5': [{'chord': study.labels[j], 'probability': prob} for j, prob in zip(top_ids[i], top_probability[i])]})
        save_table(study.tables, name + '_event_predictions', rows)
        expected = study.previous[name]
        if name == 'baseline': expected = expected['test']
        elif name == 'weighted': expected = expected['weighted']
        else: expected = expected['full_head_metrics']['multitask']
        for key in ('accuracy', 'macro_f1', 'top3_accuracy'):
            if key in expected and not math.isclose(metrics[key], expected[key], abs_tol=1e-10):
                raise AssertionError(f'{name} {key} differs from saved published evaluation: {metrics[key]} vs {expected[key]}')
        previous_comparison = study.previous['multitask']['full_head_metrics']['original' if name == 'baseline' else name]
        for key in ('accuracy', 'macro_f1', 'weighted_f1', 'top3_accuracy', 'top5_accuracy'):
            if not math.isclose(metrics[key], previous_comparison[key], abs_tol=1e-10):
                raise AssertionError(f'{name} {key} differs from the saved three-model comparison')
        print(f'{name}: {len(p):,} paired targets; exact={metrics["accuracy"]:.6%}; macro F1={metrics["macro_f1"]:.6f}', flush=True)
    np.savez_compressed(study.output / 'prediction_arrays.npz', targets=study.y, track_index=study.track_index,
        **{name + '_' + k: v for name, pred in study.predictions.items() for k, v in pred.items()})
    save_json(study.output / 'event_context.json', study.events)
    study.overall = save_table(study.tables, 'overall_metrics', [study.results[n]['metrics'] for n in MODELS])
    study.quality_table = save_table(study.tables, 'all_quality_metrics', [r for n in MODELS for r in study.results[n]['quality_rows']])
    np.savez_compressed(study.output / 'confusions.npz', **{n: study.results[n]['confusion'] for n in MODELS},
        **{n + '_quality': study.results[n]['quality_confusion'] for n in MODELS}, auxiliary_quality=study.auxiliary['confusion'])
    qnames = study.quality_labels
    changes = []
    for q in study.qualities:
        rows = {name: next(r for r in study.results[name]['quality_rows'] if r['quality'] == q) for name in MODELS}
        changes.append({'quality': q, 'support': rows['baseline']['support'],
            **{name + '_exact': rows[name]['exact_full_chord_accuracy'] for name in MODELS},
            **{name + '_quality_f1': rows[name]['f1'] for name in MODELS},
            **{name + '_quality_recall': rows[name]['recall'] for name in MODELS},
            **{name + '_exact_change_vs_baseline': rows[name]['exact_full_chord_accuracy'] - rows['baseline']['exact_full_chord_accuracy']
               if rows[name]['support'] else None for name in ('weighted', 'multitask')}})
    study.quality_changes = save_table(study.tables, 'quality_improvement_deterioration', changes)
    return study.overall


def error_analysis(study):
    categories, pairs, roots, majors, rare = [], [], {}, [], []
    y = study.y
    for name in MODELS:
        pred = study.predictions[name]; p = pred['predicted_ids']; counter = Counter()
        for ref, guess in zip(y, p):
            r, q, bass = study.parts[ref]; pr, pq, pbass = study.parts[guess]
            if ref == guess: category = 'Exact correct'
            elif r is None or pr is None: category = 'No-chord / invalid-label error'
            elif r == pr and q == pq: category = 'Correct root+quality; wrong inversion'
            elif r == pr: category = 'Correct root; wrong quality'
            elif q == pq: category = 'Wrong root; correct quality'
            else: category = 'Wrong root and quality'
            counter[category] += 1
        assert sum(counter.values()) == len(y)
        for category in ['Exact correct', 'Correct root+quality; wrong inversion', 'Correct root; wrong quality',
                         'Wrong root; correct quality', 'Wrong root and quality', 'No-chord / invalid-label error']:
            categories.append({'model': name, 'category': category, 'count': counter[category],
                'fraction_of_all_targets': counter[category] / len(y),
                'fraction_of_errors': counter[category] / int((y != p).sum()) if category != 'Exact correct' else None})
        roots[name] = confusion(study.rlookup[y], study.rlookup[p], len(study.root_labels))
        confused = Counter((int(a), int(b)) for a, b in zip(y, p) if a != b)
        for (a, b), count in confused.most_common():
            pairs.append({'model': name, 'reference': study.labels[a], 'prediction': study.labels[b], 'count': count,
                'reference_training_support': int(study.training_counts[a]), 'same_root': study.parts[a][0] == study.parts[b][0]})
        ismaj7 = np.array([study.parts[t][1] == 'maj7' for t in y]); support = int(ismaj7.sum())
        qpred = study.qlookup[p]; qtrue = study.qlookup[y]; maj7id = study.quality_labels.index('maj7')
        tp = int(((qpred == maj7id) & ismaj7).sum()); fp = int(((qpred == maj7id) & ~ismaj7).sum())
        r = {'model': name, 'support': support, 'exact_correct': int((p[ismaj7] == y[ismaj7]).sum()),
            'exact_accuracy': float((p[ismaj7] == y[ismaj7]).mean()),
            'top3_exact_accuracy': float((pred['ranks'][ismaj7] <= 3).mean()),
            'top5_exact_accuracy': float((pred['ranks'][ismaj7] <= 5).mean()),
            'predicted_as_maj': int(sum(study.parts[t][1] == 'maj' for t in p[ismaj7])),
            'same_root_maj': int(sum(study.parts[a][0] == study.parts[b][0] and study.parts[b][1] == 'maj' for a, b in zip(y[ismaj7], p[ismaj7]))),
            'full_head_quality_precision': tp / (tp + fp) if tp + fp else 0., 'full_head_quality_recall': tp / support,
            'mean_reference_probability': float(pred['reference_probability'][ismaj7].mean())}
        if name == 'multitask':
            aq = pred['auxiliary_ids']; atp = int(((aq == maj7id) & ismaj7).sum()); ap = int((aq == maj7id).sum())
            r.update(auxiliary_precision=atp / ap if ap else 0., auxiliary_recall=atp / support, auxiliary_true_positives=atp)
        majors.append(r)
        member = np.isin(y, list(study.rare_ids))
        rare.append({'model': name, 'rare_class_definition': 'Training support 1-99 valid beats',
            'test_beats': int(member.sum()), 'target_classes': len(set(y[member])),
            'exact_accuracy': float((y[member] == p[member]).mean()) if member.any() else None,
            'top3_accuracy': float((pred['ranks'][member] <= 3).mean()) if member.any() else None})
    study.errors = save_table(study.tables, 'error_categories', categories)
    study.confused_pairs = save_table(study.tables, 'confused_chord_pairs', pairs)
    study.maj7 = save_table(study.tables, 'maj7_comparison', majors)
    study.rare = save_table(study.tables, 'rare_class_errors', rare)
    study.root_confusions = roots
    np.savez_compressed(study.output / 'root_confusions.npz', **roots)
    return study.errors


def training_behavior(study):
    epoch_rows, summaries = [], []
    for name in MODELS:
        history = study.histories[name]; previous = study.previous[name]; best = previous['best_epoch']
        for epoch in history:
            train, val = epoch['train'], epoch['validation']
            loss_key = 'chord_loss' if name == 'multitask' else 'loss'
            # Logs are authoritative; absent values remain None rather than estimates.
            train_ce, val_ce = train.get(loss_key), val.get(loss_key)
            epoch_rows.append({'model': name, 'epoch': epoch['epoch'], 'learning_rate': epoch.get('learning_rate'),
                'training_full_chord_ce': train_ce, 'validation_full_chord_ce': val_ce,
                'training_objective_loss': train.get('combined_loss') if name == 'multitask' else train.get('weighted_loss') if name == 'weighted' else train.get('loss'),
                'training_accuracy': train.get('accuracy'), 'validation_accuracy': val.get('accuracy'),
                'quality_training_loss': train.get('quality_loss'), 'quality_validation_loss': val.get('quality_loss'),
                'accuracy_generalization_gap': train['accuracy'] - val['accuracy'] if 'accuracy' in train and 'accuracy' in val else None,
                'selected': epoch['epoch'] == best})
        selected = next(r for r in epoch_rows if r['model'] == name and r['selected'])
        # Sum actual logged train/validation seconds; exclude initialization, test and checkpoint I/O.
        seconds = [e[phase].get('seconds') for e in history for phase in ('train', 'validation')]
        duration = sum(seconds) if all(v is not None for v in seconds) else None
        summaries.append({'model': name, 'completed_epochs': len(history), 'best_epoch': best,
            'maximum_epochs': study.configs[name]['max_epochs'], 'patience': study.configs[name]['patience'],
            'stopped_before_maximum': len(history) < study.configs[name]['max_epochs'],
            'selected_training_accuracy': selected['training_accuracy'], 'selected_validation_accuracy': selected['validation_accuracy'],
            'selected_validation_ce': selected['validation_full_chord_ce'], 'selected_accuracy_gap': selected['accuracy_generalization_gap'],
            'final_accuracy_gap': epoch_rows[-1]['accuracy_generalization_gap'],
            'recorded_train_validation_seconds': duration,
            'historical_duration_scope': 'Sum of recorded epoch train+validation timers; not full wall-clock training job.',
            'test_accuracy': study.results[name]['metrics']['accuracy']})
    study.training_epochs = save_table(study.tables, 'training_history_comparison', epoch_rows)
    study.training_summary = save_table(study.tables, 'training_behavior_summary', summaries)
    return study.training_summary


def musical_diagnostics(study):
    """Explicit heuristics; no claims about listening quality or causal genre conditioning."""
    tones = [chord_tones(label) for label in study.labels]
    melody_rows, harmonic_rows, genre_rows, genre_behaviors, transitions, root_movements = [], [], [], [], [], []
    common = np.array([bool(e['melody_overlap_seconds']) and sum(e['melody_overlap_seconds']) > 0
        and tones[e['target_id']] is not None
        and all(tones[study.predictions[n]['predicted_ids'][i]] is not None for n in MODELS)
        for i, e in enumerate(study.events)])
    weights = np.array([sum(e['melody_overlap_seconds']) for e in study.events])
    common_reference_score = [compatible_duration(tones[e['target_id']], e['melody_pitch_classes'], e['melody_overlap_seconds'])
                              for i, e in enumerate(study.events) if common[i]]
    for name in MODELS:
        p = study.predictions[name]['predicted_ids']; rank = study.predictions[name]['ranks']
        event_scores, individual_available = [], 0
        key_count = inside = root_inside = reference_inside = function_count = function_match = 0
        functions = Counter(); predicted_functions = []
        for i, (event, target) in enumerate(zip(study.events, p)):
            score = compatible_duration(tones[target], event['melody_pitch_classes'], event['melody_overlap_seconds'])
            if score is not None: individual_available += 1
            if common[i]: event_scores.append(score)
            key = ROOT_TO_PITCH_CLASS.get(event['key']); mode = event['mode']
            f = None
            if key is not None and mode in ('major', 'minor') and tones[target] is not None and tones[event['target_id']] is not None:
                scale = (0, 2, 4, 5, 7, 9, 11) if mode == 'major' else (0, 2, 3, 5, 7, 8, 10)
                pcs = {(key + d) % 12 for d in scale}
                key_count += 1; inside += tones[target] <= pcs
                reference_inside += tones[event['target_id']] <= pcs
                root_inside += study.parts[target][0] in pcs
            if key is not None and mode in ('major', 'minor') and study.parts[target][0] is not None:
                _, f = context(simplify(study.labels[target]), event['key'], mode)
                if f == 'unknown': f = None
            predicted_functions.append(f)
            if event['harmonic_function'] not in ('unknown', 'UNK', 'PAD', 'N', '') and f is not None:
                function_count += 1; function_match += f == event['harmonic_function']
        common_weights = weights[common]
        melody_rows.append({'model': name, 'common_evaluable_beats': int(common.sum()),
            'individually_evaluable_beats': individual_available,
            'silent_or_unpitched_supervised_beats': int((weights == 0).sum()),
            'note_overlap_weighted_compatibility': float(np.average(event_scores, weights=common_weights)) if len(event_scores) else None,
            'mean_beat_compatibility': float(np.mean(event_scores)) if event_scores else None,
            'reference_overlap_weighted_compatibility_same_events': float(np.average(common_reference_score, weights=common_weights)) if event_scores else None,
            'overlap_note_seconds': float(common_weights.sum()),
            'definition': 'Overlap-duration weighted chord-tone membership, same eligible events across models; explicit slash bass included; N excluded.'})
        harmonic_rows.append({'model': name, 'diatonic_evaluable_beats': key_count,
            'predicted_all_tones_in_key_fraction': inside / key_count if key_count else None,
            'reference_all_tones_in_key_fraction': reference_inside / key_count if key_count else None,
            'predicted_root_in_key_fraction': root_inside / key_count if key_count else None,
            'known_predicted_and_input_function_beats': function_count,
            'function_agreement_with_basic_context': function_match / function_count if function_count else None,
            'rule': 'Major or natural-minor pitch-class subset. Non-diatonic does not mean invalid; basic-derived function agreement is not independent truth.'})
        movements = Counter(); changes = matched = consecutive = same_bar_pairs = cross_bar_pairs = same_bar_changes = cross_bar_changes = 0
        ref_functions = Counter()
        for i in range(1, len(study.events)):
            a, b = study.events[i - 1], study.events[i]
            if a['track_id'] != b['track_id'] or b['beat_index'] != a['beat_index'] + 1: continue
            # No bridging masked positions or source gaps; crossing saved sequence boundaries is allowed.
            pa, pb = int(p[i - 1]), int(p[i]); ra, rb = study.parts[pa][0], study.parts[pb][0]
            refa, refb = study.parts[a['target_id']][0], study.parts[b['target_id']][0]
            if None not in (ra, rb, refa, refb):
                consecutive += 1; movement = (rb - ra) % 12; movements[movement] += 1
                matched += movement == (refb - refa) % 12; changes += pa != pb
                if a['bar_number'] == b['bar_number']:
                    same_bar_pairs += 1; same_bar_changes += pa != pb
                else: cross_bar_pairs += 1; cross_bar_changes += pa != pb
            if predicted_functions[i - 1] and predicted_functions[i]: functions[(predicted_functions[i - 1], predicted_functions[i])] += 1
            if a['harmonic_function'] in ('tonic', 'predominant', 'dominant') and b['harmonic_function'] in ('tonic', 'predominant', 'dominant'):
                ref_functions[(a['harmonic_function'], b['harmonic_function'])] += 1
        harmonic_rows[-1].update(consecutive_root_pairs=consecutive,
            root_movement_agreement=matched / consecutive if consecutive else None,
            predicted_chord_change_rate=changes / consecutive if consecutive else None,
            within_bar_pairs=same_bar_pairs, within_bar_chord_change_rate=same_bar_changes / same_bar_pairs if same_bar_pairs else None,
            across_bar_pairs=cross_bar_pairs, across_bar_chord_change_rate=cross_bar_changes / cross_bar_pairs if cross_bar_pairs else None)
        root_movements.extend({'model': name, 'upward_semitones_mod12': k, 'count': movements[k],
                               'fraction': movements[k] / consecutive if consecutive else None} for k in range(12))
        for first in ('tonic', 'predominant', 'dominant'):
            for second in ('tonic', 'predominant', 'dominant'):
                transitions.append({'model': name, 'from_function': first, 'to_function': second,
                    'predicted_count': functions[(first, second)], 'input_basic_context_count': ref_functions[(first, second)]})
        genres = sorted({g for event in study.events for g in event['genre']})
        for genre in genres:
            member = np.array([genre in e['genre'] for e in study.events])
            cm = confusion(study.y[member], p[member], study.C); metrics, _ = scores(cm, study.labels)
            genre_rows.append({'model': name, 'genre': genre, 'tracks': len({e['track_id'] for e, flag in zip(study.events, member) if flag}),
                'support': int(member.sum()), 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
                'macro_classes': metrics['macro_classes'], 'top3_accuracy': float((rank[member] <= 3).mean())})
            qt, qp = study.qlookup[study.y[member]], study.qlookup[p[member]]
            for q, quality in enumerate(study.quality_labels):
                if (qt == q).any() or (qp == q).any():
                    genre_behaviors.append({'model': name, 'genre': genre, 'quality': quality,
                        'reference_count': int((qt == q).sum()), 'predicted_count': int((qp == q).sum()),
                        'predicted_fraction': float((qp == q).mean())})
    study.melody = save_table(study.tables, 'melody_compatibility', melody_rows)
    study.harmonic = save_table(study.tables, 'harmonic_diagnostics', harmonic_rows)
    study.genres = save_table(study.tables, 'genre_metrics', genre_rows)
    study.genre_behaviors = save_table(study.tables, 'genre_quality_behavior', genre_behaviors)
    study.transitions = save_table(study.tables, 'harmonic_function_transitions', transitions)
    study.movements = save_table(study.tables, 'root_movement_distribution', root_movements)
    save_json(study.output / 'musical_diagnostic_rules.json', {'intervals': INTERVALS,
        'inversion': 'Include explicit degree bass in chord-tone set; parse labels with project utilities.',
        'melody_weight': 'Per-note overlap duration in seconds reconstructed from saved float32 overlap ratio * event duration. Concurrent notes contribute individually; sustained notes contribute their overlap on each beat.',
        'excluded': 'Silent/no-note events, N or unsupported predicted/reference tone sets are excluded. Shared eligibility across models prevents changing denominators.',
        'key': 'Whole chord-tone-set inclusion in major/natural-minor scale, same eligible events; not an assertion that non-diatonic chords are wrong.',
        'function': 'Existing conservative context(simplify(prediction), key, mode). Known predicted functions compared to stored basic-derived function; endogenous diagnostic, not independent accuracy.',
        'adjacency': 'Consecutive valid-supervised source beat IDs in the same track only. Can cross sequence boundaries; never bridge masked/source gaps.',
        'genre': 'Original multi-label annotation tags, including unseen tags that map to saved UNK. Groups overlap and cannot be summed; no stylistic/listening claim.'})
    return study.melody


def track_bootstrap(study, repetitions=2000, confidence=0.95, seed=42):
    if repetitions < 2 or not 0 < confidence < 1: raise ValueError('Invalid bootstrap settings')
    tracks = len(study.track_ids); rng = np.random.default_rng(seed)
    weights = rng.multinomial(tracks, np.full(tracks, 1 / tracks), size=repetitions)
    support = np.zeros((tracks, study.C), dtype=np.int64)
    np.add.at(support, (study.track_index, study.y), 1)
    boot_support = weights @ support
    eligible = support.sum(0) > 0; eligible[:2] = False
    output, samples = [], {}
    for name in MODELS:
        p = study.predictions[name]['predicted_ids']
        pred_counts = np.zeros_like(support); tp = np.zeros_like(support)
        np.add.at(pred_counts, (study.track_index, p), 1)
        correct = p == study.y
        np.add.at(tp, (study.track_index[correct], study.y[correct]), 1)
        btp = weights @ tp; bpred = weights @ pred_counts
        acc = btp.sum(1) / boot_support.sum(1)
        bf1 = np.divide(2 * btp, boot_support + bpred, out=np.zeros_like(btp, dtype=float), where=(boot_support + bpred) > 0)
        macro = bf1[:, eligible].mean(1)
        samples[name] = {'accuracy': acc, 'macro_f1': macro}
        for metric in ('accuracy', 'macro_f1'):
            low, high = np.quantile(samples[name][metric], [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
            output.append({'comparison': name, 'metric': metric, 'point_estimate': study.results[name]['metrics'][metric],
                'ci_low': float(low), 'ci_high': float(high), 'confidence': confidence, 'resampling_unit': 'track',
                'tracks': tracks, 'repetitions': repetitions, 'seed': seed})
    for first, second in combinations(MODELS, 2):
        for metric in ('accuracy', 'macro_f1'):
            delta = samples[second][metric] - samples[first][metric]
            low, high = np.quantile(delta, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
            output.append({'comparison': second + ' minus ' + first, 'metric': metric,
                'point_estimate': study.results[second]['metrics'][metric] - study.results[first]['metrics'][metric],
                'ci_low': float(low), 'ci_high': float(high), 'confidence': confidence,
                'resampling_unit': 'paired track', 'tracks': tracks, 'repetitions': repetitions, 'seed': seed,
                'interval_excludes_zero': bool(low > 0 or high < 0)})
    study.bootstrap = save_table(study.tables, 'track_bootstrap_confidence', output)
    np.savez_compressed(study.output / 'bootstrap_samples.npz', track_weights=weights,
        **{name + '_' + metric: values for name, metrics in samples.items() for metric, values in metrics.items()})
    save_json(study.output / 'bootstrap_protocol.json', {'seed': seed, 'repetitions': repetitions,
        'confidence': confidence, 'unit': f'{tracks} track clusters treated as independent within bootstrap; not beats',
        'actual_test_tracks': tracks, 'estimator': 'Pooled-beat accuracy and fixed-original-test-supported-class macro F1.',
        'macro_policy': 'Keep original test-support class set fixed in every replicate; classes absent from a replicate contribute zero F1. This class-coverage sensitivity is a limitation.',
        'pairing': 'Same multinomial track weights for every model and metric; unequal-length tracks retain their beat contributions.',
        'limitations': 'Percentile intervals conditional on this test population and single training seed. Tracks sharing artists/albums may remain correlated. No multiple-comparison adjustment. No initialization-seed uncertainty.',
        'mcnemar': 'Not performed: beat-level independence would be inappropriate; paired track bootstrap is primary.'})
    return study.bootstrap


def select_examples(study):
    correct = {n: study.predictions[n]['predicted_ids'] == study.y for n in MODELS}
    quality_training_counts = Counter()
    for target, count in enumerate(study.training_counts):
        if count: quality_training_counts[study.parts[target][1]] += int(count)
    rare_quality = np.array([0 < quality_training_counts[study.parts[t][1]] < 100 for t in study.y])
    conditions = {
        'All three exact correct': correct['baseline'] & correct['weighted'] & correct['multitask'],
        'Multi-task improves baseline': correct['multitask'] & ~correct['baseline'],
        'Weighted improves a minority class': correct['weighted'] & ~correct['baseline'] & np.isin(study.y, list(study.rare_ids)),
        'Major-seventh failure': np.array([study.parts[y][1] == 'maj7' for y in study.y]) & ~correct['multitask'],
        'True target in top 3 but not top 1': (study.predictions['multitask']['ranks'] <= 3) & ~correct['multitask'],
        'Wrong root': np.array([study.parts[a][0] is not None and study.parts[b][0] is not None and study.parts[a][0] != study.parts[b][0]
                              for a, b in zip(study.y, study.predictions['multitask']['predicted_ids'])]),
        'Correct root but wrong quality': np.array([study.parts[a][0] is not None and study.parts[a][0] == study.parts[b][0] and study.parts[a][1] != study.parts[b][1]
                                                   for a, b in zip(study.y, study.predictions['multitask']['predicted_ids'])]),
        'Rare-quality failure': rare_quality & ~correct['multitask'],
        'Rare full-chord-class failure': np.isin(study.y, list(study.rare_ids)) & ~correct['multitask']}
    rows = []; availability = []
    for category, flags in conditions.items():
        choices = np.flatnonzero(flags)
        availability.append({'category': category, 'matching_events': len(choices), 'selection': 'First qualifying saved test event; deterministic, illustrative, not representative of frequency.'})
        if not len(choices): continue
        i = int(choices[0]); event = study.events[i]
        for name in MODELS:
            p = study.predictions[name]
            candidates = [{'chord': study.labels[int(c)], 'probability': float(v)} for c, v in zip(p['top5_ids'][i], p['top5_probability'][i])]
            rows.append({'category': category, 'model': name, **event, 'predicted_chord': study.labels[int(p['predicted_ids'][i])],
                'true_rank': int(p['ranks'][i]), 'reference_probability': float(p['reference_probability'][i]),
                'top3': candidates[:3], 'top5': candidates,
                'auxiliary_quality': study.quality_labels[int(p['auxiliary_ids'][i])] if name == 'multitask' else None})
    study.examples = save_table(study.tables, 'representative_examples', rows)
    study.example_availability = save_table(study.tables, 'example_category_availability', availability)
    return study.examples


def benchmark_worker(job_path):
    job = read_json(job_path); torch.set_num_threads(job['threads'])
    device = torch.device(job['device']); name = job['model']
    loader = load_multitask_checkpoint if name == 'multitask' else load_checkpoint
    model = loader(job['checkpoint'], job['dataset'], device).eval()
    data = CachedEvaluationDataset(Path(job['dataset']) / 'test_sequences.json')
    batches = list(make_dataloader(data, batch_size=job['batch_size'], shuffle=False, seed=job['seed']))
    inputs = [{k: batch['inputs'][k].to(device) for k in INPUT_KEYS} for batch in batches]
    valid = [int(supervised_mask(b['targets'], b['inputs']['beat_attention_mask'], len(model.spec.target_labels)).sum()) for b in batches]
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    rss_before = peak_rss = None
    try:
        import psutil
        process = psutil.Process(); rss_before = process.memory_info().rss
    except ImportError:
        process = None
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    latency_rows = []; repetitions = []
    with torch.inference_mode():
        for i in range(job['warmup_batches']): model(inputs[i % len(inputs)])
        sync()
        for repetition in range(job['repetitions']):
            total_seconds = 0.0
            for number, (x, count) in enumerate(zip(inputs, valid)):
                sync(); start = time.perf_counter(); output = model(x); sync(); elapsed = time.perf_counter() - start
                # Include both forward heads for multitask; no CE/softmax/top-k/IO in measured region.
                del output
                total_seconds += elapsed
                if process is not None:
                    memory = process.memory_info(); peak_rss = max(peak_rss or 0, memory.rss, getattr(memory, 'peak_wset', 0))
                latency_rows.append({'model': name, 'repetition': repetition + 1, 'batch': number,
                    'batch_size': len(x['beat_attention_mask']), 'valid_beats': count, 'seconds': elapsed})
            repetitions.append({'repetition': repetition + 1, 'forward_seconds': total_seconds,
                                'valid_beats_per_second': sum(valid) / total_seconds})
    latencies = np.array([r['seconds'] for r in latency_rows])
    result = {'model': name, 'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'checkpoint_bytes': Path(job['checkpoint']).stat().st_size,
        'mean_batch_latency_ms': float(latencies.mean() * 1000), 'median_batch_latency_ms': float(np.median(latencies) * 1000),
        'p95_batch_latency_ms': float(np.quantile(latencies, .95) * 1000),
        'valid_beats_per_second': sum(valid) * job['repetitions'] / float(latencies.sum()),
        'peak_process_rss_bytes': peak_rss, 'process_rss_before_timing_bytes': rss_before,
        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'device': str(device), 'batch_size': job['batch_size'], 'warmup_batches': job['warmup_batches'],
        'repetitions': job['repetitions'], 'timed_batches': len(latencies), 'valid_beats_per_pass': sum(valid),
        'memory_scope': 'Fresh isolated worker process: Python/libraries, one model, cached dataset and preloaded batches. Windows peak working set or sampled RSS; not isolated activation memory.',
        'timing_scope': 'Forward only on preloaded device batches; includes both multitask heads. Excludes data loading, transfers, loss, top-k and report generation.',
        'trial_results': repetitions, 'latencies': latency_rows}
    save_json(job['result'], result)


def benchmark(study, repetitions=3, warmup_batches=5):
    rows = []; raw = []
    for name in MODELS:
        job = {'model': name, 'checkpoint': str(study.runs[name]['checkpoint']), 'dataset': str(study.dataset),
            'device': str(study.device), 'batch_size': study.batch_size, 'threads': study.threads, 'seed': study.seed,
            'repetitions': repetitions, 'warmup_batches': warmup_batches,
            'result': str(study.tables / (name + '_benchmark_raw.json'))}
        job_path = study.tables / (name + '_benchmark_job.json'); save_json(job_path, job)
        completed = subprocess.run([sys.executable, '-m', 'experiments.rahul.src.comprehensive_evaluation',
            '--benchmark-worker', str(job_path)], cwd=study.root, capture_output=True, text=True, timeout=600)
        if completed.returncode: raise RuntimeError(completed.stdout + '\n' + completed.stderr)
        result = read_json(job['result']); raw.extend(result.pop('latencies')); trial = result.pop('trial_results')
        result.update(recorded_train_validation_seconds=float(study.training_summary.set_index('model').loc[name, 'recorded_train_validation_seconds']),
                      repetition_throughput=[r['valid_beats_per_second'] for r in trial])
        rows.append(result)
        print(f'{name}: newly measured forward mean {result["mean_batch_latency_ms"]:.2f} ms/batch', flush=True)
    study.efficiency = save_table(study.tables, 'efficiency_comparison', rows)
    save_table(study.tables, 'benchmark_batch_latencies', raw)
    return study.efficiency


def make_figures(study):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    figures = []
    def save(fig, name):
        fig.tight_layout(); path = study.figures / (name + '.png')
        fig.savefig(path, dpi=180, bbox_inches='tight'); plt.close(fig); figures.append(name)
    def grouped(frame, keys, titles, name, ylabel='Percent', percentage=True):
        fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.7), 4.5)); x = np.arange(len(keys)); width = .23
        for j, model in enumerate(MODELS):
            row = frame.set_index('model').loc[model]
            ax.bar(x + (j - 1) * width, [row[k] * (100 if percentage else 1) for k in keys], width,
                   label=DISPLAY[model], color=COLORS[model])
        ax.set_xticks(x, [textwrap.fill(title, width=22, break_long_words=False) for title in titles]); ax.set_ylabel(ylabel); ax.set_ylim(bottom=0, top=100 if percentage else None)
        ax.legend(); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); save(fig, name)
    def heat_panels(matrices, labels, titles, name, normalize=True):
        fig, axes = plt.subplots(1, len(matrices), figsize=(6.1 * len(matrices), 6), squeeze=False)
        for ax, matrix, title in zip(axes[0], matrices, titles):
            matrix = np.asarray(matrix, dtype=float); sums = matrix.sum(1)
            values = np.divide(matrix, sums[:, None], out=np.zeros_like(matrix), where=sums[:, None] > 0) * 100 if normalize else matrix
            masked = np.ma.masked_where(np.broadcast_to(sums[:, None] == 0, values.shape), values)
            cmap = plt.get_cmap('Blues').copy(); cmap.set_bad('#eeeeee')
            im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100 if normalize else max(1, values.max()), aspect='equal')
            ax.set_xticks(range(len(labels)), labels, rotation=55, ha='right'); ax.set_yticks(range(len(labels)), labels)
            ax.set(xlabel='Predicted', ylabel='Reference', title=title)
            for r, c in zip(*np.where(matrix > 0)):
                text = ('<0.1' if 0 < values[r,c] < .1 else f'{values[r,c]:.1f}') if normalize else str(int(matrix[r,c]))
                ax.text(c, r, text, ha='center', va='center', fontsize=7, color='white' if values[r,c] > (50 if normalize else values.max() / 2) else 'black')
            fig.colorbar(im, ax=ax, shrink=.7, label='Row percent' if normalize else 'Count')
        save(fig, name)
    grouped(study.overall, ['accuracy', 'macro_f1', 'weighted_f1', 'micro_f1'],
            ['Exact accuracy', 'Macro F1', 'Weighted F1', 'Micro F1'], 'overall_classification')
    grouped(study.overall, ['cross_entropy'], ['Unweighted test full-chord CE'], 'test_cross_entropy', 'Cross-entropy (nats)', False)
    grouped(study.overall, ['top1_accuracy', 'top3_accuracy', 'top5_accuracy'], ['Top 1', 'Top 3', 'Top 5'], 'top_k_accuracy')
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in MODELS:
        ax.plot(range(1, 11), [100 * (study.predictions[name]['ranks'] <= k).mean() for k in range(1, 11)],
                marker='o', color=COLORS[name], label=DISPLAY[name])
    ax.set(xlabel='Number of full-chord candidates (K)', ylabel='Exact target in top K (%)', ylim=(0, 100), xticks=range(1, 11)); ax.legend(); ax.grid(alpha=.2)
    save(fig, 'top_k_curves')
    rank_names = ['1', '2', '3', '4', '5', '6-10', '11-25', '26+']; rank_rows = []
    for name in MODELS:
        rank = study.predictions[name]['ranks']
        intervals = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 25), (26, study.C)]
        rank_rows.append({'model': name, **{label: float(((rank >= lo) & (rank <= hi)).mean()) for label, (lo, hi) in zip(rank_names, intervals)}})
    ranks = save_table(study.tables, 'reference_rank_distribution', rank_rows)
    grouped(ranks, rank_names, rank_names, 'reference_rank_distribution', 'Valid targets (%)')
    observed = [q for q in study.qualities if any(r['quality'] == q and r['support'] for r in study.results['baseline']['quality_rows'])]
    fig, ax = plt.subplots(figsize=(max(10, len(observed) * 1.3), 4.7)); x = np.arange(len(observed)); width = .19
    for j, name in enumerate((*MODELS, 'auxiliary')):
        rows = study.auxiliary['rows'] if name == 'auxiliary' else study.results[name]['quality_rows']
        lookup = {r['quality']: r for r in rows}
        ax.bar(x + (j - 1.5) * width, [100 * (lookup[q]['f1'] or 0) for q in observed], width,
               color=COLORS[name], label=DISPLAY[name] + (' head' if name == 'auxiliary' else ' full-head quality'))
    ax.set_xticks(x, observed); ax.set(ylabel='Quality-only F1 (%)', ylim=(0, 100), title='Quality recognition; not exact full-chord accuracy'); ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, 1.30), ncol=2); ax.grid(axis='y', alpha=.2)
    save(fig, 'per_quality_f1')
    heat_panels([study.results[n]['quality_confusion'] for n in MODELS], study.quality_labels,
                [DISPLAY[n] + ' full-head quality' for n in MODELS], 'quality_confusions')
    heat_panels([study.auxiliary['confusion']], study.quality_labels, ['Multi-task auxiliary quality'], 'auxiliary_quality_confusion')
    recall = np.array([[next(r['recall'] for r in study.results[n]['quality_rows'] if r['quality'] == q) * 100 for q in observed] for n in MODELS]
                      + [[next(r['recall'] for r in study.auxiliary['rows'] if r['quality'] == q) * 100 for q in observed]])
    fig, ax = plt.subplots(figsize=(11, 3.4)); im = ax.imshow(recall, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(observed)), observed); ax.set_yticks(range(4), [DISPLAY[n] for n in (*MODELS, 'auxiliary')])
    for r in range(4):
        for c in range(len(observed)):ax.text(c, r, f'{recall[r,c]:.1f}', ha='center', va='center', color='white' if recall[r,c] > 50 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, label='Quality recall (%)');ax.set_title('Detailed-quality recall; zero-support qualities omitted');save(fig, 'quality_recall_heatmap')
    grouped(study.maj7, ['exact_accuracy', 'top3_exact_accuracy', 'top5_exact_accuracy'], ['Maj7 exact top 1', 'Maj7 exact top 3', 'Maj7 exact top 5'], 'maj7_candidate_accuracy')
    # All maj7 predictions are retained in this plot, with an explicit residual category.
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, name in enumerate(MODELS):
        row = study.maj7.set_index('model').loc[name]; count = row['support']
        ax.bar(j, row['same_root_maj'] / count * 100, color='#D87949', label='Same-root ordinary major' if j == 0 else None)
        ax.bar(j, 100 - row['same_root_maj'] / count * 100, bottom=row['same_root_maj'] / count * 100, color='#999999', label='Other outcomes' if j == 0 else None)
    ax.set_xticks(range(3), [DISPLAY[n] for n in MODELS]);ax.set(ylabel='Maj7 references (%)', ylim=(0, 100), title='Major-seventh failure replacement');ax.legend();save(fig, 'maj7_failure_replacements')
    heat_panels([study.root_confusions[n] for n in MODELS], study.root_labels, [DISPLAY[n] + ' root' for n in MODELS], 'root_confusions')
    fig, ax = plt.subplots(figsize=(10, 5)); bottom = np.zeros(3)
    cats = study.errors['category'].unique(); colors = ['#5AAE78', '#9AC5E3', '#E6A65A', '#B889BA', '#D66B64', '#888888']
    for category, color in zip(cats, colors):
        values = [float(study.errors[(study.errors.model == n) & (study.errors.category == category)].iloc[0]['fraction_of_all_targets']) * 100 for n in MODELS]
        ax.bar(range(3), values, bottom=bottom, color=color, label=category);bottom += values
    ax.set_xticks(range(3), [DISPLAY[n] for n in MODELS]);ax.set(ylabel='All valid targets (%)', ylim=(0, 100), title='Mutually exclusive exact/root/quality outcomes')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9);save(fig, 'error_category_stack')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, name in zip(axes, MODELS):
        rows = study.confused_pairs[study.confused_pairs.model == name].head(8).iloc[::-1]
        ax.barh([a + ' -> ' + b for a, b in zip(rows.reference, rows.prediction)], rows['count'], color=COLORS[name]); ax.set(title=DISPLAY[name], xlabel='Incorrect predictions', xlim=(0, None))
    save(fig, 'most_confused_chord_pairs')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, MODELS):
        rows = study.training_epochs[study.training_epochs.model == name]
        ax.plot(rows.epoch, rows.training_full_chord_ce, '-o', color=COLORS[name], label='Training full-chord CE')
        ax.plot(rows.epoch, rows.validation_full_chord_ce, '--o', color=COLORS[name], label='Validation full-chord CE')
        ax.axvline(study.previous[name]['best_epoch'], color='#777777', ls=':');ax.set(title=DISPLAY[name], xlabel='Epoch', ylabel='Unweighted full-chord CE', ylim=(0, None));ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig, 'comparable_training_loss')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in MODELS:
        rows = study.training_epochs[study.training_epochs.model == name]
        axes[0].plot(rows.epoch, rows.validation_accuracy * 100, '-o', color=COLORS[name], label=DISPLAY[name])
        axes[1].plot(rows.epoch, rows.accuracy_generalization_gap * 100, '-o', color=COLORS[name], label=DISPLAY[name])
    gap_bound = max(5., float(study.training_epochs.accuracy_generalization_gap.abs().max()) * 120)
    axes[0].set(xlabel='Epoch', ylabel='Validation exact accuracy (%)', ylim=(0, 100));axes[1].set(xlabel='Epoch', ylabel='Train minus validation accuracy (pp)', ylim=(-gap_bound, gap_bound))
    axes[1].axhline(0, color='#777777');axes[0].legend();axes[1].legend();save(fig, 'validation_accuracy_and_gap')
    grouped(study.efficiency, ['parameters'], ['Trainable model parameters'], 'parameter_counts', 'Parameters', False)
    grouped(study.efficiency, ['mean_batch_latency_ms', 'median_batch_latency_ms', 'p95_batch_latency_ms'], ['Mean', 'Median', '95th percentile'], 'inference_latency', 'Forward latency per batch (ms)', False)
    grouped(study.efficiency, ['valid_beats_per_second'], ['Valid supervised beats / second'], 'inference_throughput', 'Valid beats per second', False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, marker in zip(MODELS, ('o', 's', '^')):
        x = study.efficiency.set_index('model').loc[name, 'mean_batch_latency_ms']; y = study.results[name]['metrics']['accuracy'] * 100
        ax.scatter(x, y, s=85, marker=marker, color=COLORS[name], label=DISPLAY[name])
    ax.set(xlabel='New forward-only mean latency (ms/batch)', ylabel='Exact accuracy (%)',
           xlim=(0, float(study.efficiency.mean_batch_latency_ms.max()) * 1.18), ylim=(0, 100))
    ax.legend(loc='upper left');ax.grid(alpha=.2);save(fig, 'accuracy_latency_tradeoff')
    grouped(study.melody, ['note_overlap_weighted_compatibility'], ['Common-event melody chord-tone membership'], 'melody_compatibility', 'Overlap-duration compatibility (%)')
    grouped(study.harmonic, ['predicted_all_tones_in_key_fraction', 'function_agreement_with_basic_context', 'root_movement_agreement'], ['All tones in major/natural-minor key', 'Basic-context function agreement', 'Reference root-movement agreement'], 'harmonic_compatibility')
    genres = study.genres[study.genres.model == 'baseline'].sort_values('support', ascending=False).genre.tolist()
    fig, ax = plt.subplots(figsize=(max(10, len(genres) * .45), 3.8)); values = np.array([[study.genres[(study.genres.model == name) & (study.genres.genre == g)].iloc[0].accuracy * 100 for g in genres] for name in MODELS])
    im = ax.imshow(values, vmin=0, vmax=100, aspect='auto', cmap='Blues');ax.set_xticks(range(len(genres)), genres, rotation=60, ha='right');ax.set_yticks(range(3), [DISPLAY[n] for n in MODELS]);ax.set_title('Original multi-label genre groups: exact chord accuracy');fig.colorbar(im, ax=ax, label='Exact accuracy (%)');save(fig, 'genre_accuracy_heatmap')
    top_genres = genres[:12]; fig, ax = plt.subplots(figsize=(13, 5));x = np.arange(len(top_genres))
    for j, name in enumerate(MODELS):
        ax.bar(x + (j - 1) * .24, [study.genres[(study.genres.model == name) & (study.genres.genre == g)].iloc[0].accuracy * 100 for g in top_genres], .24, color=COLORS[name], label=DISPLAY[name])
    ax.set_xticks(x, top_genres, rotation=40, ha='right');ax.set(ylabel='Exact accuracy (%)', ylim=(0, 100), title='Twelve largest test genre groups; groups overlap');ax.legend();save(fig, 'genre_accuracy_bars')
    for metric, label in [('accuracy', 'Exact accuracy'), ('macro_f1', 'Macro F1')]:
        rows = study.bootstrap[study.bootstrap.metric == metric]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5)); scale = 100 if metric == 'accuracy' else 1
        for ax, differences in zip(axes, (False, True)):
            sub = rows[rows.comparison.str.contains(' minus ') == differences]; yy = np.arange(len(sub))
            ax.hlines(yy, sub.ci_low * scale, sub.ci_high * scale, color='#555555')
            ax.scatter(sub.point_estimate * scale, yy, color='#2878B5');ax.axvline(0, ls=':', color='#777777')
            ax.set_yticks(yy, list(sub.comparison));ax.set_xlabel(label + (' difference (pp)' if metric == 'accuracy' and differences else ' (%)' if metric == 'accuracy' else ' difference' if differences else ''))
            ax.set_title('Paired differences: 95% track intervals' if differences else 'Model scores: 95% track intervals')
            if differences:
                bound = max(abs(sub.ci_low.min()), abs(sub.ci_high.max())) * scale * 1.15
                ax.set_xlim(-max(bound, 1e-6), max(bound, 1e-6))
            else: ax.set_xlim(0, 100 if metric == 'accuracy' else 1)
        save(fig, 'bootstrap_' + metric)
    # Record every plotted raw confusion in CSV and JSON, not only PNGs.
    for name in MODELS:
        save_table(study.tables, name + '_quality_confusion', [{'reference': q, **{p: int(study.results[name]['quality_confusion'][i, j]) for j, p in enumerate(study.quality_labels)}} for i, q in enumerate(study.quality_labels)])
        save_table(study.tables, name + '_root_confusion', [{'reference': q, **{p: int(study.root_confusions[name][i, j]) for j, p in enumerate(study.root_labels)}} for i, q in enumerate(study.root_labels)])
    save_json(study.output / 'figure_inventory.json', figures)
    return figures


def finalize(study):
    """Measured report, integrity audit and compact presentation table."""
    efficiency = {r['model']: r for r in read_json(study.tables / 'efficiency_comparison.json')}
    training = {r['model']: r for r in read_json(study.tables / 'training_behavior_summary.json')}
    rows = []
    for name in MODELS:
        row = dict(study.results[name]['metrics'])
        row.update({k: efficiency[name][k] for k in ('parameters', 'trainable_parameters', 'checkpoint_bytes',
            'mean_batch_latency_ms', 'median_batch_latency_ms', 'p95_batch_latency_ms', 'valid_beats_per_second', 'peak_process_rss_bytes')})
        row.update({k: training[name][k] for k in ('best_epoch', 'completed_epochs', 'recorded_train_validation_seconds')})
        for quality in ('maj', 'min', 'maj7', 'min7', '7'):
            qrow = next(r for r in study.results[name]['quality_rows'] if r['quality'] == quality)
            row[quality + '_exact_accuracy'] = qrow['exact_full_chord_accuracy']
        rows.append(row)
    study.comparison = save_table(study.output, 'model_comparison', rows)
    winners = []
    for metric in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall',
                   'weighted_f1', 'micro_f1', 'cross_entropy', 'top3_accuracy', 'top5_accuracy', 'mean_reciprocal_rank',
                   'mean_reference_probability', 'mean_batch_latency_ms', 'valid_beats_per_second']:
        smaller = metric in ('cross_entropy', 'mean_batch_latency_ms')
        best = min(r[metric] for r in rows) if smaller else max(r[metric] for r in rows)
        winners.append({'metric': metric, 'best_models': [r['model'] for r in rows if math.isclose(r[metric], best, abs_tol=1e-12)],
                        'best_value': best, 'direction': 'minimize' if smaller else 'maximize'})
    study.winners = save_table(study.tables, 'best_model_by_metric', winners)
    pp1keys = ['accuracy', 'macro_f1', 'weighted_f1', 'top3_accuracy', 'top5_accuracy', 'cross_entropy', 'maj7_exact_accuracy', 'min7_exact_accuracy']
    pp1 = [{'metric': k, **{r['model']: r[k] for r in rows}} for k in pp1keys]
    save_table(study.tables, 'pp1_presentation', pp1)
    bs = read_json(study.tables / 'track_bootstrap_confidence.json')
    delta = next(r for r in bs if r['comparison'] == 'multitask minus baseline' and r['metric'] == 'accuracy')
    lines = ['# Comprehensive Transformer evaluation', '',
        f'All three best checkpoints were freshly evaluated on the same {len(study.y):,} valid beats, {len(study.data):,} sequences and {len(study.track_ids)} test tracks. No training, vocabulary fitting, data rebuilding, candidate filtering or quality re-ranking occurred.', '',
        '| Metric | Baseline | Weighted | Multi-task |', '|---|---:|---:|---:|']
    for row in pp1:
        key = row['metric']; percent = 'accuracy' in key
        lines.append('| ' + key + ' | ' + ' | '.join(f'{row[n]*100:.2f}%' if percent else f'{row[n]:.4f}' for n in MODELS) + ' |')
    lines += ['', '## Main observations', '',
        f'Multi-task minus baseline exact accuracy: {delta["point_estimate"]*100:+.3f} percentage points; paired 95% track-bootstrap interval [{delta["ci_low"]*100:+.3f}, {delta["ci_high"]*100:+.3f}] points. '
        + ('The interval excludes zero, conditional on the stated resampling assumptions.' if delta['interval_excludes_zero'] else 'The interval includes zero; this test population does not establish a reliable accuracy advantage.'),
        '', 'Major-seventh results (individual exact labels, not broad major-family scores):']
    for row in read_json(study.tables / 'maj7_comparison.json'):
        lines.append(f'- {DISPLAY[row["model"]]}: exact {row["exact_correct"]}/{row["support"]}, top-3 {row["top3_exact_accuracy"]:.2%}, top-5 {row["top5_exact_accuracy"]:.2%}; mean reference probability {row["mean_reference_probability"]:.2%}.')
    aux = study.auxiliary['metrics']; aq = next(r for r in study.auxiliary['rows'] if r['quality'] == 'maj7')
    lines += ['', f'Auxiliary head only: quality accuracy {aux["accuracy"]:.2%}, macro F1 {aux["macro_f1"]:.4f}. Maj7 quality recall {aq["recall"]:.3%}, precision {aq["precision"]:.2%}. These are not full-chord accuracies.',
        '', '## Best model by metric', '']
    lines += [f'- {r["metric"]}: ' + ', '.join(DISPLAY[n] for n in r['best_models']) for r in winners]
    lines += ['', '## Musical diagnostics and efficiency', '',
        'Chord-tone membership is overlap-duration weighted, with a common eligible population across models. Key compatibility uses explicit major/natural-minor chord-tone membership. Function agreement is against basic-derived context and is not independent harmonic truth. Non-chord tones and non-diatonic chords may be musically appropriate. See musical_diagnostic_rules.json for exact rules and tables for denominators.', '',
        'Genre analyses use overlapping original annotation groups. They describe reference-label prediction, not causal evidence that genre inputs improve generation or stylistic listening quality.', '',
        'Inference timing is new forward-only measurement after warm-up, with the same CPU/device, threads, batch size and saved test data. Each model is measured in its own fresh worker process. Loading/transfers/CE/top-k are excluded; both multitask heads are included. Process peak memory includes libraries/model/cached data, not just activations. Historical training duration is a sum of logged epoch timers, not inferred from epoch counts.', '',
        '## Limitations', '',
        '- Single training seed per approach: initialization and training-run variability have not been measured.',
        '- Only 32 test tracks; long tracks contribute more beats. Track bootstrap does not account fully for shared artist/album dependence. Macro-F1 intervals use a fixed original test class set and are sensitive to classes absent from resamples.',
        '- Reference chords and melody/key/beat estimates are automatic pseudo-labels; basic inputs are reference-derived synthetic simplifications.',
        '- Masked/unseen targets are excluded; N is included in objective metrics and excluded from pitch-class compatibility when no tone set is defined.',
        '- Quality-only correctness does not imply exact root/extension/inversion correctness. Musical proxies and top-k candidates are not listening-test results.',
        '- CPU throughput fluctuates with scheduling, thermal state and other processes. Historical training and newly measured inference durations are different measurements.', '',
        '## Recommended next experiments', '',
        '1. Repeat training across multiple seeds before declaring an overall winner; keep the held-out test set out of tuning decisions.',
        '2. Audit maj7 pseudo-labels, per-track distribution and discriminating melody/context evidence. Test validation-selected context/loss ablations; the auxiliary task has not solved maj7 exact prediction.',
        '3. Evaluate independently generated basic progressions and use expert listening or a defined compatibility study for musical suitability.',
        '4. If candidate suggestions are the application, evaluate validation-designed top-k selection and user choices separately from raw top-1 classification.', '',
        'All numerical tables, raw top-five predictions and uncertainty samples are saved with this run. Zero-support/unmeasurable entries remain unavailable rather than invented.']
    (study.output / 'evaluation_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report = {'completed': ['Three compatible best checkpoints loaded', 'Paired masked inference and metric reproduction',
        'Complete-class and quality-only metrics', 'Full-head and auxiliary quality confusion', 'Root/quality/inversion error partitions',
        'Top-k/MRR/probability ranks and real examples', 'Saved training-history comparison', 'Isolated-process warmed repeated inference benchmark',
        'Explicit melody/key/function/genre diagnostics', 'Paired track bootstrap for accuracy and macro F1', 'Saved figures, CSV/JSON tables and presentation summary'],
        'not_measured': {'listening_quality': 'No listening-test data supplied; no subjective score invented.',
            'GPU_latency_or_memory': 'CPU experiment; CUDA metrics unavailable here.',
            'activation_only_peak_CPU_memory': 'No reliable tensor-only native allocation profiler used; report isolated process peak working set instead.',
            'full_job_training_wall_time': 'Only actual logged per-epoch timers available; no inference from epoch count.',
            'training_seed_uncertainty': 'One existing checkpoint per approach; retraining prohibited.',
            'causal_genre_effect': 'No ablation/retraining performed; observational overlapping genre groups only.',
            'mcnemar': 'Optional test omitted because beat independence is inappropriate; paired track bootstrap provided.',
            'seaborn': 'Not installed; Matplotlib supplies all heatmaps and charts without adding dependencies.'},
        'source_modules': ['model/checkpoint.py', 'model/multitask_transformer.py', 'model/config.py', 'transformer_dataset.py',
            'weighted_training.py', 'normalize_chords.py', 'canonicalize_chords.py', 'build_synthetic_dataset.py'],
        'raw_probabilities_policy': 'Only top-five and reference-class probabilities stored; no full probability cube.',
        'new_files': ['notebooks/11_comprehensive_model_evaluation.ipynb', 'src/comprehensive_evaluation.py', 'tests/test_comprehensive_evaluation.py']}
    save_json(study.output / 'implementation_report.json', report)
    (study.output / 'implementation_report.md').write_text('# Implementation report\n\n' + '\n'.join('- ' + s for s in report['completed']) +
        '\n\n## Unavailable or deliberately unmeasured\n\n' + '\n'.join('- ' + k + ': ' + v for k, v in report['not_measured'].items()) + '\n', encoding='utf-8')
    print('Verifying protected existing artifacts...', flush=True)
    for relative, expected in study.protected.items():
        assert sha(study.root / relative) == expected, relative
    for relative, expected in study.ready['artifacts'].items():
        assert sha(study.dataset / relative) == expected, relative
    # Check finite saved metrics; intentional unavailable values are JSON null / empty CSV cells.
    json_files = list(study.tables.glob('*.json')) + [study.output / 'model_comparison.json']
    for path in json_files: clean(read_json(path))
    assert all(study.results[n]['metrics']['valid_targets'] == len(study.y) for n in MODELS)
    assert all((study.results[n]['confusion'].sum() == len(study.y)) for n in MODELS)
    assert all(study.errors[study.errors.model == n]['count'].sum() == len(study.y) for n in MODELS)
    inventory = {p.relative_to(study.output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in study.output.rglob('*') if p.is_file() and p.name not in ('artifact_inventory.json', 'verification.json')}
    save_json(study.output / 'artifact_inventory.json', inventory)
    verification = {'status': 'complete', 'same_targets_all_models': True, 'valid_targets_per_model': len(study.y),
        'test_tracks': len(study.track_ids), 'published_metrics_reproduced': True, 'all_predictions_finite': True,
        'existing_artifacts_unchanged': len(study.protected), 'dataset_artifact_hashes_verified': len(study.ready['artifacts']),
        'saved_figures': len(list(study.figures.glob('*.png'))), 'saved_csv_tables': len(list(study.tables.glob('*.csv'))) + 1,
        'no_retraining': True, 'no_reranking': True, 'paired_event_identity_sha256': study.identity_hash}
    save_json(study.output / 'verification.json', verification)
    return study.comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-worker', type=Path, required=True)
    benchmark_worker(parser.parse_args().benchmark_worker)
