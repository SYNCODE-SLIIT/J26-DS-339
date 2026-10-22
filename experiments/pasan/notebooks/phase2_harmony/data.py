"""Conservative cleaning, beat alignment and portable dataset snapshots.

Only melody, metrical positions, mode and genre become model inputs. Chords are
used for targets/quality selection, never for melody correction or features.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd

FAMILIES = ('MAJ', 'MIN', 'DOM7', 'DIM', 'HDIM7', 'DIM7', 'AUG', 'SUS2', 'SUS4')
DEGREES = ('I', 'bII', 'II', 'bIII', 'III', 'IV', '#IV', 'V', 'bVI', 'VI', 'bVII', 'VII')
MINOR_FAMILIES = {'MIN', 'DIM', 'HDIM7', 'DIM7'}
LABELS = [f'{d.lower() if q in MINOR_FAMILIES else d}:{q}' for d in DEGREES for q in FAMILIES]
IGNORE = -100
FILES = {'melody': 'melody/combined.json', 'beats': 'beats/beats.json',
         'roman': 'roman/submission.json', 'key': 'key.json', 'comparison': 'key_comparison.json'}
FEATURE_DIM = 77
MELODY_DIM = 72


@dataclass(frozen=True)
class CleanConfig:
    window_bars: int = 8
    min_note_seconds: float = .03
    min_melody_coverage: float = .50
    min_note_count: int = 10
    max_gap_beats: float = 16.
    min_chord_coverage: float = .80
    min_chord_dominance: float = .65
    min_valid_target_fraction: float = .70
    strict_key: bool = True
    # Turn this on only for a documented sensitivity experiment, not silently.
    repair_isolated_octaves: bool = False


def load_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def interval_stats(intervals, start, end):
    """Union coverage and largest uncovered gap: never double-count overlaps."""
    cursor, covered, gap = start, 0., 0.
    for a, b in sorted(intervals, key=lambda pair: (pair[0], pair[1])):
        a, b = max(start, a), min(end, b)
        if b <= a:
            continue
        gap = max(gap, a - cursor)
        covered += max(0., b - max(a, cursor))
        cursor = max(cursor, b)
    return covered / (end - start), max(gap, end - cursor)


def clean_notes(notes, duration, config=CleanConfig()):
    """Remove invalid/duplicate/micro events; preserve rests and chromaticism.

    Same-pitch overlapping fragments are merged. Different-pitch overlaps stay
    explicit (and are audited), because choosing a voice needs audio evidence.
    A short isolated +/- octave candidate is flagged; repair is opt-in.
    """
    clean, actions, seen = [], [], set()
    for i, n in enumerate(notes):
        reason = None
        try:
            a, b, p = float(n['onset_sec']), float(n['offset_sec']), float(n['midi_pitch_rounded'])
            if not np.isfinite([a, b, p]).all() or not 0 <= p <= 127 or p != round(p) or b <= a:
                reason = 'invalid_event'
            else:
                a, b, p = max(0., a), min(duration, b), int(p)
                if b - a < config.min_note_seconds:
                    reason = 'too_short_or_outside'
                elif (a, b, p) in seen:
                    reason = 'exact_duplicate'
        except (KeyError, TypeError, ValueError):
            reason = 'malformed_event'
        if reason:
            actions.append({'index': i, 'action': reason})
            continue
        seen.add((a, b, p))
        clean.append({'onset_sec': a, 'offset_sec': b, 'pitch': p,
                      'source': n.get('source', 'unknown')})
    clean.sort(key=lambda n: (n['onset_sec'], n['offset_sec'], n['pitch']))
    merged = []
    for n in clean:
        if (merged and n['pitch'] == merged[-1]['pitch'] and
                n['onset_sec'] < merged[-1]['offset_sec']):
            merged[-1]['offset_sec'] = max(n['offset_sec'], merged[-1]['offset_sec'])
            actions.append({'onset_sec': n['onset_sec'], 'action': 'merge_same_pitch_overlap'})
        else:
            merged.append(n.copy())
    for i in range(1, len(merged) - 1):
        prev, n, nxt = merged[i-1:i+2]
        jump = n['pitch'] - prev['pitch']
        if (abs(jump) == 12 and abs(nxt['pitch'] - prev['pitch']) <= 2 and
                n['offset_sec'] - n['onset_sec'] <= .18 and
                0 <= n['onset_sec'] - prev['offset_sec'] <= .1 and
                0 <= nxt['onset_sec'] - n['offset_sec'] <= .1):
            actions.append({'onset_sec': n['onset_sec'], 'pitch': n['pitch'],
                            'action': 'repair_octave' if config.repair_isolated_octaves else 'flag_octave'})
            if config.repair_isolated_octaves:
                n['pitch'] -= jump
    return merged, actions


def melody_features(notes, boundaries, tonic_pc, beats=None):
    """4 ordered sub-beat slots + onsets + meter. Key/octave invariant.

    Each slot: 12 duration-weighted pitch classes, union voiced coverage,
    duration-weighted contour relative to the FIRST window note / 24 semitones,
    onset count / 4. Last 17: 12 onset pitch classes + five meter features.
    Notes use continuous beat units; rests are zero with zero coverage.
    """
    bounds = np.asarray(boundaries, float)
    if bounds.ndim != 1 or len(bounds) < 2 or not np.isfinite(bounds).all() or np.any(np.diff(bounds) <= 0):
        raise ValueError('Beat boundaries must be finite and strictly increasing.')
    if not isinstance(tonic_pc, (int, np.integer)) or not 0 <= tonic_pc < 12:
        raise ValueError('tonic_pc must be an integer in [0, 11].')
    count = len(bounds) - 1
    if beats is None:
        beats = [{'beat_in_bar': i % 4 + 1, 'meter_numerator': 4, 'meter_denominator': 4} for i in range(count)]
    if len(beats) != count:
        raise ValueError('One metrical record is required per beat.')
    x = np.zeros((count, FEATURE_DIM), np.float32)
    selected = sorted((n for n in notes if n['onset_sec'] < bounds[-1] and n['offset_sec'] > bounds[0]),
                      key=lambda n: n['onset_sec'])
    anchor = selected[0]['pitch'] if selected else 0
    packed = []
    for n in selected:
        a, b = np.interp([n['onset_sec'], n['offset_sec']], bounds, np.arange(len(bounds)))
        if b > a:
            packed.append((a, b, (n['pitch'] - tonic_pc) % 12, np.clip((n['pitch'] - anchor) / 24., -2, 2)))
    for i in range(count):
        for j in range(4):
            left, right = i + j/4, i + (j+1)/4
            slot, mass, contour, spans, onsets = np.zeros(12), 0., 0., [], 0
            for a, b, pc, delta in packed:
                overlap = max(0., min(right, b) - max(left, a))
                if overlap:
                    slot[pc] += overlap
                    mass += overlap
                    contour += overlap * delta
                    spans.append((a, b))
                if left <= a < right:
                    onsets += 1
            offset = 15*j
            x[i, offset:offset+12] = slot / max(mass, .25)
            x[i, offset+12] = interval_stats(spans, left, right)[0]
            x[i, offset+13] = contour / max(mass, 1e-8)
            x[i, offset+14] = min(onsets / 4., 1.)
        for a, b, pc, delta in packed:
            if i <= a < i+1:
                x[i, 60+pc] += 1
        x[i, 60:72] = np.minimum(x[i, 60:72] / 4., 1.)
        beat = beats[i]
        phase = 2*np.pi*(beat['beat_in_bar']-1)/beat['meter_numerator']
        x[i, 72:] = [np.sin(phase), np.cos(phase), float(beat['beat_in_bar'] == 1),
                     beat['meter_numerator']/12., beat['meter_denominator']/8.]
    return x


def chord_targets(events, boundaries, dominance=.65):
    """Dominant duration per beat; unknown/N/ambiguous spans stay masked."""
    targets, certainty, coverages = [], [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        overlaps, spans = defaultdict(float), []
        for e in events:
            if e.get('mapping_status') != 'mapped' or e.get('family') not in FAMILIES:
                continue
            root = e.get('relative_root_semitones')
            if not isinstance(root, int) or not 0 <= root < 12:
                continue
            a, b = max(left, e['start_sec']), min(right, e['end_sec'])
            if b > a:
                overlaps[root*len(FAMILIES)+FAMILIES.index(e['family'])] += b-a
                spans.append((a, b))
        best = max(overlaps, key=overlaps.get) if overlaps else IGNORE
        fraction = overlaps[best] / (right-left) if overlaps else 0.
        targets.append(best if fraction >= dominance else IGNORE)
        certainty.append(fraction)
        coverages.append(interval_stats(spans, left, right)[0])
    return np.array(targets, np.int64), np.array(certainty, np.float32), float(np.mean(coverages))


def complete_windows(beats, config):
    """Consecutive complete bars, fixed meter, non-overlap, true next boundary."""
    groups = defaultdict(list)
    for i, b in enumerate(beats):
        groups[b['bar_index']].append(i)
    run, previous_meter, previous_bar = [], None, None
    runs = []
    for bar, indices in sorted(groups.items()):
        entries = [beats[i] for i in indices]
        meters = {(b['meter_numerator'], b['meter_denominator']) for b in entries}
        meter = next(iter(meters)) if len(meters) == 1 else None
        valid = (meter in {(2, 4), (3, 4), (4, 4), (6, 8)} and len(entries) == meter[0]
                 and [b['beat_in_bar'] for b in entries] == list(range(1, meter[0]+1)))
        if not valid or meter != previous_meter or previous_bar is None or bar != previous_bar+1:
            if run:
                runs.append(run)
            run = []
        if valid:
            run.append(indices)
        previous_meter, previous_bar = meter, bar
    if run:
        runs.append(run)
    for run in runs:
        for i in range(0, len(run)-config.window_bars+1, config.window_bars):
            idx = [j for block in run[i:i+config.window_bars] for j in block]
            if idx[-1]+1 < len(beats):
                yield idx, np.array([beats[j]['time_sec'] for j in idx] + [beats[idx[-1]+1]['time_sec']])


def build_snapshot(project_root, destination, config=CleanConfig()):
    """Audit every extracted folder, save strict clean examples and rejections.

    Existing snapshots are verified and reused. Change destination for a new
    policy; a partial or stale snapshot fails rather than silently resuming.
    """
    project_root, destination = Path(project_root), Path(destination)
    pipeline = project_root/'datasets/mtg-jamendo/melody-chord-extraction-pipeline'
    if destination.exists():
        meta = verify_snapshot(destination, verify_raw=True, project_root=project_root)
        if meta['clean_config'] != asdict(config):
            raise ValueError('Existing snapshot uses a different cleaning policy; use a new destination.')
        return meta
    manifest = pd.read_csv(pipeline/'manifest.csv')
    if manifest.mtg_track_id.duplicated().any():
        raise ValueError('Manifest track join is not unique.')
    meta_by_id = manifest.set_index('mtg_track_id').to_dict('index')
    audit, actions, windows, examples, hashes, cleaned = [], [], [], [], {}, []
    hashes['manifest.csv'] = file_hash(pipeline/'manifest.csv')
    for folder in sorted((pipeline/'outputs').glob('track_*')):
        if not folder.is_dir():
            continue
        tid = folder.name
        row = {'track_id': tid, 'status': 'rejected'}
        audit.append(row)
        missing = [name for name, rel in FILES.items() if not (folder/rel).is_file()]
        if missing:
            row['reason'] = 'missing:'+','.join(missing)
            continue
        try:
            obj = {k: load_json(folder/v) for k, v in FILES.items()}
            for rel in FILES.values():
                hashes[f'outputs/{tid}/{rel}'] = file_hash(folder/rel)
            key = obj['key']['melody_key'].get('selected') or {}
            chord_key = (obj['comparison'].get('chord_key') or {}).get('selected') or {}
            row.update(mode=key.get('mode'), melody_confidence=key.get('confidence'),
                       chord_confidence=chord_key.get('confidence'), agreement=obj['comparison'].get('agreement'))
            metadata = meta_by_id.get(tid)
            if not metadata or pd.isna(metadata['artist_id']) or metadata['split_0'] not in {'train', 'validation', 'test'}:
                row['reason'] = 'missing_metadata'
                continue
            row.update(artist_id=metadata['artist_id'], split=metadata['split_0'])
            notes, beats, events = obj['melody']['notes'], obj['beats']['beats'], obj['roman']['events']
            duration = float(obj['melody']['duration_sec'])
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError('invalid_duration')
            clean, edits = clean_notes(notes, duration, config)
            actions.extend({'track_id': tid, **e} for e in edits)
            row.update(raw_notes=len(notes), cleaned_notes=len(clean), cleaning_actions=len(edits),
                       octave_candidates=sum(e['action'] in {'flag_octave', 'repair_octave'} for e in edits),
                       beats=len(beats), chord_events=len(events),
                       melody_coverage=interval_stats([(n['onset_sec'], n['offset_sec']) for n in clean], 0, duration)[0],
                       vocal_fraction=np.mean([n['source'] == 'game_vocal' for n in clean]) if clean else 0,
                       overlapping_notes=sum(a['offset_sec'] > b['onset_sec'] for a, b in zip(clean[:-1], clean[1:])),
                       large_leaps=sum(abs(a['pitch']-b['pitch']) > 12 for a,b in zip(clean[:-1],clean[1:])))
            # Validate alignment and source freshness before admitting supervision.
            if len(beats) < 2 or not np.isfinite([b['time_sec'] for b in beats]).all() or np.any(np.diff([b['time_sec'] for b in beats]) <= 0):
                raise ValueError('invalid_beat_timeline')
            if any(not np.isfinite([e['start_sec'],e['end_sec']]).all() or e['end_sec'] <= e['start_sec'] for e in events):
                raise ValueError('invalid_chord_timeline')
            if any(a['end_sec'] > b['start_sec'] + 1e-5 for a,b in zip(events[:-1],events[1:])):
                raise ValueError('overlapping_or_unordered_chords')
            if abs(float(obj['roman']['duration_sec'])-duration) > .1 or abs(float(obj['beats']['duration_sec'])-duration) > .1:
                raise ValueError('inconsistent_duration')
            if obj['key'].get('melody_json_sha256') != hashes[f'outputs/{tid}/melody/combined.json']:
                raise ValueError('stale_melody_key')
            if obj['roman'].get('key_json_sha256') != hashes[f'outputs/{tid}/key.json']:
                raise ValueError('stale_roman_key')
            if obj['comparison'].get('melody_key_json_sha256') != hashes[f'outputs/{tid}/key.json']:
                raise ValueError('stale_comparison')
            chord_file = folder/'chords/submission.json'
            if chord_file.exists():
                hashes[f'outputs/{tid}/chords/submission.json'] = file_hash(chord_file)
                if (obj['roman'].get('chord_json_sha256') != file_hash(chord_file) or
                        obj['comparison'].get('submission_chord_json_sha256') != file_hash(chord_file)):
                    raise ValueError('stale_chord_derivatives')
            if not key or key.get('mode') not in {'major', 'minor'} or key.get('tonic_pc') not in range(12):
                raise ValueError('missing_key')
            roman_key = obj['roman'].get('key') or {}
            if any(roman_key.get(k) != key.get(k) for k in ['tonic_pc', 'mode']):
                raise ValueError('different_normalization_key')
            if config.strict_key and not (key['confidence'] == chord_key.get('confidence') == 'high'
                                          and obj['comparison'].get('agreement') == 'agree'):
                row['reason'] = 'key_uncertain_or_disagree'
                continue
            genres = sorted(t.removeprefix('genre---') for t in str(metadata.get('tags','')).split(';') if t.startswith('genre---'))
            row['status'], row['reason'] = 'candidate', ''
            cleaned.append({'track_id': tid, 'notes': clean})
            for indices, bounds in complete_windows(beats, config):
                ns = [n for n in clean if n['onset_sec'] < bounds[-1] and n['offset_sec'] > bounds[0]]
                coverage, gap_sec = interval_stats([(n['onset_sec'], n['offset_sec']) for n in ns], bounds[0], bounds[-1])
                beat_spans = [np.interp([n['onset_sec'], n['offset_sec']], bounds, np.arange(len(bounds))) for n in ns]
                _, gap_beats = interval_stats(beat_spans, 0, len(indices))
                y, confidence, ccov = chord_targets(events, bounds, config.min_chord_dominance)
                reasons = []
                if coverage < config.min_melody_coverage: reasons.append('sparse_melody')
                if len(ns) < config.min_note_count: reasons.append('few_notes')
                if gap_beats > config.max_gap_beats: reasons.append('large_gap')
                if ccov < config.min_chord_coverage: reasons.append('sparse_chords')
                if np.mean(y >= 0) < config.min_valid_target_fraction: reasons.append('ambiguous_targets')
                w = {'track_id': tid, 'artist_id': metadata['artist_id'], 'split': metadata['split_0'],
                     'start_sec': float(bounds[0]), 'end_sec': float(bounds[-1]), 'n_beats': len(indices),
                     'mode': key['mode'], 'genres': genres, 'note_count': len(ns), 'coverage': coverage,
                     'gap_sec': gap_sec, 'gap_beats': gap_beats, 'chord_coverage': ccov,
                     'valid_target_fraction': float(np.mean(y>=0)), 'rejection': ','.join(reasons),
                     'accepted': not reasons, 'example_index': -1}
                windows.append(w)
                if reasons:
                    continue
                x = melody_features(ns, bounds, key['tonic_pc'], [beats[j] for j in indices])
                fingerprint = sha256(np.round(x,4).tobytes()+y.tobytes()).hexdigest()
                w['content_hash'] = fingerprint
                w['example_index'] = len(examples)
                examples.append((x, y, confidence, bounds, w))
        except (ValueError, KeyError, TypeError, OSError) as e:
            row['status'], row['reason'] = 'rejected', f'{type(e).__name__}:{e}'
    # Quarantine exact normalized window duplicates crossing official splits.
    fingerprint_splits = defaultdict(set)
    for *_, w in examples:
        fingerprint_splits[w['content_hash']].add(w['split'])
    filtered = []
    for example in examples:
        w = example[-1]
        if len(fingerprint_splits[w['content_hash']]) > 1:
            w.update(accepted=False, rejection='cross_split_exact_duplicate', example_index=-1)
        else:
            w['example_index'] = len(filtered)
            filtered.append(example)
    examples = filtered
    if not examples:
        raise ValueError(f'No examples survived. Track reasons: {Counter(r.get("reason") for r in audit)}; '
                         f'window reasons: {Counter(w["rejection"] for w in windows)}')
    window_df = pd.DataFrame(windows)
    accepted = window_df[window_df.accepted]
    if accepted.groupby('artist_id').split.nunique().max() != 1:
        raise ValueError('Artist leakage across official splits.')
    for split in ['train','validation','test']:
        if not (accepted.split == split).any():
            raise ValueError(f'No examples in {split}.')
    training_tracks = accepted[accepted.split == 'train'].drop_duplicates('track_id')
    genre_counts = Counter(g for gs in training_tracks.genres for g in gs)
    genre_vocab = ['<unknown>'] + sorted(g for g,c in genre_counts.items() if c >= 3)
    max_beats = max(len(e[1]) for e in examples)
    n = len(examples)
    x = np.zeros((n,max_beats,FEATURE_DIM),np.float32)
    y = np.full((n,max_beats),IGNORE,np.int64)
    confidence = np.zeros((n,max_beats),np.float32)
    lengths, modes = np.zeros(n,np.int64), np.zeros(n,np.int64)
    genres = np.zeros((n,len(genre_vocab)),np.float32)
    for i,(xi,yi,ci,bounds,w) in enumerate(examples):
        length = len(yi)
        x[i,:length], y[i,:length], confidence[i,:length] = xi,yi,ci
        lengths[i],modes[i] = length,int(w['mode']=='minor')
        known = [genre_vocab.index(g) for g in w['genres'] if g in genre_vocab]
        genres[i,known or [0]] = 1
    destination.mkdir(parents=True)
    np.savez_compressed(destination/'features.npz', x=x,y=y,confidence=confidence,lengths=lengths,mode=modes,genres=genres)
    # Boundaries remain portable, for inference examples and listening review.
    pd.DataFrame([{**e[-1], 'beat_boundaries_sec': e[-2].tolist()} for e in examples]).to_parquet(destination/'examples.parquet',index=False)
    window_df.to_parquet(destination/'window_audit.parquet',index=False)
    pd.DataFrame(audit).to_csv(destination/'track_audit.csv',index=False)
    for name,rows in [('note_actions',actions),('cleaned_notes',cleaned)]:
        (destination/f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    metadata = {'schema_version': 1, 'clean_config': asdict(config), 'genre_vocab': genre_vocab,
                'labels': LABELS, 'families': list(FAMILIES), 'feature_dim': FEATURE_DIM,
                'python': platform.python_version(), 'raw_sha256': hashes,
                'summary': {'folders':len(audit),'candidate_tracks':sum(r['status']=='candidate' for r in audit),
                            'accepted_tracks':int(accepted.track_id.nunique()),'accepted_windows':n,
                            'rejected_windows':int((~window_df.accepted).sum()),
                            'splits':accepted.split.value_counts().to_dict()},
                'files_sha256':{p.name:file_hash(p) for p in sorted(destination.iterdir()) if p.is_file()}}
    (destination/'metadata.json').write_text(json.dumps(metadata,indent=2))
    verify_snapshot(destination)
    return metadata


def verify_snapshot(destination, verify_raw=False, project_root=None):
    destination = Path(destination)
    meta = load_json(destination/'metadata.json')
    for name, expected in meta['files_sha256'].items():
        if file_hash(destination/name) != expected:
            raise ValueError(f'Snapshot integrity mismatch: {name}')
    if verify_raw:
        pipeline = Path(project_root)/'datasets/mtg-jamendo/melody-chord-extraction-pipeline'
        for name, expected in meta['raw_sha256'].items():
            if file_hash(pipeline/name) != expected:
                raise ValueError(f'Raw source changed: {name}; build a new versioned snapshot.')
        current = {p.name for p in (pipeline/'outputs').glob('track_*') if p.is_dir()}
        recorded = set(pd.read_csv(destination/'track_audit.csv').track_id)
        if current != recorded:
            raise ValueError('Source folder inventory changed; build a new snapshot.')
        for row in pd.read_csv(destination/'track_audit.csv').itertuples():
            if str(row.reason).startswith('missing:'):
                absent=[k for k,v in FILES.items() if not (pipeline/'outputs'/row.track_id/v).is_file()]
                if ','.join(absent)!=str(row.reason).removeprefix('missing:'):
                    raise ValueError(f'Source completeness changed: {row.track_id}; build a new snapshot.')
    return meta


def load_snapshot(destination):
    destination = Path(destination)
    meta = verify_snapshot(destination)
    with np.load(destination/'features.npz',allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    examples = pd.read_parquet(destination/'examples.parquet')
    assert len(examples) == len(arrays['x'])
    assert np.isfinite(arrays['x']).all()
    assert ((arrays['y'] == IGNORE) | ((arrays['y'] >= 0) & (arrays['y'] < len(LABELS)))).all()
    return arrays,examples,meta
