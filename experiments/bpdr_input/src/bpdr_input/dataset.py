"""Source validation, singer partitions, crop provenance and pending review sheets."""
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from .pipeline import checksum


@dataclass(frozen=True)
class DataPolicy:
    version: str = 'phase2-data-0.1'
    split_seed: int = 339
    max_crop_seconds: float = 20.
    boundary_guard_seconds: float = .10
    annotator_pitch_tolerance_cents: float = 50.
    reference_pitch_tolerance_cents: float = 100.


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def read_array(path, columns):
    values = np.loadtxt(path, delimiter=',', ndmin=2)
    if values.ndim != 2 or values.shape[1] != columns or not len(values) or not np.isfinite(values).all():
        raise ValueError(f'Invalid {columns}-column annotations: {path}')
    return values


def validate_annotations(f0, notes, duration, tolerance):
    errors, warnings = [], []
    if np.any(f0[:, 0] < 0) or np.any(np.diff(f0[:, 0]) <= 0):
        errors.append('F0_TIMESTAMPS_INVALID')
    if np.any(f0[:, 1] < 0) or f0[-1, 0] > duration+tolerance:
        errors.append('F0_VALUES_OR_BOUNDS_INVALID')
    if len(f0) > 2 and np.max(abs(np.diff(f0[:, 0])-np.median(np.diff(f0[:, 0])))) > 1e-6:
        errors.append('F0_GRID_NOT_EVEN')
    if len(f0) < 2:
        errors.append('F0_COVERAGE_INCOMPLETE')
    else:
        hop = float(np.median(np.diff(f0[:, 0])))
        if hop <= 0 or f0[0, 0] > 2*hop+tolerance or duration-f0[-1, 0] > 2*hop+tolerance:
            errors.append('F0_COVERAGE_INCOMPLETE')
    for name, values in notes.items():
        if np.any(values[:, 0] < 0) or np.any(np.diff(values[:, 0]) < 0):
            errors.append(name+'_STARTS_INVALID')
        if np.any(values[:, 1:] <= 0) or np.any(values[:, 0]+values[:, 2] > duration+tolerance):
            errors.append(name+'_VALUES_OR_BOUNDS_INVALID')
        if np.any(values[:-1, 0]+values[:-1, 2] > values[1:, 0]+tolerance):
            warnings.append(name+'_OVERLAPPING_NOTES_MASKED')
    return errors, warnings


def singer_splits(singers, seed=339):
    # Hash order is stable across Python/platform versions and input row ordering.
    ordered = sorted(set(singers), key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
    if len(ordered) < 4:
        raise ValueError('At least four independent singers are required for four partitions')
    n = len(ordered)
    final = max(1, round(n*.10))
    evaluation = max(1, round(n*.20))
    development = max(1, round(n*.20))
    result = {}
    for index, singer in enumerate(ordered):
        result[singer] = ('final_holdout' if index < final else 'pp1_evaluation' if index < final+evaluation
                          else 'development' if index < final+evaluation+development else 'train')
    return result


def crop_ranges(sample_count, sr, notes, max_seconds=20.):
    # Prefer a common rest midpoint. If none exists, an explicit hard edge is flagged.
    intervals = sorted((float(n[0]), float(n[0]+n[2])) for values in notes.values() for n in values)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    rest_midpoints = [(a[1]+b[0])/2 for a, b in zip(merged, merged[1:]) if b[0]-a[1] >= .10]
    max_samples = int(max_seconds*sr)
    start, ranges = 0, []
    while start < sample_count:
        limit = min(start+max_samples, sample_count)
        end = limit
        if limit < sample_count:
            candidates = [round(t*sr) for t in rest_midpoints if start+max_samples//2 < round(t*sr) <= limit]
            if candidates:
                end = max(candidates)
        if end <= start:
            raise ValueError('Non-progressing crop range')
        ranges.append((start, end))
        start = end
    return ranges


def crop_notes(values, start, end):
    rows = []
    for onset, hz, length in values:
        offset = onset+length
        if onset < end and offset > start:
            rows.append(dict(start_seconds=float(max(onset, start)-start), pitch_hz=float(hz),
                             duration_seconds=float(min(offset, end)-max(onset, start)),
                             partial=bool(onset < start or offset > end),
                             original_start_seconds=float(onset), original_duration_seconds=float(length)))
    return rows


def reference_mask(times, f0, notes, policy=DataPolicy()):
    # Do not interpolate across unvoiced reference samples: nearest-frame mask.
    idx = np.clip(np.searchsorted(f0[:, 0], times), 0, len(f0)-1)
    prev = np.maximum(idx-1, 0)
    idx = np.where(abs(f0[prev, 0]-times) < abs(f0[idx, 0]-times), prev, idx)
    pitch = f0[idx, 1]
    hop = float(np.median(np.diff(f0[:, 0]))) if len(f0) > 1 else 0.
    near_reference = (hop > 0) & (abs(f0[idx, 0]-times) <= 1.5*hop+1e-9)
    streams = []
    for name in ('A1', 'A2'):
        active_count = np.zeros(len(times), dtype=int)
        hz = np.zeros(len(times))
        safe = np.zeros(len(times), dtype=bool)
        for note in notes[name]:
            onset, length = note['start_seconds'], note['duration_seconds']
            active = (times >= onset) & (times < onset+length)
            active_count += active
            hz[active] = note['pitch_hz']
            interior = (times >= onset+policy.boundary_guard_seconds) & (times <= onset+length-policy.boundary_guard_seconds)
            if not note['partial']:
                safe |= interior
        streams.append((hz, safe & (active_count == 1)))
    a, b = streams
    both = a[1] & b[1] & near_reference & (pitch > 0)
    agree = np.zeros(len(times), dtype=bool)
    agree[both] = abs(1200*np.log2(a[0][both]/b[0][both])) <= policy.annotator_pitch_tolerance_cents
    ref = both & agree
    ref[ref] &= abs(1200*np.log2(pitch[ref]/np.sqrt(a[0][ref]*b[0][ref]))) <= policy.reference_pitch_tolerance_cents
    return ref, pitch


REVIEW_FIELDS = ['phrase_id', 'crop_sha256', 'reviewer_id', 'musical_experience', 'decision',
                 'approved_intervals_json', 'reason', 'reviewed_at_utc']


def reviewed_intervals(phrase, review_rows):
    """Two distinct approvals required; approved support is their intersection."""
    rows = [r for r in review_rows if r['phrase_id'] == phrase['phrase_id'] and r['crop_sha256'] == phrase['sha256']]
    decided = [r for r in rows if r.get('decision') not in ('', 'pending')]
    if any(r.get('decision') != 'approve' for r in decided):
        return [], 'review_disagreement_or_rejection'
    if len(decided) != 2 or len({r.get('reviewer_id', '').strip() for r in decided}) != 2:
        return [], 'two_distinct_reviews_required'
    if any(not r.get(k, '').strip() for r in decided for k in ('reviewer_id', 'musical_experience', 'reason', 'reviewed_at_utc')):
        return [], 'review_evidence_incomplete'
    try:
        stamps = [datetime.fromisoformat(r['reviewed_at_utc'].replace('Z', '+00:00')) for r in decided]
        if any(stamp.utcoffset() != timedelta(0) for stamp in stamps):
            raise ValueError()
    except ValueError:
        return [], 'review_timestamp_invalid'
    intervals = []
    for row in decided:
        try:
            values = json.loads(row['approved_intervals_json'])
            if not isinstance(values, list) or not values:
                raise ValueError()
            for pair in values:
                if len(pair) != 2 or not all(type(v) in (float, int) and np.isfinite(v) for v in pair) or not 0 <= pair[0] < pair[1] <= phrase['duration_seconds']:
                    raise ValueError()
            intervals.append(values)
        except (ValueError, TypeError, KeyError):
            return [], 'review_intervals_invalid'
    intersection = sorted([max(a[0], b[0]), min(a[1], b[1])] for a in intervals[0] for b in intervals[1] if max(a[0], b[0]) < min(a[1], b[1]))
    return intersection, 'approved' if intersection else 'no_common_reviewed_interval'


def build_dataset(dataset_root, directory, policy=DataPolicy()):
    root, directory = Path(dataset_root).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    with (root/'vocadito_metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
        metadata = list(csv.DictReader(stream))
    ids = [r['track_id'] for r in metadata]
    if len(ids) != len(set(ids)) or any(not r['singer_id'].strip() for r in metadata):
        raise ValueError('Duplicate source IDs or missing singer metadata')
    splits = singer_splits([r['singer_id'] for r in metadata], policy.split_seed)
    sources, phrases, errors = [], [], []
    for row in sorted(metadata, key=lambda r: int(r['track_id'])):
        sid = 'vocadito_'+row['track_id']
        audio_path = root/'Audio'/(sid+'.wav')
        paths = {'audio': audio_path, 'f0': root/'Annotations'/'F0'/(sid+'_f0.csv'),
                 **{n: root/'Annotations'/'Notes'/(sid+'_notes'+n+'.csv') for n in ('A1', 'A2')}}
        try:
            audio, sr = sf.read(audio_path, dtype='float32', always_2d=True)
            if not len(audio) or not np.isfinite(audio).all():
                raise ValueError('Empty or non-finite source audio')
            if len(audio) != sf.info(audio_path).frames:
                raise ValueError('Source audio is truncated relative to its header')
            duration = len(audio)/sr
            f0 = read_array(paths['f0'], 2)
            notes = {n: read_array(paths[n], 3) for n in ('A1', 'A2')}
            issues, warnings = validate_annotations(f0, notes, duration, 1/sr)
            source = dict(source_id=sid, singer_id=row['singer_id'], split=splits[row['singer_id']],
                          language=row.get('language'), sample_rate_hz=sr, sample_count=len(audio), duration_seconds=duration,
                          paths={k: str(v) for k, v in paths.items()}, sha256={k: checksum(v) for k, v in paths.items()},
                          errors=issues, warnings=warnings)
            sources.append(source)
            if issues:
                continue
            for index, (start, end) in enumerate(crop_ranges(len(audio), sr, notes, policy.max_crop_seconds)):
                pid = f'{sid}_p{index:02d}'
                out = directory/'crops'/pid
                out.mkdir(parents=True)
                # Copy source-rate samples exactly; analysis resampling happens later.
                sf.write(out/'audio.wav', audio[start:end], sr, subtype='FLOAT')
                start_seconds, end_seconds = start/sr, end/sr
                selected = f0[(f0[:, 0] >= start_seconds) & (f0[:, 0] < end_seconds)].copy()
                selected[:, 0] -= start_seconds
                np.savetxt(out/'f0.csv', selected, delimiter=',')
                cropped_notes = {name: crop_notes(values, start_seconds, end_seconds) for name, values in notes.items()}
                dump(out/'notes.json', cropped_notes)
                phrase = dict(phrase_id=pid, source_id=sid, singer_id=row['singer_id'], split=source['split'],
                              start_sample=start, end_sample=end, original_sample_rate_hz=sr,
                              original_offset_seconds=start_seconds, duration_seconds=(end-start)/sr,
                              audio_path=str(out/'audio.wav'), f0_path=str(out/'f0.csv'), notes_path=str(out/'notes.json'),
                              sha256=checksum(out/'audio.wav'), review_status='pending',
                              reference_sha256={'f0': checksum(out/'f0.csv'), 'notes': checksum(out/'notes.json')},
                              partial_note_count=sum(n['partial'] for vals in cropped_notes.values() for n in vals))
                phrases.append(phrase)
        except (OSError, ValueError) as exc:
            errors.append(dict(source_id=sid, error=str(exc)))
    known = {s+'.wav' for s in ('vocadito_'+i for i in ids)}
    orphans = [p.name for p in (root/'Audio').glob('*.wav') if not p.name.startswith('._') and p.name not in known]
    with (directory/'reviews.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for p in phrases:
            for _ in range(2):
                writer.writerow(dict(phrase_id=p['phrase_id'], crop_sha256=p['sha256'], decision='pending', approved_intervals_json='[]'))
    manifest = dict(schema_version='phase2-0.1', policy=asdict(policy), dataset_root=str(root),
                    metadata_sha256=checksum(root/'vocadito_metadata.csv'),
                    sources=sources, phrases=phrases, singer_splits=splits, source_errors=errors, orphan_audio=orphans,
                    permissions_status='Dataset license/redistribution terms not yet recorded; raw/generated audio stays untracked',
                    review_status='pending actual human review', research_status='not training-ready')
    dump(directory/'manifest.json', manifest)
    summary = dict(source_count=len(sources), singer_count=len(splits), crop_count=len(phrases),
                   fatal_source_errors=errors, invalid_annotation_sources=[s['source_id'] for s in sources if s['errors']],
                   split_singers={name: sum(v == name for v in splits.values()) for name in set(splits.values())},
                   split_sources={name: sum(s['split'] == name for s in sources) for name in set(splits.values())},
                   approved_phrase_count=0, training_ready_examples=0)
    dump(directory/'data_summary.json', summary)
    return manifest


def verify_manifest(manifest):
    root = Path(manifest['dataset_root'])
    if checksum(root/'vocadito_metadata.csv') != manifest['metadata_sha256']:
        raise ValueError('Source metadata changed after partitioning')
    source_audio = {}
    for source in manifest['sources']:
        if source['split'] != manifest['singer_splits'][source['singer_id']]:
            raise ValueError('Singer partition leakage')
        for name, path in source['paths'].items():
            if checksum(path) != source['sha256'][name]:
                raise ValueError('Source/reference changed after validation')
        if not source['errors']:
            source_audio[source['source_id']] = sf.read(source['paths']['audio'], dtype='float32', always_2d=True)
    for phrase in manifest['phrases']:
        if phrase['split'] != manifest['singer_splits'][phrase['singer_id']]:
            raise ValueError('Crop partition leakage')
        if checksum(phrase['audio_path']) != phrase['sha256']:
            raise ValueError('Cropped audio changed after review-sheet creation')
        for name in ('f0', 'notes'):
            if checksum(phrase[name+'_path']) != phrase['reference_sha256'][name]:
                raise ValueError('Cropped references changed after validation')
        crop, rate = sf.read(phrase['audio_path'], dtype='float32', always_2d=True)
        source, original_rate = source_audio[phrase['source_id']]
        if rate != original_rate or not np.array_equal(crop, source[phrase['start_sample']:phrase['end_sample']]):
            raise ValueError('Cropped samples do not match their mapped original interval')
    for source in manifest['sources']:
        if source['errors']:
            continue
        crops = sorted((p for p in manifest['phrases'] if p['source_id'] == source['source_id']), key=lambda p: p['start_sample'])
        position = 0
        for crop in crops:
            if crop['start_sample'] != position or crop['end_sample'] <= position:
                raise ValueError('Crop coverage contains a gap or overlap')
            position = crop['end_sample']
        if position != source['sample_count']:
            raise ValueError('Crop coverage does not span its original source')
    return True
