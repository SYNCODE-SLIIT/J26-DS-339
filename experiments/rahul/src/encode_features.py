"""Schema-checked, train-only encoders for Phase 1 chord events."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from experiments.rahul.src.build_synthetic_dataset import context, read_csv
from experiments.rahul.src.canonicalize_chords import canonicalize_chord
from experiments.rahul.src.normalize_chords import parse_chord_label

SPECIALS = ['PAD', 'UNK', 'N', 'START', 'END', 'unknown']
IGNORE_INDEX = -100
CATEGORICAL = {
    'basic_chord_ids': 'basic_chord_canonical', 'basic_root_ids': 'basic_root',
    'basic_quality_ids': 'basic_quality', 'previous_basic_chord_ids': 'previous_basic_chord',
    'next_basic_chord_ids': 'next_basic_chord', 'key_ids': 'key', 'mode_ids': 'mode',
    'key_confidence_ids': 'key_confidence', 'roman_ids': 'roman_numeral',
    'harmonic_function_ids': 'harmonic_function',
}
CHORD_FIELDS = {'basic_chord_canonical', 'previous_basic_chord', 'next_basic_chord'}
LIST_FIELDS = ['melody_midi', 'melody_pitch_classes', 'melody_durations',
               'melody_relative_start', 'melody_overlap_durations', 'melody_sources', 'genre_tags']
USED_INPUTS = set(CATEGORICAL.values()) | set(LIST_FIELDS) | {
    'beat_index', 'bar_number', 'beat_position', 'event_duration'}
FORBIDDEN_INPUTS = {'melody', 'family', 'basic_label', 'source_roman', 'target_valid',
                    'reference_chord', 'reference_chord_canonical', 'reference_quality',
                    'reference_bass_degree', 'reference_root_pitch_class', 'reference_overlap_ratio',
                    'reference_candidates', 'source_chord_event_id', 'reference_ambiguous'}


def parse_list(value, field):
    """Only JSON lists are accepted; no eval, implicit empty, or scalar coercion."""
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{field}: invalid JSON list') from exc
    if not isinstance(parsed, list):
        raise ValueError(f'{field}: expected JSON list')
    return parsed


def finite(value, field):
    if isinstance(value, bool) or value in ('', None):
        raise ValueError(f'{field}: missing or nonnumeric value')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field}: nonnumeric value') from exc
    if not math.isfinite(result):
        raise ValueError(f'{field}: non-finite value')
    return result


def boolean(value):
    if value is True or value == 'True':
        return True
    if value is False or value == 'False':
        return False
    raise ValueError(f'Invalid boolean: {value!r}')


def chord_token(value):
    if value in SPECIALS or value in (None, ''):
        return value or 'UNK'
    try:
        return canonicalize_chord(value)[0]
    except ValueError:
        return 'UNK'


def supported_target(row):
    raw = row.get('reference_chord')
    label = row.get('reference_chord_canonical')
    if not raw or not label:
        return False
    parsed = parse_chord_label(raw)
    try:
        return parsed.parsed and parsed.normalized and canonicalize_chord(raw)[0] == label
    except ValueError:
        return False


def verify_schema(schema):
    inputs = set(schema['input_columns'])
    targets = set(schema['target_columns'])
    if inputs & targets or inputs & FORBIDDEN_INPUTS or any(c.startswith('reference_') for c in inputs):
        raise ValueError('Schema leaks target/audit columns into model inputs')
    if not USED_INPUTS <= inputs or 'reference_chord_canonical' not in targets:
        raise ValueError('Schema does not declare the required input/target features')
    for name in inputs:
        if schema['columns'][name]['role'] != 'input':
            raise ValueError(f'Schema input role mismatch: {name}')


def load_events(csv_path, schema_path=None):
    """Validate, then sort a complete track. Malformed numerical inputs fail loudly."""
    csv_path = Path(csv_path)
    schema = json.loads(Path(schema_path or csv_path.parent/'dataset_schema.json').read_text(encoding='utf-8'))
    verify_schema(schema)
    rows = read_csv(csv_path)
    if not rows:
        raise ValueError(f'Empty dataset: {csv_path}')
    for row in rows:
        if set(row) != set(schema['columns']):
            raise ValueError('CSV columns do not match schema')
        for field in LIST_FIELDS:
            row[field] = parse_list(row[field], field)
        lengths = [len(row[f]) for f in LIST_FIELDS if f != 'genre_tags']
        if len(set(lengths)) != 1:
            raise ValueError('Melody array lengths differ')
        for field in ('beat_index', 'start_sec', 'end_sec', 'event_duration'):
            row[field] = finite(row[field], field)
        if row['beat_index'] < 0 or not row['beat_index'].is_integer():
            raise ValueError('beat_index must be a nonnegative integer')
        row['beat_index'] = int(row['beat_index'])
        if row['event_duration'] <= 0 or row['start_sec'] < 0 or not math.isclose(
                row['end_sec']-row['start_sec'], row['event_duration'], abs_tol=1e-8):
            raise ValueError('Invalid event interval/duration')
        for field in ('bar_number', 'beat_position'):
            value = None if row[field] in ('', None) else finite(row[field], field)
            if value is not None and (not value.is_integer() or value < (1 if field == 'beat_position' else 0)):
                raise ValueError(f'Invalid {field}')
            row[field] = int(value) if value is not None else None
        for field in ('melody_midi', 'melody_pitch_classes', 'melody_durations',
                      'melody_relative_start', 'melody_overlap_durations'):
            row[field] = [finite(v, field) for v in row[field]]
        if any(not 0 <= v <= 127 for v in row['melody_midi']):
            raise ValueError('MIDI pitch outside [0,127]; no clipping/transposition is performed')
        if any(not v.is_integer() or not 0 <= v < 12 for v in row['melody_pitch_classes']):
            raise ValueError('Pitch classes must be integers in [0,11]')
        if any(v <= 0 for v in row['melody_durations']) or any(
                not 0 < v <= row['event_duration']+1e-8 for v in row['melody_overlap_durations']):
            raise ValueError('Invalid note duration/overlap')
        if any(o > d+1e-8 for o,d in zip(row['melody_overlap_durations'],row['melody_durations'])):
            raise ValueError('Note overlap exceeds original duration')
        if any(not isinstance(v,str) or not v for f in ('melody_sources','genre_tags') for v in row[f]):
            raise ValueError('Source/genre lists must contain nonempty strings')
        row['target_valid'] = boolean(row['target_valid'])
        basic = {f: row[f] or None for f in ('basic_chord','basic_root','basic_quality')}
        if context(basic,row['key'],row['mode']) != (row['roman_numeral'],row['harmonic_function']):
            raise ValueError('Basic/key context differs from Phase 1 rules')
        if chord_token(row['basic_chord']) != (row['basic_chord_canonical'] or 'UNK'):
            raise ValueError('Basic chord canonicalization mismatch')
    if len({r['track_id'] for r in rows}) != 1:
        raise ValueError('Each source file must contain exactly one track')
    rows.sort(key=lambda r:r['beat_index'])
    if len({r['beat_index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate beat IDs')
    for i,row in enumerate(rows):
        if i and (row['beat_index'] != rows[i-1]['beat_index']+1 or row['start_sec'] < rows[i-1]['end_sec']-1e-8):
            raise ValueError('Nonconsecutive beats or nonchronological intervals')
        previous = rows[i-1]['basic_chord'] if i else 'START'
        following = rows[i+1]['basic_chord'] if i+1<len(rows) else 'END'
        if row['previous_basic_chord'] != previous or row['next_basic_chord'] != following:
            raise ValueError('Basic chord context does not match track order')
        row['input_snapshot'] = {f:row[f] for f in sorted(USED_INPUTS)}
    return rows,schema


class Vocabulary:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        if len(set(self.tokens)) != len(self.tokens) or self.tokens[:2] != ['PAD','UNK']:
            raise ValueError('Invalid vocabulary')
        self.token_to_id = {token:i for i,token in enumerate(self.tokens)}

    @classmethod
    def fit(cls, values, specials=SPECIALS):
        return cls(list(specials)+sorted({v for v in values if v and v not in specials}))

    def encode(self, token):
        return self.token_to_id.get(token,1)

    def decode(self, index):
        return self.tokens[index]

    def state(self):
        return {'token_to_id':self.token_to_id, 'id_to_token':self.tokens}

    @classmethod
    def restore(cls, state):
        vocab = cls(state['id_to_token'])
        if vocab.token_to_id != state['token_to_id']:
            raise ValueError('Inconsistent saved vocabulary mapping')
        return vocab


class FeatureEncoder:
    """Fit only with explicitly supplied training tracks; transform never fits."""
    def __init__(self, vocabs, targets, genres, max_notes, training_tracks, truncate=False):
        self.vocabs,self.targets,self.genres = vocabs,targets,list(genres)
        self.max_notes,self.training_tracks,self.truncate = max_notes,list(training_tracks),truncate
        if max_notes < 1:
            raise ValueError('max_notes must be >= 1, including all-silent training sets')

    @staticmethod
    def category(row, field):
        value = row.get(field)
        if field in CHORD_FIELDS:
            return chord_token(value)
        if field == 'basic_root' and row.get('basic_chord_canonical') == 'N':
            return 'N'
        return value or 'UNK'

    @classmethod
    def fit(cls, tracks, training_tracks, max_notes=None, truncate=False):
        training_tracks = sorted(training_tracks)
        if not training_tracks or not set(training_tracks) <= set(tracks):
            raise ValueError('Explicit nonempty training-track set is required')
        rows = [r['input_snapshot'] for t in training_tracks for r in tracks[t]]
        chord_vocab = Vocabulary.fit(cls.category(r,f) for r in rows for f in sorted(CHORD_FIELDS))
        vocabs = {field:chord_vocab if field in CHORD_FIELDS else Vocabulary.fit(cls.category(r,field) for r in rows)
                  for field in sorted(set(CATEGORICAL.values()))}
        vocabs['melody_sources'] = Vocabulary.fit(s for r in rows for s in r['melody_sources'])
        target_vocab = Vocabulary.fit((r['reference_chord_canonical'] for t in training_tracks for r in tracks[t]
                                      if r['target_valid'] and supported_target(r)), ['PAD','UNK','N'])
        genres = ['UNK']+sorted({g for r in rows for g in r['genre_tags']} - {'UNK'})
        observed = max((len(r['melody_midi']) for r in rows),default=0)
        maximum = max(1,observed) if max_notes is None else max_notes
        if maximum < observed and not truncate:
            raise ValueError('Configured note maximum would truncate training notes; enable --truncate-notes explicitly')
        return cls(vocabs,target_vocab,genres,maximum,training_tracks,truncate)

    def encode_inputs(self, row):
        """Whitelist projection makes target-field changes irrelevant to all inputs."""
        row = {f:row[f] for f in sorted(USED_INPUTS)}
        result = {name:np.int64(self.vocabs[field].encode(self.category(row,field))) for name,field in CATEGORICAL.items()}
        m = self.max_notes
        count = len(row['melody_midi'])
        if count > m and not self.truncate:
            raise ValueError(f'{count} notes exceed training-fitted maximum {m}; opt into deterministic truncation or choose a larger training policy')
        count = min(count,m)
        result.update(melody_pitch_ids=np.zeros(m,dtype=np.int64),melody_pitch_class_ids=np.zeros(m,dtype=np.int64),
            melody_pitch_values=np.zeros(m,dtype=np.float32),melody_source_ids=np.zeros(m,dtype=np.int64),
            melody_timing=np.zeros((m,3),dtype=np.float32),melody_note_mask=np.zeros(m,dtype=np.bool_))
        result['melody_note_mask'][:count] = True
        for i in range(count):
            pitch = row['melody_midi'][i]
            # Python round is nearest integer with ties to even. Fractional value is also retained.
            result['melody_pitch_ids'][i] = round(pitch)+2
            result['melody_pitch_values'][i] = pitch/127.0
            result['melody_pitch_class_ids'][i] = int(row['melody_pitch_classes'][i])+2
            result['melody_source_ids'][i] = self.vocabs['melody_sources'].encode(row['melody_sources'][i])
            result['melody_timing'][i] = np.array([row['melody_durations'][i],row['melody_relative_start'][i],
                                                   row['melody_overlap_durations'][i]])/row['event_duration']
        safe_mask = result['melody_note_mask'].copy()
        if not safe_mask.any():
            safe_mask[0] = True
        result['melody_attention_mask'] = safe_mask
        result['melody_silent'] = np.bool_(count==0)
        genres = np.zeros(len(self.genres),dtype=np.float32)
        for genre in row['genre_tags']:
            genres[self.genres.index(genre) if genre in self.genres else 0] = 1
        result['genre_features'] = genres
        for field in ('beat_index','bar_number','beat_position','event_duration'):
            result[field] = np.float32(row[field] if row[field] is not None else 0)
        result['beat_timing_known_mask'] = np.array([row['bar_number'] is not None,row['beat_position'] is not None],dtype=np.bool_)
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError('Encoded inputs exceed finite numerical range')
        return result

    def encode_target(self, row):
        label = row.get('reference_chord_canonical')
        known = label in self.targets.token_to_id and label not in ('PAD','UNK')
        valid = bool(row['target_valid'] and supported_target(row) and known)
        return {'target_chord_ids':np.int64(self.targets.encode(label) if valid else IGNORE_INDEX),
                'target_loss_mask':np.bool_(valid)}

    def save(self, directory):
        directory = Path(directory)
        features = {'schema_version':1,'vocabs':{k:v.state() for k,v in self.vocabs.items()},'genres':self.genres,
                    'max_notes_per_beat':self.max_notes,'training_tracks':self.training_tracks,'truncate_notes':self.truncate}
        (directory/'feature_vocabularies.json').write_text(json.dumps(features,indent=2)+'\n',encoding='utf-8')
        (directory/'target_vocabulary.json').write_text(json.dumps(self.targets.state(),indent=2)+'\n',encoding='utf-8')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        state=json.loads((directory/'feature_vocabularies.json').read_text(encoding='utf-8'))
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported encoder schema version')
        target=json.loads((directory/'target_vocabulary.json').read_text(encoding='utf-8'))
        return cls({k:Vocabulary.restore(v) for k,v in state['vocabs'].items()},Vocabulary.restore(target),
                   state['genres'],state['max_notes_per_beat'],state['training_tracks'],state['truncate_notes'])

    def statistics(self, rows):
        sizes = [len(r['melody_midi']) for r in rows]
        return dict(beats=len(rows),source_valid_targets=sum(r['target_valid'] for r in rows),
            loss_eligible_targets=sum(bool(self.encode_target(r)['target_loss_mask']) for r in rows),
            silent_beats=sizes.count(0),maximum_notes_per_beat=max(sizes,default=0),
            mean_notes_per_beat=float(np.mean(sizes)) if sizes else 0,median_notes_per_beat=float(np.median(sizes)) if sizes else 0,
            configured_max_notes=self.max_notes,beats_exceeding_maximum=sum(n>self.max_notes for n in sizes),
            notes_truncated=sum(max(0,n-self.max_notes) for n in sizes) if self.truncate else 0,
            unseen_targets=dict(Counter(r['reference_chord_canonical'] for r in rows
                                       if r['reference_chord_canonical'] and r['reference_chord_canonical'] not in self.targets.token_to_id)),
            unsupported_or_missing_targets=sum(not supported_target(r) for r in rows),
            unseen_genres=dict(Counter(g for r in rows for g in r['genre_tags'] if g not in self.genres)),
            unknown_categorical_values={f:dict(Counter(self.category(r,f) for r in rows if self.category(r,f) not in v.token_to_id)) for f,v in self.vocabs.items() if f != 'melody_sources'})
