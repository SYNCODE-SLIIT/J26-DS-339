"""Saved aligned features supply beats, key and context outside the live crop. No chord targets read."""
from functools import lru_cache
from pathlib import Path
import csv
import json
import math

import numpy as np
import soundfile as sf

from pp1_demo.config import DATASET, EXTRACTIONS, VOCAB, PC_NAMES
from pp1_demo.contracts import sha


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def genres_by_track():
    result = {}
    with (DATASET / 'annotations/autotagging_genre.tsv').open() as stream:
        for row in csv.reader(stream, delimiter='\t'):
            if row and row[0].startswith('track_'):
                result[row[0]] = [x.removeprefix('genre---') for x in row[5:] if x.startswith('genre---')]
    return result


def complete_context(beats, crop_start, crop_end):
    """Find sixteen complete contiguous bars covering (or overlapping) the crop."""
    starts = [i for i, b in enumerate(beats) if b['beat_in_bar'] == 1]
    candidates = []
    for k in range(len(starts) - 16):
        left, right = starts[k], starts[k + 16]
        selected = beats[left:right]
        a, z = beats[left]['time_sec'], beats[right]['time_sec']
        if a > crop_start + 1e-6 or z < crop_end - 1e-6:
            continue
        valid = True
        for j in range(16):
            bar = beats[starts[k+j]:starts[k+j+1]]
            meter = int(bar[0]['meter_numerator'])
            if not 1 <= meter <= 9 or [b['beat_in_bar'] for b in bar] != list(range(1, meter+1)) or any(b['meter_numerator'] != meter for b in bar):
                valid = False
        if valid:
            candidates.append((abs((a+z)/2-(crop_start+crop_end)/2), left, right))
    if not candidates:
        raise ValueError('This vocal crop has no covering sixteen-bar context with valid beats.')
    _, left, right = min(candidates)
    return left, right


@lru_cache(maxsize=24)
def context(track_id):
    if not track_id.startswith('track_') or not track_id[6:].isdigit():
        raise ValueError('Invalid recording ID')
    folder = EXTRACTIONS / track_id
    record = read(folder / 'bpdr/record.json')
    audio = folder / 'bpdr/vocal.wav'
    info = sf.info(audio)
    offset = float(record['crop']['offset_seconds'])
    duration = float(info.duration)
    beat_data = read(folder / 'beats/beats.json')
    source_beats = beat_data['beats']
    left, right = complete_context(source_beats, offset, offset+duration)
    window_start = float(source_beats[left]['time_sec'])
    window_end = float(source_beats[right]['time_sec'])
    first_bar = source_beats[left]['bar_index']
    beats = []
    for i in range(left, right):
        b = source_beats[i]
        beats.append({'start_s':float(b['time_sec'])-window_start,
                      'end_s':float(source_beats[i+1]['time_sec'])-window_start,
                      'beat_index':i-left, 'bar_number':int(b['bar_index'])-first_bar,
                      'beat_position':int(b['beat_in_bar']), 'meter':int(b['meter_numerator'])})
    melody_path = folder / 'melody/combined.json'
    melody = read(melody_path)
    notes = []
    for n in melody['notes']:
        a, b = float(n['onset_sec']), float(n['offset_sec'])
        if a >= window_end or b <= window_start:
            continue
        pitch = int(round(n.get('midi_pitch_rounded', n['midi_pitch'])))
        if not 0 <= pitch <= 127 or not math.isfinite(a+b) or b <= a:
            continue
        notes.append({'onset_s':max(a,window_start)-window_start,
                      'duration_s':min(b,window_end)-max(a,window_start),
                      'pitch_midi':pitch, 'source':n.get('source','saved_extraction')})
    notes.sort(key=lambda n:n['onset_s'])
    if not notes:
        raise ValueError('No saved melody notes in context')
    key = read(folder / 'key.json')['melody_key']['selected']
    if key is None:
        raise ValueError('No saved key estimate')
    reference_pc = (int(key['tonic_pc']) + (3 if key['mode']=='minor' else 0)) % 12
    source_genres = genres_by_track().get(track_id, [])
    allowed = read(VOCAB / 'feature_vocabularies.json')['genres']
    matching = [g for g in source_genres if g in allowed]
    genre = next((g for g in ['pop','rock','electronic','folk','jazz'] if g in matching), next(iter(matching),'pop'))
    visible_start = offset-window_start
    display_notes = []
    for n in notes:
        a, b = n['onset_s']-visible_start, n['onset_s']+n['duration_s']-visible_start
        if a < duration and b > 0:
            display_notes.append(dict(n,onset_s=max(0,a),duration_s=min(duration,b)-max(0,a)))
    display_beats = crop_rows(beats, visible_start, duration)
    return {'track_id':track_id,'audio_path':str(audio),'duration_s':duration,
            'crop_offset_s':offset,'context_start_s':window_start,'context_end_s':window_end,
            'visible_start_s':visible_start,'notes':notes,'beats':beats,
            'display_notes':display_notes,'display_beats':display_beats,
            'key':key,'reference_pc':reference_pc,'reference_key':PC_NAMES[reference_pc]+' major',
            'genre':genre,'genres':source_genres,'genre_source':'dataset_metadata' if matching else 'demo_default',
            'bpm':float(beat_data['median_beat_rate_bpm']),
            'files':{name:str(folder / name) for name in ['bpdr/record.json','melody/combined.json','beats/beats.json','key.json']},
            'audio_sha256':record['artifacts']['vocal.wav'],
            'metadata_source':'Saved GAME / SheetSage melody, saved beat grid and melody-key estimate; Pre-run preview only; live melody replaces the crop during inference.'}


def crop_rows(rows, start, duration):
    out = []
    for row in rows:
        a, b = row['start_s']-start, row['end_s']-start
        if a < duration and b > 0:
            out.append(dict(row,start_s=max(0.,a),end_s=min(duration,b)))
    return out


@lru_cache(maxsize=1)
def catalogue():
    entries = []
    for folder in sorted(EXTRACTIONS.glob('track_*')):
        if not (folder / 'bpdr/vocal.wav').is_file():
            continue
        try:
            c = context(folder.name)
            if not .25 <= c['duration_s'] <= 20:
                continue
            entries.append(public_context(c, preview=False))
        except (OSError, KeyError, ValueError, TypeError):
            continue
    if not entries:
        raise ValueError('No aligned local vocal examples are available.')
    return entries


@lru_cache(maxsize=24)
def track_preview(track_id):
    return public_context(context(track_id))


def public_context(c, preview=True):
    result = {'id':c['track_id'],'title':'Jamendo '+str(int(c['track_id'][6:])),
            'duration_s':c['duration_s'],'crop_offset_s':c['crop_offset_s'],
            'key':c['key'],'reference_key':c['reference_key'],'genre':c['genre'],'genres':c['genres'],
            'genre_source':c['genre_source'],'bpm':round(c['bpm'],1),
            'context_bars':16,'context_duration_s':c['context_end_s']-c['context_start_s'],
            'context_notes':len(c['notes']),'context_beats':len(c['beats']),
            'audio_url':f"/api/audio/{c['track_id']}",
            'source':'Local vocal dataset','metadata_source':c['metadata_source']}
    if preview:
        audio, sr = sf.read(c['audio_path'],dtype='float32')
        # Decode waveforms only for the selected recording, not the whole library.
        buckets = np.array_split(audio, min(420,len(audio)))
        wave = [round(float(np.max(np.abs(x))),5) for x in buckets]
        result.update(notes=c['display_notes'], beats=c['display_beats'], waveform=wave)
    return result


def verify_audio(c):
    if sha(c['audio_path']) != c['audio_sha256']:
        raise ValueError('Dataset audio differs from its saved extraction record.')
