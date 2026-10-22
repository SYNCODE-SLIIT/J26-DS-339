"""Deterministic track-group splits and chronological windows; no beat-level splitting."""
from __future__ import annotations

import math
import random


def split_tracks(track_ids, ratios=(.8,.1,.1), seed=42, identities=None, development_only=False):
    ids = sorted(track_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Nonempty unique track IDs required')
    if len(ratios) != 3 or any(not math.isfinite(v) or v < 0 for v in ratios) or not math.isclose(sum(ratios),1):
        raise ValueError('Train/validation/test ratios must be nonnegative and sum to one')
    if ratios[0] <= 0:
        raise ValueError('Training ratio must be positive')
    parent = {t:t for t in ids}

    def root(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    seen = {}
    for track in ids:
        for identity in (identities or {}).get(track,[]):
            if identity in seen:
                parent[root(track)] = root(seen[identity])
            else:
                seen[identity] = track
    groups = {}
    for t in ids:
        groups.setdefault(root(t),[]).append(t)
    grouped = sorted(groups.values())
    if development_only:
        assignment = {'train':ids,'validation':[],'test':[]}
    else:
        active = [i for i,v in enumerate(ratios) if v > 0]
        if len(ids) == 1 or len(grouped) < len(active):
            raise ValueError('Too few independent recording/artist groups for requested splits; use --development-only for smoke testing')
        random.Random(seed).shuffle(grouped)
        # Allocate whole groups; preserve at least one group per requested subset.
        raw = [len(grouped)*r for r in ratios]
        counts = [math.floor(v) for v in raw]
        for i in sorted(range(3),key=lambda i: (-(raw[i]-counts[i]),i))[:len(grouped)-sum(counts)]:
            counts[i] += 1
        for i in active:
            if counts[i] == 0:
                donor = max(active,key=lambda j:counts[j])
                counts[donor] -= 1
                counts[i] += 1
        assignment = {}
        offset = 0
        for name,count in zip(('train','validation','test'),counts):
            assignment[name] = sorted(t for group in grouped[offset:offset+count] for t in group)
            offset += count
    manifest = dict(seed=seed,development_only=development_only,requested_ratios=list(ratios),splits=assignment,
                    identity_groups=sorted(groups.values()),identities=identities or {},
                    policy='Shuffle recording/artist connected groups with local seeded RNG; round group counts and preserve nonempty requested subsets. Actual track ratios may differ.')
    validate_splits(manifest,ids)
    return manifest


def validate_splits(manifest, track_ids):
    splits = manifest['splits']
    if set(splits) != {'train','validation','test'} or not splits['train']:
        raise ValueError('Invalid split names or empty training split')
    assigned = [t for tracks in splits.values() for t in tracks]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(track_ids):
        raise ValueError('Split overlap or incomplete coverage')
    owner = {t:s for s,tracks in splits.items() for t in tracks}
    for group in manifest.get('identity_groups',[]):
        if len({owner[t] for t in group}) != 1:
            raise ValueError('Related recording identities cross splits')
    if len(track_ids)==1 and not manifest.get('development_only'):
        raise ValueError('Single-track data requires development-only mode')
    if manifest.get('development_only') and (splits['validation'] or splits['test']):
        raise ValueError('Development-only mode must not contain validation/test partitions')


def sequence_windows(tracks, sequence_length=8, stride=None):
    stride = sequence_length if stride is None else stride
    if sequence_length < 1 or stride < 1 or stride > sequence_length:
        raise ValueError('Require 1 <= stride <= sequence_length to preserve all beats')
    windows = []
    for track in sorted(tracks):
        rows = tracks[track]
        if not rows:
            raise ValueError(f'Empty track: {track}')
        if any(r['track_id'] != track for r in rows) or any(a['beat_index']>=b['beat_index'] for a,b in zip(rows,rows[1:])):
            raise ValueError('Track mismatch or nonchronological beat ordering')
        for start in range(0,len(rows),stride):
            windows.append(dict(track_id=track,start_row=start,stop_row=min(start+sequence_length,len(rows)),
                                real_beats=min(sequence_length,len(rows)-start)))
    return windows


def bar_windows(tracks, sequence_length):
    """One contiguous source bar per window; never guess boundaries from count."""
    if sequence_length < 1:
        raise ValueError('Positive padded bar capacity required')
    windows = []
    for track in sorted(tracks):
        rows = tracks[track]
        if not rows or any(r['track_id'] != track for r in rows):
            raise ValueError('Empty track or track mismatch')
        if any(a['beat_index'] >= b['beat_index'] for a,b in zip(rows,rows[1:])):
            raise ValueError('Nonchronological beat ordering')
        seen, start = set(), 0
        while start < len(rows):
            bar = rows[start].get('bar_number')
            if bar is None or bar in seen:
                raise ValueError('Missing or noncontiguous source bar metadata')
            seen.add(bar)
            stop = start+1
            while stop < len(rows) and rows[stop].get('bar_number') == bar:
                stop += 1
            positions = [r.get('beat_position') for r in rows[start:stop]]
            if any(p is None or p < 1 for p in positions) or any(a >= b for a,b in zip(positions,positions[1:])):
                raise ValueError('Missing, duplicate or unordered positions within a source bar')
            if stop-start > sequence_length:
                raise ValueError('Source bar exceeds padded capacity; never truncate beats')
            windows.append(dict(track_id=track,start_row=start,stop_row=stop,
                                real_beats=stop-start,bar_number=bar))
            start = stop
    return windows


def multi_bar_windows(tracks, bars_per_sequence=4, bar_stride=None):
    """Non-overlapping windows of consecutive source bars, with real beat lengths.

    Missing/invalid metadata fails. Gaps from previously excluded bars break runs.
    Partial boundary bars are retained; an internal partial bar is isolated rather
    than silently treated as a complete bar. Meter never determines token count.
    """
    if not isinstance(bars_per_sequence,int) or bars_per_sequence<1:
        raise ValueError('bars_per_sequence must be a positive integer')
    if bar_stride not in (None,bars_per_sequence):
        raise ValueError('Only non-overlapping bar stride is currently supported')
    windows=[]
    for track in sorted(tracks):
        rows=tracks[track]
        bars=bar_windows({track:rows},max(1,len(rows)))
        for row in rows:
            if 'start_sec' in row and (not all(math.isfinite(float(row[k])) for k in ('start_sec','end_sec')) or row['end_sec']<=row['start_sec']):
                raise ValueError('Invalid beat timestamps')
        for a,b in zip(rows,rows[1:]):
            if 'start_sec' in a and 'start_sec' in b:
                if not all(math.isfinite(float(r[k])) for r in (a,b) for k in ('start_sec','end_sec')):
                    raise ValueError('Nonfinite beat timestamps')
                if a['end_sec']<=a['start_sec'] or b['end_sec']<=b['start_sec'] or b['start_sec']<a['end_sec']-1e-8:
                    raise ValueError('Invalid or overlapping beat timestamps')
        for i,bar in enumerate(bars):
            if i and bar['bar_number']<=bars[i-1]['bar_number']:
                raise ValueError('Source bar numbers are not increasing')
            events=rows[bar['start_row']:bar['stop_row']]
            positions=[r['beat_position'] for r in events]
            ordered_full_prefix=positions==list(range(1,len(events)+1))
            # End of the next source downbeat confirms a bar without consulting
            # the numerator (compound-meter event counts can differ from it).
            next_boundary=(i+1<len(bars) and bars[i+1]['bar_number']==bar['bar_number']+1
                and rows[bars[i+1]['start_row']]['beat_position']==1
                and rows[bars[i+1]['start_row']]['beat_index']==events[-1]['beat_index']+1)
            bar['partial']=not ordered_full_prefix or (i==len(bars)-1 and not events[-1].get('bar_end_confirmed',False))
            bar['complete']=ordered_full_prefix and (next_boundary or events[-1].get('bar_end_confirmed',False))
            bar['irregular']=not ordered_full_prefix or any(b['beat_index']!=a['beat_index']+1 for a,b in zip(events,events[1:])) or any(r.get('source_meter_disagreement',False) for r in events)
        pending=[]

        def emit(reason):
            if not pending:return
            first,last=pending[0],pending[-1]
            windows.append(dict(track_id=track,start_row=first['start_row'],stop_row=last['stop_row'],
                real_beats=last['stop_row']-first['start_row'],sequence_start_bar=first['bar_number'],
                sequence_end_bar=last['bar_number'],bars_in_sequence=len(pending),
                contains_partial_bar=any(b['partial'] for b in pending),
                contains_irregular_bar=any(b['irregular'] for b in pending),
                partial_bars_count=sum(b['partial'] for b in pending),irregular_bars_count=sum(b['irregular'] for b in pending),
                complete_bar_count=sum(b['complete'] for b in pending),
                complete_four_bar_window=len(pending)==bars_per_sequence and all(b['complete'] for b in pending),
                final_partial_window=last['stop_row']==len(rows) and len(pending)<bars_per_sequence,
                window_reason=reason))
            pending.clear()

        for i,bar in enumerate(bars):
            if i:
                prev=bars[i-1]
                previous_event=rows[prev['stop_row']-1]
                next_event=rows[bar['start_row']]
                gap=bar['bar_number']!=prev['bar_number']+1 or next_event['beat_index']!=previous_event['beat_index']+1
                if 'start_sec' in next_event and 'end_sec' in previous_event:
                    gap=gap or not math.isclose(next_event['start_sec'],previous_event['end_sec'],abs_tol=1e-8)
                if gap:emit('source_gap')
            if bar['partial'] and 0<i<len(bars)-1:
                emit('before_internal_partial_bar')
                pending.append(bar)
                emit('internal_partial_bar_retained')
                continue
            pending.append(bar)
            if len(pending)==bars_per_sequence:emit('full_bar_window')
        emit('track_end')
    return windows
