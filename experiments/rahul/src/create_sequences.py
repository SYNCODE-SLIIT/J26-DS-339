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
