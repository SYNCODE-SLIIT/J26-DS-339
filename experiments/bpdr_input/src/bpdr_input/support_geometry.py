"""Read-only candidate geometry; never renders audio or authorizes paired targets."""
import numpy as np
from .interventions import AuditPolicy


PROTOCOL = {
    'version': 'multi-interior-geometry-0.1',
    'partitions': ['train', 'development'],
    'endpoint_grid': 'existing frame timestamps, at least analysis_guard from crop edges',
    'plateau': 'unchanged reliable mask AND start+ramp+guard <= time <= end-ramp-guard',
    'outside': 'unchanged reliable mask AND (time < start-guard OR time > end+guard)',
    'minimum_distinct_annotator_note_pairs_in_plateau': 2,
    'minimum_physical_plateau_seconds': 'unchanged AuditPolicy.min_plateau_seconds',
    'minimum_plateau_and_outside_frames': 'unchanged AuditPolicy minimum audit frame counts',
    'selection': 'shortest duration rounded to 12 decimals; largest min(plateau,outside) count; earliest start; earliest end',
    'coverage': 'no post-intervention coverage or acoustic success inferred from geometry',
    'training_grid_without_cached_acoustics': 'annotation-only 20 ms grid used by original laboratory ranking',
    'training_eligible': False,
}


def note_pair_ids(times, notes, guard):
    """Distinct human annotator note identities, not tracker-fragment identities."""
    streams = []
    for name in ('A1', 'A2'):
        ids = np.zeros(len(times), dtype=np.int64)
        counts = np.zeros(len(times), dtype=np.int64)
        for k, note in enumerate(notes[name], 1):
            a = note['start_seconds']; b = a + note['duration_seconds']
            counts += (times >= a) & (times < b)
            if not note['partial']:
                ids[(times >= a+guard) & (times <= b-guard)] = k
        ids[counts != 1] = 0
        streams.append(ids)
    a, b = streams
    return np.where((a > 0) & (b > 0), a*(len(notes['A2'])+1)+b, 0)


def geometry_masks(times, reliable, support, policy=AuditPolicy()):
    a, b = support
    plateau = reliable & (times >= a+policy.ramp_seconds+policy.analysis_guard_seconds) & (times <= b-policy.ramp_seconds-policy.analysis_guard_seconds)
    outside = reliable & ((times < a-policy.analysis_guard_seconds) | (times > b+policy.analysis_guard_seconds))
    return plateau, outside


def select_multi_interior(times, reliable, note_ids, duration, split, policy=AuditPolicy()):
    """Enumerate declared grid; return necessary geometry only, never audit success."""
    if split not in PROTOCOL['partitions']:
        raise ValueError('Reserved partition; geometry access refused')
    times = np.asarray(times)
    reliable = np.asarray(reliable)
    note_ids = np.asarray(note_ids)
    if (times.ndim != 1 or not len(times) or reliable.dtype != bool or reliable.shape != times.shape
            or note_ids.shape != times.shape or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not np.isfinite(duration) or duration <= 0
            or times[0] < 0 or times[-1] > duration or np.any(reliable & (note_ids <= 0))):
        raise ValueError('Invalid geometry inputs or reliable frame outside trusted notes')
    grid = times[(times >= policy.analysis_guard_seconds) & (times <= duration-policy.analysis_guard_seconds)]
    minimum = policy.min_plateau_seconds+2*(policy.ramp_seconds+policy.analysis_guard_seconds)
    # Prefix counts search necessary frame counts without inspecting acoustic outcomes.
    prefix = np.r_[0, np.cumsum(reliable)]
    best = None
    for a in grid:
        ends = grid[grid-a >= minimum]
        if not len(ends):
            continue
        lo = np.searchsorted(times, a+policy.ramp_seconds+policy.analysis_guard_seconds, side='left')
        hi = np.searchsorted(times, ends-policy.ramp_seconds-policy.analysis_guard_seconds, side='right')
        count = prefix[hi]-prefix[lo]
        left = prefix[np.searchsorted(times, a-policy.analysis_guard_seconds, side='left')]
        right = prefix[-1]-prefix[np.searchsorted(times, ends+policy.analysis_guard_seconds, side='right')]
        outside = left+right
        candidates = np.flatnonzero((count >= policy.min_audit_frames) & (outside >= policy.min_outside_audit_frames))
        for j in candidates:
            b = ends[j]
            distinct = len(np.unique(note_ids[lo:hi[j]][reliable[lo:hi[j]]]))
            if distinct < PROTOCOL['minimum_distinct_annotator_note_pairs_in_plateau']:
                continue
            score = (round(float(b-a), 12), -int(min(count[j], outside[j])), float(a), float(b))
            if best is None or score < best[0]:
                best = (score, dict(support_seconds=[float(a), float(b)],
                    physical_plateau_seconds=float(b-a-2*(policy.ramp_seconds+policy.analysis_guard_seconds)),
                    plateau_reference_frames=int(count[j]), outside_reference_frames=int(outside[j]),
                    distinct_annotator_note_pairs=int(distinct)))
            # Earliest feasible end is shortest for this start; longer ends cannot win.
            break
    return None if best is None else best[1]
