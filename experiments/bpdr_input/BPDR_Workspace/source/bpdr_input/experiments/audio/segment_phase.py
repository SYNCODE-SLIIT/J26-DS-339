"""Declared local phase diagnostic on frozen support; never an audio renderer."""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from bpdr_input.experiments.audio.source_estimator import nearest_indices
from bpdr_input.preparation.diagnostics.support_fragmentation import diagnose, intervals

VERSION = 'segment-local-phase-diagnostic-0.1'
MINIMUM_RAW_SAMPLES = 4411
RENDERER_MINIMUM_SAMPLES = 18963


def interval_summary(frequency, retained):
    """Whole frozen intervals only; bad samples never split an interval."""
    frequency = np.asarray(frequency)
    if frequency.shape != retained.shape or retained.dtype != bool:
        raise ValueError('Frequency and fixed support grids differ')
    rows = []
    for region in intervals(retained):
        values = frequency[region['start_sample']:region['end_sample']]
        nonfinite = int((~np.isfinite(values)).sum())
        nonpositive = int((np.isfinite(values) & (values <= 0)).sum())
        feasible = nonfinite == nonpositive == 0
        rows.append(dict(**region, nonfinite_samples=nonfinite, nonpositive_samples=nonpositive,
                         phase_feasible=feasible, reaches_renderer_minimum=region['samples'] >= RENDERER_MINIMUM_SAMPLES,
                         phase_feasible_and_long_enough=feasible and region['samples'] >= RENDERER_MINIMUM_SAMPLES))
    return dict(intervals=rows, retained_samples=int(retained.sum()), retained_intervals=len(rows),
                phase_feasible_intervals=sum(r['phase_feasible'] for r in rows),
                phase_feasible_samples=sum(r['samples'] for r in rows if r['phase_feasible']),
                phase_feasible_long_intervals=sum(r['phase_feasible_and_long_enough'] for r in rows),
                nonfinite_on_retained=sum(r['nonfinite_samples'] for r in rows),
                nonpositive_on_retained=sum(r['nonpositive_samples'] for r in rows))


def estimate_segment_phase(audio_float32, sample_rate_hz, frozen_arrays):
    """Only samples/rate and saved source estimates enter; no truth or labels.

    Local phase has independent origins on each raw reliable island. Fixed
    retained support is a comparison domain, not validated local amplitude or
    resynthesis support. No audio output or whole-source acceptance is returned.
    """
    audio = np.asarray(audio_float32); sr = sample_rate_hz
    if (audio.ndim != 1 or audio.dtype != np.float32 or len(audio) < 2 or sr != 22050
            or not np.isfinite(audio).all()):
        raise ValueError('Expected finite mono float32 at 22050 Hz')
    diagnose(frozen_arrays, sr)
    if len(frozen_arrays['sample_times']) != len(audio):
        raise ValueError('Audio and frozen sample grids differ')
    n = len(audio)
    arrays = dict(local_sine_phase=np.full(n, np.nan), local_amplitude=np.full(n, np.nan),
                  local_frequency_hz=np.full(n, np.nan), local_computed=np.zeros(n, dtype=bool))
    for key in ('sample_reliable', 'guarded_support', 'retained_support'):
        arrays[key] = frozen_arrays[key].copy()
    indices = nearest_indices(frozen_arrays['frame_times'], frozen_arrays['sample_times'])
    islands = []
    for region in intervals(arrays['sample_reliable'], sr):
        row = dict(**region, status='refused', reason=None, local_center_hz=None)
        islands.append(row)
        if region['samples'] < MINIMUM_RAW_SAMPLES:
            row['reason'] = 'SHORT_RAW_ISLAND'; continue
        lo, hi = region['start_sample'], region['end_sample']
        represented = np.unique(indices[lo:hi])
        if not frozen_arrays['frame_reliable'][represented].all():
            raise ValueError('Unreliable frame inside frozen reliable island')
        center = float(np.median(frozen_arrays['frame_yin_hz'][represented]))
        row['local_center_hz'] = center
        if not 0 < .7*center < 1.3*center < sr/2:
            row['reason'] = 'INVALID_LOCAL_BAND'; continue
        try:
            sos = butter(4, [.7*center, 1.3*center], btype='bandpass', fs=sr, output='sos')
            analytic = hilbert(sosfiltfilt(sos, audio[lo:hi].astype(np.float64), padtype='odd', padlen=27))
        except (ValueError, FloatingPointError) as exc:
            row.update(reason='LOCAL_FILTER_ERROR', error_type=type(exc).__name__, error=str(exc)); continue
        if not np.isfinite(analytic).all():
            row['reason'] = 'NONFINITE_LOCAL_ANALYTIC_SIGNAL'; continue
        phase = np.unwrap(np.angle(analytic))
        frequency = np.gradient(phase)*sr/(2*np.pi)
        arrays['local_sine_phase'][lo:hi] = phase+np.pi/2
        arrays['local_amplitude'][lo:hi] = abs(analytic)
        arrays['local_frequency_hz'][lo:hi] = frequency
        arrays['local_computed'][lo:hi] = True
        row.update(status='computed', nonfinite_frequency_samples=int((~np.isfinite(frequency)).sum()),
                   nonpositive_frequency_samples=int((np.isfinite(frequency) & (frequency <= 0)).sum()))
    result = interval_summary(arrays['local_frequency_hz'], arrays['retained_support'])
    for row in result['intervals']:
        matches = [i for i, island in enumerate(islands)
                   if island['start_sample'] <= row['start_sample'] and island['end_sample'] >= row['end_sample']]
        if len(matches) != 1:
            raise ValueError('Retained interval must belong to exactly one raw island')
        row['raw_island_index'] = matches[0]
    result.update(version=VERSION, status='diagnostic_only', raw_islands=islands,
                  computed_islands=sum(r['status'] == 'computed' for r in islands),
                  short_islands=sum(r['reason'] == 'SHORT_RAW_ISLAND' for r in islands),
                  production_eligible=False, accepted_audio=None, arrays=arrays)
    return result
