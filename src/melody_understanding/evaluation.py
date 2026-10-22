"""MIDI-reference pilot metrics and controlled note segmentation variants."""
import numpy as np
from scipy.ndimage import median_filter
import mir_eval.transcription
from .features import segment_notes


# Smoothing Pitch Only within Continuous Voiced Regions
def notes_with_smoothing(features, end_s, width=1):
    """Median-filter F0 within voiced runs only, then reuse baseline grouping."""
    # Require an odd filter window so the median has a centered neighborhood.
    if width < 1 or width % 2 != 1:
        raise ValueError("Smoothing width must be a positive odd integer")
    values = features['f0_hz'].copy()
    voiced = features['voiced'] & np.isfinite(values) & (values > 0)
    # Never fill a rest or smooth across an unvoiced boundary.
    bounds = np.flatnonzero(np.diff(np.r_[False, voiced, False].astype(int)))
    for start, stop in zip(bounds[::2], bounds[1::2]):
        values[start:stop] = median_filter(values[start:stop], size=width, mode='nearest')
    return segment_notes({**features, 'f0_hz': values}, end_s)


# Matching Predicted Notes to MIDI Reference Events
def score_notes(reference, predicted):
    """Score onset+pitch and onset+pitch+offset using one-to-one matching."""
    if not reference:
        raise ValueError('Cannot evaluate without reference notes')
    if not predicted:
        return {'note_precision': 0., 'note_recall': 0., 'note_f1': 0., 'note_offset_f1': 0.}
    # Convert note dictionaries to intervals and MIDI pitches to frequencies.
    ref = np.asarray(reference, dtype=float)
    est = np.array([[n['onset_s'], n['onset_s']+n['duration_s'], n['pitch_midi']] for n in predicted])
    to_hz = lambda midi: 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    # Score onset and pitch matches first, without requiring matching offsets.
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref[:, :2], to_hz(ref[:, 2]), est[:, :2], to_hz(est[:, 2]),
        onset_tolerance=.05, pitch_tolerance=50, offset_ratio=None)
    # Repeat with an offset tolerance of 20% of note duration or at least 50 ms.
    _, _, offset_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref[:, :2], to_hz(ref[:, 2]), est[:, :2], to_hz(est[:, 2]),
        onset_tolerance=.05, pitch_tolerance=50, offset_ratio=.2, offset_min_tolerance=.05)
    return {'note_precision': float(precision), 'note_recall': float(recall),
            'note_f1': float(f1), 'note_offset_f1': float(offset_f1)}


# Measuring Pitch and Voicing Agreement on the Frame Grid
def score_frames(reference, features):
    """Measure agreement with MIDI-held pitches, not continuous F0 ground truth."""
    # Assign one reference pitch per frame and reject overlapping reference notes.
    times = features['times_s']
    target = np.full(len(times), np.nan)
    for start, end, pitch in reference:
        mask = (times >= start) & (times < end)
        if np.any(np.isfinite(target[mask])):
            raise ValueError('Reference is not monophonic on the evaluation grid')
        target[mask] = pitch
    # Compare valid estimated MIDI pitches against reference pitches within 50 cents.
    ref_voiced = np.isfinite(target)
    f0 = features['f0_hz']
    est_voiced = features['voiced'] & np.isfinite(f0) & (f0 > 0)
    est = np.full(len(times), np.nan)
    est[est_voiced] = 69 + 12 * np.log2(f0[est_voiced] / 440)
    correct = ref_voiced & est_voiced & (np.abs(est - target) < .5)
    if not ref_voiced.any():
        raise ValueError('No reference voiced frames on grid')
    # Report voiced pitch agreement, overall agreement including rests, and voicing recall.
    return {
        'midi_frame_pitch_agreement_50c': float(correct.sum() / ref_voiced.sum()),
        'midi_frame_overall_agreement_50c': float(np.mean(correct | (~ref_voiced & ~est_voiced))),
        'midi_voicing_recall': float((ref_voiced & est_voiced).sum() / ref_voiced.sum()),
    }
