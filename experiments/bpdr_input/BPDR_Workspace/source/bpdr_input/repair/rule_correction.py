"""Conservative rule-based pitch-centre correction: the proposal's non-learning reference condition (Table 2).

Rule (proposal 3.3): estimate the singer's own overall tuning offset, then apply a limited chromatic pitch-centre
correction to stable held notes. No key or scale is assumed: the grid is all twelve semitones shifted by the
singer's offset, so a whole-phrase transposition produces no edit. Each corrected note gets one constant
correction, applied by scaling the measured pitch tier, so vibrato and contour inside the note are kept.

Every proposed edit then goes through the existing component path, unchanged:
  repair_controller.plan_fixture      permission limits (max 30 cents, support >= 0.86 s, affected fraction <= 0.25)
  framewise_renderer.render_candidate Praat pitch-tier rendering, samples outside the support copied exactly
  framewise_audit.audit_curve         re-measurement of realized shift and outside drift (PESTO and YIN)
  repair_controller.transact_fixture  rollback on any failure
Thresholds are fixed engineering values, not calibrated on development data or listening tests.
Limitation: YIN both decides the correction and checks the realized shift (with PESTO as the second tracker).
"""
from dataclasses import asdict, dataclass
import numpy as np
from bpdr_input.repair.interventions import independent_pitch
from bpdr_input.repair.repair_controller import FixturePolicy, plan_fixture, transact_fixture

HOP_SECONDS = .02


@dataclass(frozen=True)
class RulePolicy:
    version: str = 'rule-chromatic-0.1'
    min_deviation_cents: float = 10.     # smaller deviations are left alone
    max_deviation_cents: float = 30.     # larger ones may be intended or misread: withheld, never clipped
    max_note_spread_cents: float = 12.   # detrended interior spread above this = vibrato/slide/unstable: withheld
    note_split_cents: float = 50.        # frame-to-frame jump that starts a new note
    tracker_agreement_cents: float = 50. # YIN and PESTO must agree for a frame to count as reliable
    interior_trim_seconds: float = .10   # note onset/offset excluded from the centre estimate
    min_note_seconds: float = .86        # equals the renderer/controller minimum support

    def __post_init__(self):
        if not 0 < self.min_deviation_cents < self.max_deviation_cents <= FixturePolicy().max_correction_cents:
            raise ValueError('Invalid deviation limits')


def fine_pitch(audio, sr, times):
    """YIN cents re A440 on the 20-ms feature grid (nearest frame)."""
    f0 = independent_pitch(audio, sr)
    t = np.arange(len(f0))*441/sr
    idx = np.clip(np.searchsorted(t, times), 0, len(t)-1)
    prev = np.maximum(idx-1, 0)
    idx = np.where(np.abs(t[prev]-times) < np.abs(t[idx]-times), prev, idx)
    return 1200*np.log2(np.maximum(f0[idx], 1e-6)/440.)


def reliable_frames(features, cents, policy=RulePolicy()):
    pesto = 1200*np.log2(np.where(features['voiced_mask'], np.nan_to_num(features['f0_hz'], nan=1.), 1.)/440.)
    return features['voiced_mask'].astype(bool) & np.isfinite(cents) & (np.abs(cents-pesto) <= policy.tracker_agreement_cents)


def tuning_offset(cents, mask):
    """Circular mean of pitch modulo 100 cents: the singer's own grid offset, in (-50, 50]."""
    if mask.sum() < 10:
        return None
    return float(np.angle(np.mean(np.exp(2j*np.pi*cents[mask]/100)))*100/(2*np.pi))


def segment_notes(cents, mask, policy=RulePolicy()):
    """Contiguous reliable runs, split where pitch leaves the level of the note's last 25 frames by more than
    note_split_cents. A gradual glide between notes is split too, not only a single-frame jump; the 0.5-s
    (25-frame) level window spans about three 5.5-Hz vibrato cycles, so vibrato stays one note."""
    notes, start = [], None
    for i in range(len(cents)+1):
        inside = i < len(cents) and mask[i]
        # Until a note has 10 frames its level is unknown: split only on a frame-to-frame jump.
        level = np.median(cents[max(start, i-25):i]) if start is not None and i-start >= 10 else (cents[i-1] if i else 0.)
        if inside and start is not None and abs(cents[i]-level) > policy.note_split_cents:
            notes.append((start, i)); start = i
        elif inside and start is None:
            start = i
        elif not inside and start is not None:
            notes.append((start, i)); start = None
    return notes


def decide(cents, mask, policy=RulePolicy()):
    """Per-note decisions and the frame correction curve (cents). Nothing here renders audio."""
    corrections = np.zeros(len(cents)); uncertainty = np.zeros(len(cents))
    offset = tuning_offset(cents, mask)
    decisions = []
    if offset is None:
        return corrections, uncertainty, offset, decisions
    trim = int(round(policy.interior_trim_seconds/HOP_SECONDS))
    for a, b in segment_notes(cents, mask, policy):
        row = dict(start_seconds=a*HOP_SECONDS, end_seconds=b*HOP_SECONDS, frames=b-a)
        interior = np.arange(a+trim, b-trim)
        if (b-a)*HOP_SECONDS < policy.min_note_seconds or len(interior) < 5:
            decisions.append(dict(row, action='NO_EDIT', reason='NOTE_TOO_SHORT')); continue
        x = cents[interior]
        centre = float(np.median(x))
        slope = np.polyfit(np.arange(len(x)), x, 1)
        spread = float(np.std(x-np.polyval(slope, np.arange(len(x)))))
        deviation = float(((centre-offset+50) % 100)-50)
        drift = float(slope[0]*len(x))
        row.update(centre_cents=centre, deviation_cents=deviation, detrended_spread_cents=spread, drift_cents=drift)
        if spread > policy.max_note_spread_cents or abs(drift) > policy.max_note_spread_cents:
            decisions.append(dict(row, action='WITHHELD', reason='UNSTABLE_NOTE_VIBRATO_OR_SLIDE')); continue
        if abs(deviation) < policy.min_deviation_cents:
            decisions.append(dict(row, action='NO_EDIT', reason='WITHIN_TOLERANCE')); continue
        if abs(deviation) > policy.max_deviation_cents:
            decisions.append(dict(row, action='WITHHELD', reason='DEVIATION_ABOVE_LIMIT')); continue
        corrections[a:b] = -deviation
        uncertainty[a:b] = 1.2533*spread/np.sqrt(len(x))
        decisions.append(dict(row, action='CORRECT', correction_cents=-deviation))
    return corrections, uncertainty, offset, decisions


def run(audio, sr, features, analyze, policy=RulePolicy()):
    """Decide, permit, render, audit and roll back. `analyze(audio)` returns feature dicts for re-measurement.

    Returns (output audio, report). Output is the unchanged input unless a candidate passed every check.
    """
    from bpdr_input.repair.framewise_audit import audit_curve
    from bpdr_input.repair.framewise_renderer import render_candidate
    times = features['times_seconds']
    cents = fine_pitch(audio, sr, times)
    mask = reliable_frames(features, cents, policy)
    corrections, uncertainty, offset, decisions = decide(cents, mask, policy)
    plan = plan_fixture(audio, sr, times, corrections, corrections, uncertainty, mask, np.zeros(len(times), bool))
    report = dict(policy=asdict(policy), tuning_offset_cents=offset, reliable_frames=int(mask.sum()), notes=decisions, plan=plan)

    def renderer(candidate_audio, rate, intervals):
        return render_candidate(candidate_audio, rate, times, corrections, intervals)[0]

    def auditor(before_audio, after_audio, rate, intervals):
        after = analyze(after_audio)
        return audit_curve(features, after, fine_hz(before_audio, rate, times), fine_hz(after_audio, rate, times), mask, intervals, corrections)

    output, transaction = transact_fixture(audio, sr, plan, renderer, auditor)
    report['transaction'] = transaction
    report['outcome'] = transaction['outcome']
    report['corrections_cents'] = corrections.tolist()
    report['fine_pitch_cents'] = [float(c) if m else None for c, m in zip(cents, mask)]
    return output, report


def fine_hz(audio, sr, times):
    return 440.*2**(fine_pitch(audio, sr, times)/1200)
