"""Provisional acoustic evidence for a declared vocal idea, not musical approval.

The original datasets have performed-pitch references, not validated labels of
novice input usefulness or intended perfect notes. This module therefore does
not train or claim a speech/singing classifier. The caller supplies intent and
keeps the technical ProcessingResult separate.

The frozen demo settings below are engineering prerequisites for inspecting a
short phrase: 0.8 reuses the analyzer's voiced confidence, 50 cents reuses the
component's two-tracker agreement convention, and 0.20/0.12 seconds require a
little total/contiguous support rather than isolated frames. Majority tracker
agreement prevents a few agreeable frames from representing a mostly disputed
track. These are unvalidated demo settings, not population-calibrated musical
usability thresholds. No minimum note count, pitch variation, scale membership,
steady pitch, or whole-recording voiced fraction is required. Slides, repeated
notes and pauses must not be mistaken for errors.
"""
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np


class EvidenceDecision(StrEnum):
    PASS_EVIDENCE = "PASS_EVIDENCE"
    NEED_CLEARER_INPUT = "NEED_CLEARER_INPUT"
    NEED_INTENT_CONFIRMATION = "NEED_INTENT_CONFIRMATION"
    NOT_ASSESSED = "NOT_ASSESSED"


class DeclaredIntent(StrEnum):
    SINGING_HUMMING_CHANT = "singing_humming_chant"
    READING_SPOKEN_TEXT = "reading_spoken_text"
    UNSURE = "unsure"


@dataclass(frozen=True)
class UsabilityPolicy:
    version: str = "provisional-melodic-evidence-demo-0.1"
    minimum_confidence: float = .8
    maximum_tracker_disagreement_cents: float = 50.
    minimum_reliable_seconds: float = .20
    minimum_contiguous_reliable_seconds: float = .12
    minimum_agreeing_fraction_of_primary_voiced: float = .5
    scope: str = "Fixed unvalidated demo engineering settings; no musical approval"

    def __post_init__(self):
        values = (self.minimum_confidence, self.maximum_tracker_disagreement_cents,
                  self.minimum_reliable_seconds, self.minimum_contiguous_reliable_seconds,
                  self.minimum_agreeing_fraction_of_primary_voiced)
        if not all(np.isfinite(values)):
            raise ValueError("Policy settings must be finite")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        if not 0 <= self.minimum_agreeing_fraction_of_primary_voiced <= 1:
            raise ValueError("Agreement fraction must be in [0, 1]")
        if min(values[1:4]) <= 0:
            raise ValueError("Agreement and duration settings must be positive")


LIMITATIONS = [
    "Provisional acoustic evidence only; not a speech/singing or musical correctness classifier.",
    "Tracker confidence and agreement do not establish the singer's intended notes or scale.",
    "No independent musical-usability labels or calibrated production thresholds are available.",
    "This assessment does not approve edits or change the technical handoff.",
]


def _runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _longest(mask, weights):
    return max((float(weights[a:b].sum()) for a, b in _runs(mask)), default=0.)


def _arrays(features, independent_f0_hz, duration_seconds):
    """Validate aligned measurements; NaN F0 is an allowed unvoiced marker."""
    keys = ("times_seconds", "f0_hz", "confidence", "voiced_mask")
    if not isinstance(features, dict) or any(key not in features for key in keys):
        raise ValueError("Missing aligned time, pitch, confidence or voicing measurements")
    times, pitch, confidence = [np.asarray(features[k], dtype=np.float64) for k in keys[:3]]
    raw_voiced = np.asarray(features["voiced_mask"])
    arrays = [times, pitch, confidence, raw_voiced]
    if any(a.ndim != 1 for a in arrays) or len(times) < 2 or any(len(a) != len(times) for a in arrays):
        raise ValueError("Measurements require at least two frames on one aligned grid")
    if (not np.isfinite(times).all() or np.any(times < 0) or not np.isclose(times[0], 0., atol=1e-9)
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Timestamps must be finite, increasing and start at zero")
    hop = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), hop, rtol=1e-4, atol=1e-9):
        raise ValueError("Measurements must share a regular aligned frame grid")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Confidence must be finite and in [0, 1]")
    if not np.isin(raw_voiced, [False, True]).all():
        raise ValueError("Voicing flags must be Boolean")
    if np.isinf(pitch).any() or np.any(pitch[np.isfinite(pitch)] < 0):
        raise ValueError("Pitch must be nonnegative Hz or NaN for unvoiced frames")
    if "log_energy_db" in features:
        energy = np.asarray(features["log_energy_db"], dtype=np.float64)
        if energy.shape != times.shape or not np.isfinite(energy).all():
            raise ValueError("Energy measurements must be finite on the same frame grid")
    if duration_seconds is None:
        duration = float(times[-1] + hop)
        duration_basis = "Analysis frame-span estimate, not a decoded sample count"
    else:
        duration = float(duration_seconds)
        if not np.isfinite(duration) or duration <= 0 or times[-1] > duration + 1e-9:
            raise ValueError("Analysis frame timestamps leave the provided audio duration")
        duration_basis = "Caller-provided decoded audio duration"
    weights = np.clip(duration - times, 0., hop)
    secondary = None
    if independent_f0_hz is not None:
        secondary = np.asarray(independent_f0_hz, dtype=np.float64)
        if secondary.shape != times.shape:
            raise ValueError("Independent tracker must be aligned to the same frame grid")
        if np.isinf(secondary).any() or np.any(secondary[np.isfinite(secondary)] < 0):
            raise ValueError("Independent pitch must be nonnegative Hz or NaN")
    return times, pitch, confidence, raw_voiced.astype(bool), secondary, weights, duration, duration_basis


def assess(features, technical_status, intent="unsure", independent_f0_hz=None,
           policy=UsabilityPolicy(), *, duration_seconds=None):
    """Return JSON-safe observations and a provisional decision without mutation.

    ``features`` uses ``audio.analysis.extract`` field names. Supply an already
    aligned independent tracker (normally YIN); this function never loads a
    model or analyzes a waveform. ``duration_seconds`` should come from decoded
    samples when available so out-of-range timestamps can be rejected. Missing
    or malformed measurements abstain instead of implying singing is wrong.
    """
    technical = str(getattr(technical_status, "value", technical_status))
    declared = str(getattr(intent, "value", intent))
    if declared not in {item.value for item in DeclaredIntent}:
        declared = DeclaredIntent.UNSURE.value
    result = dict(schema_version="provisional-melodic-evidence-1", provisional=True,
                  technical_status=technical, declared_intent=declared,
                  policy=asdict(policy), observations={}, limitations=LIMITATIONS.copy(),
                  musical_correctness_assessed=False, intended_note_identified=False,
                  handoff_approval=False)

    def finish(decision, reasons, feedback, next_step):
        result.update(decision=decision.value, reason_codes=list(reasons),
                      feedback=feedback, next_step=next_step)
        return result

    if technical not in {"ACCEPT_UNCHANGED", "ACCEPT_PREPARED"}:
        return finish(EvidenceDecision.NOT_ASSESSED, ["TECHNICAL_INPUT_NOT_ACCEPTED"],
                      "Technical checks did not accept this recording. Resolve the reported recording or file issue before assessing melodic evidence.",
                      "FOLLOW_TECHNICAL_FEEDBACK")
    if declared == DeclaredIntent.READING_SPOKEN_TEXT:
        return finish(EvidenceDecision.NEED_CLEARER_INPUT, ["NO_INTENDED_TUNE_DECLARED"],
                      "You identified this as reading or spoken text without an intended tune. Please sing or hum the melody you want the next component to use; a repeated note or chant is also allowed.",
                      "RECORD_INTENDED_TUNE")
    if features is None:
        return finish(EvidenceDecision.NOT_ASSESSED, ["ANALYSIS_UNAVAILABLE"],
                      "Melodic evidence was not measured. This is not a judgment of your singing.",
                      "RETRY_ANALYSIS")
    try:
        times, pitch, confidence, voiced, secondary, weights, duration, duration_basis = _arrays(
            features, independent_f0_hz, duration_seconds)
    except (ValueError, TypeError, OverflowError) as exc:
        result["observations"]["analysis_issue"] = str(exc)
        return finish(EvidenceDecision.NOT_ASSESSED, ["ANALYSIS_INVALID"],
                      "The acoustic measurements are incomplete or inconsistent. Retry analysis before deciding whether another take is needed.",
                      "RETRY_ANALYSIS")
    primary = voiced & (confidence >= policy.minimum_confidence) & np.isfinite(pitch) & (pitch > 0)
    obs = result["observations"]
    obs.update(frame_count=len(times), duration_seconds=duration, duration_basis=duration_basis,
               primary_voiced_frame_count=int(primary.sum()),
               primary_voiced_seconds=float(weights[primary].sum()),
               primary_voiced_fraction_of_all_frames=float(primary.mean()),
               independent_tracker_available=secondary is not None)
    if not primary.any():
        return finish(EvidenceDecision.NEED_CLEARER_INPUT, ["NO_RELIABLE_VOICED_EVIDENCE"],
                      "The analyzer could not recover reliable pitched voice in this take. Try a short solo hum or sung phrase closer to the microphone in a quieter place.",
                      "RECORD_CLEARER_TUNE")
    if secondary is None:
        return finish(EvidenceDecision.NOT_ASSESSED, ["INDEPENDENT_TRACKER_UNAVAILABLE"],
                      "One tracker found pitched voice, but an independent pitch measurement is unavailable. Retry analysis; a single pitch track cannot establish usable melody or intent.",
                      "RETRY_ANALYSIS")
    common = primary & np.isfinite(secondary) & (secondary > 0)
    disagreement = np.full(len(times), np.nan)
    disagreement[common] = abs(1200*np.log2(secondary[common]/pitch[common]))
    reliable = common & (disagreement <= policy.maximum_tracker_disagreement_cents)
    reliable_seconds = float(weights[reliable].sum())
    fraction = float(reliable.sum()/primary.sum())
    reliable_runs = _runs(reliable)
    # Only internal unvoiced gaps are pauses; leading/trailing silence is not a musical error.
    first, last = np.flatnonzero(primary)[[0, -1]]
    internal = np.zeros(len(times), dtype=bool)
    internal[first:last+1] = ~primary[first:last+1]
    obs.update(tracker_comparable_frame_count=int(common.sum()),
               tracker_agreeing_frame_count=int(reliable.sum()),
               agreeing_fraction_of_primary_voiced=fraction,
               reliable_fraction_of_all_frames=float(reliable.mean()),
               reliable_seconds=reliable_seconds,
               longest_contiguous_reliable_seconds=_longest(reliable, weights),
               contiguous_reliable_region_count=len(reliable_runs),
               internal_unvoiced_gap_count=len(_runs(internal)),
               longest_internal_unvoiced_gap_seconds=_longest(internal, weights),
               median_tracker_disagreement_cents=float(np.median(disagreement[common])) if common.any() else None)
    if reliable.any():
        cents = 1200*np.log2(pitch[reliable]/440.)
        obs["reliable_pitch_p90_minus_p10_cents"] = float(np.percentile(cents, 90)-np.percentile(cents, 10))
    reasons = []
    if reliable_seconds + 1e-9 < policy.minimum_reliable_seconds:
        reasons.append("TOO_LITTLE_AGREEING_PITCH_SUPPORT")
    if obs["longest_contiguous_reliable_seconds"] + 1e-9 < policy.minimum_contiguous_reliable_seconds:
        reasons.append("PITCH_SUPPORT_TOO_FRAGMENTED")
    if fraction + 1e-9 < policy.minimum_agreeing_fraction_of_primary_voiced:
        reasons.append("TRACKERS_DISAGREE_ON_MUCH_OF_VOICED_INPUT")
    if reasons:
        return finish(EvidenceDecision.NEED_CLEARER_INPUT, reasons,
                      "The two pitch measurements do not provide enough shared continuous evidence for this demo. Try a clearer solo take with a short audible hum or sung phrase; pauses, repeated notes and slides are allowed.",
                      "RECORD_CLEARER_TUNE")
    if declared == DeclaredIntent.UNSURE:
        return finish(EvidenceDecision.NEED_INTENT_CONFIRMATION, ["INTENDED_TUNE_NOT_CONFIRMED"],
                      "There is measurable pitched voice, but pitch alone cannot tell whether you intended a tune or ordinary speech. Confirm whether this is the sung, hummed or chanted idea you want to preserve.",
                      "CONFIRM_INTENDED_TUNE")
    return finish(EvidenceDecision.PASS_EVIDENCE, ["PROVISIONAL_ACOUSTIC_SUPPORT_PRESENT"],
                  "Your declared vocal idea has enough agreeing pitch evidence for a demo review. Preserve it unchanged as the baseline and compare any optional preview. This does not certify musical correctness or identify your intended notes.",
                  "REVIEW_UNCHANGED_BASELINE_AND_OPTIONAL_PREVIEW")
