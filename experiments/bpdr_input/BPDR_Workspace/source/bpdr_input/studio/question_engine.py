"""Active Intent Confirmation (AIC) Question Engine for Component 1.

Treats singer intent as a latent variable with a calibrated posterior across the
two BPDR factors:
  1. Boundary: is an ambiguous transition one held note or two notes?
  2. Pitch: is a steady 18-45 c deviation an intended inflection ("as sung")
     or an unintended error ("centred on nearest note")?

Selects up to K (default 3, hard cap 5) A/B listening cards ranked by
Expected Information Gain * Downstream Impact / Listening Cost, with a
dedicated pitch-question budget slot so impact weighting never starves pitch
clarification.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from . import boundary, evidence as E, span_repair, synth

DEFAULT_K = 3
MAX_K = 5
LISTENER_FLIP_EPS = 0.10

DEFAULT_CALIBRATION = {
    "boundary_temp": 1.35,
    "boundary_bias": 0.0,
    "pitch_alpha0": 0.15,
    "pitch_alpha1": 0.85,
    "pitch_alpha2": 0.45,
    "pitch_alpha3": 0.30,
}


def policy_hash(params: Optional[Dict[str, Any]] = None) -> str:
    """Stable SHA-256 hash of the AIC question-selection policy parameters."""
    cfg = {
        "default_k": DEFAULT_K,
        "max_k": MAX_K,
        "listener_flip_eps": LISTENER_FLIP_EPS,
        "calibration": params or DEFAULT_CALIBRATION,
        "version": "aic_v0.2",
    }
    raw = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def _logit(p: float) -> float:
    p = float(np.clip(p, 0.02, 0.98))
    return float(np.log(p / (1.0 - p)))


def binary_entropy(p: float) -> float:
    """Binary Shannon entropy H(p) in bits."""
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def expected_information_gain(p: float, eps: float = LISTENER_FLIP_EPS) -> float:
    """Expected information gain I(Z; Y) for a binary latent Z and noisy listener flip eps."""
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    eps = float(np.clip(eps, 0.0, 0.49))
    q = p * (1.0 - eps) + (1.0 - p) * eps
    return max(0.0, binary_entropy(q) - binary_entropy(eps))


def _local_note_pair(notes: Sequence[Dict[str, Any]], t_b: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Find the note immediately before/covering t_b and the note immediately after t_b."""
    if not notes:
        return None, None
    sorted_notes = sorted(notes, key=lambda n: n["onset_s"])
    left, right = None, None
    for n in sorted_notes:
        if n["onset_s"] < t_b - 0.03:
            left = n
        elif n["onset_s"] >= t_b - 0.03 and right is None:
            right = n
            break
    if left is None and len(sorted_notes) >= 2 and sorted_notes[0]["onset_s"] <= t_b:
        left = sorted_notes[0]
        right = sorted_notes[1]
    return left, right


def extract_ambiguity_candidates(
    trk: Dict[str, Any],
    x: Optional[np.ndarray] = None,
    sr: int = E.SR,
    boundary_hints: Optional[Dict[str, Any]] = None,
    span_diagnostics: Optional[Dict[str, Any]] = None,
    duration_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Extract boundary (split/merge) and pitch (as-sung vs centred) ambiguity candidates."""
    aligned = boundary.align_track_to_20ms(trk, sr=sr)
    t_grid = np.asarray(aligned["t"], dtype=float)
    f0 = np.asarray(aligned["f0"], dtype=float)
    voiced = np.asarray(aligned["voiced"], dtype=bool) & np.isfinite(f0) & (f0 > 0)
    if len(t_grid) == 0:
        return []

    dur = float(duration_s) if duration_s is not None else float(t_grid[-1] + aligned["hop_s"])

    if boundary_hints is None:
        boundary_hints = boundary.extract_boundary_hints(aligned, x=x, sr=sr)
    if span_diagnostics is None:
        span_diagnostics = boundary.analyze_boundary_spans(aligned, boundary_hints, end_s=dur, sr=sr)

    probs_t = np.asarray(boundary_hints.get("times_s", t_grid), dtype=float)
    probs = np.asarray(boundary_hints.get("probabilities", np.zeros(len(probs_t))), dtype=float)
    model_peaks = np.asarray(boundary_hints.get("peaks_s", []), dtype=float)

    base_notes = E.notes_from_track(aligned, dur)
    prov_edges = boundary.provisional_boundary_edges(aligned, x=x, sr=sr)
    ext_onsets = sorted(set(
        [round(float(n["onset_s"]), 3) for n in base_notes[1:]]
        + [round(float(e), 3) for e in prov_edges]
    ))

    candidates: List[Dict[str, Any]] = []
    seen_times: List[float] = []

    def _is_near_seen(tb: float, tol: float = 0.14) -> bool:
        return any(abs(tb - s) < tol for s in seen_times)

    def _p_model_near(tb: float, win_s: float = 0.05) -> float:
        m = np.abs(probs_t - tb) <= win_s
        return float(np.max(probs[m])) if m.any() else 0.5

    def _voiced_sides_ok(tb: float, min_frames: int = 4) -> bool:
        left_ok = int(np.sum(voiced & (t_grid >= tb - 0.45) & (t_grid < tb)))
        right_ok = int(np.sum(voiced & (t_grid >= tb) & (t_grid <= tb + 0.45)))
        return left_ok >= min_frames and right_ok >= min_frames

    # 1a. Boundary: possible_false_split (extractor/provisional edge with low FastTNet support)
    for tb in ext_onsets:
        if tb <= 0.12 or tb >= dur - 0.12 or _is_near_seen(tb):
            continue
        if not _voiced_sides_ok(tb, min_frames=3):
            continue
        p_mod = _p_model_near(tb, win_s=0.05)
        if p_mod >= 0.72:
            continue
        left_n, right_n = _local_note_pair(base_notes, tb)
        delta_midi = abs(left_n["pitch_midi"] - right_n["pitch_midi"]) if (left_n and right_n) else 1
        if delta_midi > 2:
            continue
        t0 = float(left_n["onset_s"]) if left_n else max(0.0, tb - 0.45)
        t1 = float(right_n["onset_s"] + right_n["duration_s"]) if right_n else min(dur, tb + 0.45)
        t0 = max(0.0, min(t0, tb - 0.12))
        t1 = min(dur, max(t1, tb + 0.12))
        span_dur = max(0.16, t1 - t0)
        midi_l = int(left_n["pitch_midi"]) if left_n else int(round(69 + 12 * np.log2(np.median(f0[voiced]) / 440.0)))
        midi_r = int(right_n["pitch_midi"]) if right_n else midi_l

        seen_times.append(tb)
        candidates.append({
            "qid": f"b_split_{len(candidates) + 1}",
            "factor": "boundary",
            "subtype": "possible_false_split",
            "boundary_s": round(tb, 3),
            "start_s": round(t0, 3),
            "end_s": round(t1, 3),
            "span_duration_s": round(span_dur, 3),
            "raw_p_model": round(p_mod, 4),
            "delta_midi": int(delta_midi),
            "midi_left": midi_l,
            "midi_right": midi_r,
            "midi_merged": int(round(0.5 * (midi_l + midi_r))),
            "question_text": f"At {tb:.2f} s, did you sing one held note or two separate notes?",
            "option_a": {"id": "keep_merged", "label": "One held note", "action": "merge"},
            "option_b": {"id": "split_two", "label": "Two notes (split)", "action": "split"},
        })

    # 1b. Boundary: possible_false_merge (FastTNet uncertain/sub-threshold peak or missed repeat)
    sub_pk_idx, _ = find_peaks(probs, height=0.30, distance=6)
    cand_peaks = sorted(set([round(float(probs_t[i]), 3) for i in sub_pk_idx] + [round(float(p), 3) for p in model_peaks]))
    for tb in cand_peaks:
        if tb <= 0.12 or tb >= dur - 0.12 or _is_near_seen(tb):
            continue
        if not _voiced_sides_ok(tb, min_frames=4):
            continue
        p_mod = _p_model_near(tb, win_s=0.04)
        ext_near = any(abs(tb - eo) <= 0.07 for eo in ext_onsets)
        if not ((0.30 <= p_mod < 0.80) or (p_mod >= 0.65 and not ext_near)):
            continue
        left_n, right_n = _local_note_pair(base_notes, tb)
        # Compute local median MIDI on left and right of tb
        m_l = voiced & (t_grid >= max(0.0, tb - 0.40)) & (t_grid < tb)
        m_r = voiced & (t_grid >= tb) & (t_grid <= min(dur, tb + 0.40))
        midi_l = int(round(69.0 + 12.0 * np.log2(float(np.median(f0[m_l])) / 440.0))) if m_l.any() else 60
        midi_r = int(round(69.0 + 12.0 * np.log2(float(np.median(f0[m_r])) / 440.0))) if m_r.any() else midi_l
        delta_midi = abs(midi_l - midi_r)
        t0 = max(0.0, tb - 0.40)
        t1 = min(dur, tb + 0.40)
        if left_n and left_n["onset_s"] < tb:
            t0 = max(0.0, min(t0, float(left_n["onset_s"])))
        if right_n and right_n["onset_s"] + right_n["duration_s"] > tb:
            t1 = min(dur, max(t1, float(right_n["onset_s"] + right_n["duration_s"])))
        span_dur = max(0.16, t1 - t0)

        seen_times.append(tb)
        candidates.append({
            "qid": f"b_merge_{len(candidates) + 1}",
            "factor": "boundary",
            "subtype": "possible_false_merge",
            "boundary_s": round(tb, 3),
            "start_s": round(t0, 3),
            "end_s": round(t1, 3),
            "span_duration_s": round(span_dur, 3),
            "raw_p_model": round(p_mod, 4),
            "delta_midi": int(delta_midi),
            "midi_left": midi_l,
            "midi_right": midi_r,
            "midi_merged": int(round(0.5 * (midi_l + midi_r))),
            "question_text": f"At {tb:.2f} s, did you re-articulate two notes or hold one continuous note?",
            "option_a": {"id": "split_two", "label": "Two notes (re-articulated)", "action": "split"},
            "option_b": {"id": "keep_merged", "label": "One continuous note", "action": "merge"},
        })

    # 2. Pitch: steady spans in the 18-45 c repair-candidate band (or high BPDR multi-view sensitivity)
    for sp in span_diagnostics.get("spans", []):
        bpdr_sens = float(sp.get("bpdr_boundary_sensitivity_cents", 0.0))
        if not sp.get("review_repair_candidate") and bpdr_sens <= 6.0:
            continue
        dev = float(sp.get("tuning_adjusted_deviation_cents", 0.0))
        if abs(dev) < 12.0:
            continue
        rec = float(sp.get("recommended_shift_cents", 0.0) or np.clip(-dev, -35.0, 35.0))
        note_name = sp.get("note_name", "Note")
        direction = "flat" if dev < 0 else "sharp"
        bpdr_nom = float(sp.get("bpdr_nominal_shift_cents", rec))
        candidates.append({
            "qid": f"p_span_{sp['span_id']}",
            "factor": "pitch",
            "subtype": "bpdr_multiview_gated" if bpdr_sens > 0 else "tuning_deviation",
            "span_id": int(sp["span_id"]),
            "start_s": round(float(sp["start_s"]), 3),
            "end_s": round(float(sp["end_s"]), 3),
            "span_duration_s": round(float(sp["duration_s"]), 3),
            "note_name": note_name,
            "pitch_midi": int(sp.get("pitch_midi", 60)),
            "deviation_cents": round(dev, 1),
            "recommended_shift_cents": round(rec, 1),
            "bpdr_nominal_shift_cents": round(bpdr_nom, 1),
            "bpdr_boundary_sensitivity_cents": round(bpdr_sens, 2),
            "vibrato_std_cents": round(float(sp.get("vibrato_std_cents", 0.0)), 1),
            "mean_confidence": round(float(sp.get("mean_confidence", 0.8)), 3),
            "question_text": (
                f"From {sp['start_s']:.2f}–{sp['end_s']:.2f} s ({note_name}), pitch sits "
                f"{abs(dev):.0f} c {direction} (BPDR multi-view sensitivity {bpdr_sens:.1f} c). "
                f"Keep your inflection or apply BPDR centring on {note_name}?"
            ),
            "option_a": {"id": "as_sung", "label": "Keep as sung (preserve inflection)", "action": "preserve"},
            "option_b": {"id": "centred", "label": f"Centre on {note_name} ({rec:+.0f} c)", "action": "centre"},
        })

    return candidates


def calibrate_candidate_probabilities(
    candidates: List[Dict[str, Any]],
    calibration_params: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Attach calibrated prior probabilities, EIG, downstream impact, and cost."""
    cal = {**DEFAULT_CALIBRATION, **(calibration_params or {})}
    out = []
    for c in candidates:
        item = dict(c)
        clip_dur = min(3.0, max(1.0, float(item.get("span_duration_s", 0.5)) + 0.7))
        cost = 2.0 + clip_dur

        if item["factor"] == "boundary":
            p_raw = float(item.get("raw_p_model", 0.5))
            z = (_logit(p_raw) - float(cal["boundary_bias"])) / max(0.25, float(cal["boundary_temp"]))
            p_split = _sigmoid(z)
            eig = expected_information_gain(p_split)
            impact = float(item.get("span_duration_s", 0.4)) * (1.0 + float(item.get("delta_midi", 0))) * 10.0
            conf_gap = 1.0 - abs(2.0 * p_split - 1.0)
            item.update({
                "p_split": round(p_split, 4),
                "p_merge": round(1.0 - p_split, 4),
                "p_uncertain": round(min(p_split, 1.0 - p_split), 4),
                "eig_bits": round(eig, 4),
                "downstream_impact": round(impact, 3),
                "listening_cost_s": round(cost, 2),
                "lowest_confidence_score": round(conf_gap, 4),
                "eig_value_score": round((eig * impact) / cost, 4),
            })
        else:
            abs_dev = abs(float(item.get("deviation_cents", 25.0)))
            vib = float(item.get("vibrato_std_cents", 10.0))
            conf = float(item.get("mean_confidence", 0.8))
            z = (
                float(cal["pitch_alpha0"])
                + float(cal["pitch_alpha1"]) * ((abs_dev - 25.0) / 10.0)
                - float(cal["pitch_alpha2"]) * (vib / 25.0)
                + float(cal["pitch_alpha3"]) * (conf - 0.75)
            )
            p_error = _sigmoid(z)
            p_inflection = 1.0 - p_error
            eig = expected_information_gain(p_error)
            shift_mag = abs(float(item.get("recommended_shift_cents", 0.0)))
            dur_weight = min(1.5, float(item.get("span_duration_s", 0.3)) / 0.30)
            impact = shift_mag * p_error * dur_weight
            conf_gap = 1.0 - abs(2.0 * p_error - 1.0)
            item.update({
                "p_error": round(p_error, 4),
                "p_inflection": round(p_inflection, 4),
                "p_uncertain": round(min(p_error, p_inflection), 4),
                "eig_bits": round(eig, 4),
                "downstream_impact": round(impact, 3),
                "listening_cost_s": round(cost, 2),
                "lowest_confidence_score": round(conf_gap, 4),
                "eig_value_score": round((eig * impact) / cost, 4),
            })
        out.append(item)
    return out


def select_questions(
    candidates: Sequence[Dict[str, Any]],
    k: int = DEFAULT_K,
    max_k: int = MAX_K,
    strategy: str = "eig",
    eig_floor: float = 0.0,
    seed: int = 339,
    pitch_budget: int = 1,
) -> List[Dict[str, Any]]:
    """Select up to min(k, max_k) clarification questions under the chosen strategy.

    Strategies:
      - 'none': return []
      - 'random': deterministic pseudo-random sample
      - 'lowest_confidence': highest binary uncertainty (1 - |2p - 1|)
      - 'eig': greedy EIG * impact / cost with a dedicated pitch-question slot
    """
    k_eff = min(max(0, int(k)), int(max_k))
    if strategy == "none" or k_eff == 0 or not candidates:
        return []

    cands = [dict(c) for c in candidates]
    if strategy == "random":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(cands))[:k_eff]
        return [cands[int(i)] for i in idx]

    if strategy == "lowest_confidence":
        ranked = sorted(cands, key=lambda c: c.get("lowest_confidence_score", 0.0), reverse=True)
        return ranked[:k_eff]

    if strategy != "eig":
        raise ValueError(f"Unknown question selection strategy: {strategy}")

    # EIG * impact / cost with declared pitch-question budget slot
    eligible = [c for c in cands if float(c.get("eig_value_score", 0.0)) >= float(eig_floor)]
    if not eligible:
        return []

    b_cands = sorted(
        [c for c in eligible if c["factor"] == "boundary"],
        key=lambda c: c.get("eig_value_score", 0.0),
        reverse=True,
    )
    p_cands = sorted(
        [c for c in eligible if c["factor"] == "pitch"],
        key=lambda c: c.get("eig_value_score", 0.0),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    # Reserve up to pitch_budget slot for top pitch candidate when pitch candidates exist
    if p_cands and pitch_budget > 0 and k_eff >= 1:
        n_p = min(int(pitch_budget), len(p_cands), k_eff)
        selected.extend(p_cands[:n_p])
        p_cands = p_cands[n_p:]

    remaining_slots = k_eff - len(selected)
    if remaining_slots > 0:
        rest = sorted(b_cands + p_cands, key=lambda c: c.get("eig_value_score", 0.0), reverse=True)
        selected.extend(rest[:remaining_slots])

    selected.sort(key=lambda c: c.get("start_s", 0.0))
    return selected


def render_question_audio_clips(
    questions: Sequence[Dict[str, Any]],
    x: np.ndarray,
    session_dir: Path,
    sr: int = E.SR,
) -> List[Dict[str, Any]]:
    """Render 2-3 s original, option A, and option B WAV clips for each card."""
    sdir = Path(session_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    dur_total = len(x) / float(sr)
    out = []

    for q in questions:
        qid = q["qid"]
        t0 = max(0.0, float(q["start_s"]) - 0.35)
        t1 = min(dur_total, float(q["end_s"]) + 0.35)
        if t1 - t0 < 1.2:
            pad = 0.5 * (1.2 - (t1 - t0))
            t0 = max(0.0, t0 - pad)
            t1 = min(dur_total, t1 + pad)

        s0 = max(0, int(round(t0 * sr)))
        s1 = min(len(x), int(round(t1 * sr)))
        orig_seg = x[s0:s1].astype(np.float32)
        seg_dur = max(0.2, len(orig_seg) / float(sr))

        orig_name = f"q_{qid}_orig.wav"
        opt_a_name = f"q_{qid}_optA.wav"
        opt_b_name = f"q_{qid}_optB.wav"

        sf.write(sdir / orig_name, E.normalize(orig_seg) if len(orig_seg) else orig_seg, sr, subtype="PCM_16")

        if q["factor"] == "boundary":
            tb_rel = max(0.05, min(seg_dur - 0.05, float(q["boundary_s"]) - t0))
            n_start = max(0.0, float(q["start_s"]) - t0)
            n_end = min(seg_dur, float(q["end_s"]) - t0)
            merged_notes = [{
                "pitch_midi": int(q.get("midi_merged", 60)),
                "onset_s": n_start,
                "duration_s": max(0.08, n_end - n_start),
            }]
            split_notes = [
                {
                    "pitch_midi": int(q.get("midi_left", 60)),
                    "onset_s": n_start,
                    "duration_s": max(0.06, tb_rel - n_start - 0.02),
                },
                {
                    "pitch_midi": int(q.get("midi_right", 60)),
                    "onset_s": tb_rel,
                    "duration_s": max(0.06, n_end - tb_rel),
                },
            ]
            wav_merged = synth.render_notes(merged_notes, seg_dur, sr=sr)
            wav_split = synth.render_notes(split_notes, seg_dur, sr=sr)
            if q["option_a"]["action"] == "merge":
                wav_a, wav_b = wav_merged, wav_split
            else:
                wav_a, wav_b = wav_split, wav_merged
        else:
            # Pitch card: Option A = original ("As sung"), Option B = shifted ("Centred")
            wav_a = E.normalize(orig_seg)
            rel_a = max(0.0, float(q["start_s"]) - t0)
            rel_b = min(seg_dur, float(q["end_s"]) - t0)
            shift_c = float(q.get("recommended_shift_cents", 0.0))
            wav_b = span_repair._apply_span_pitch_shift(
                wav_a, sr=sr, start_s=rel_a, end_s=rel_b, shift_cents=shift_c, fade_s=0.025
            )

        sf.write(sdir / opt_a_name, wav_a, sr, subtype="PCM_16")
        sf.write(sdir / opt_b_name, wav_b, sr, subtype="PCM_16")

        q_copy = dict(q)
        q_copy["clips"] = {
            "orig_file": orig_name,
            "opt_a_file": opt_a_name,
            "opt_b_file": opt_b_name,
            "window_start_s": round(t0, 3),
            "window_end_s": round(t1, 3),
        }
        out.append(q_copy)

    return out


def compute_intent_and_trust(
    span_diagnostics: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    answers_by_qid: Optional[Dict[str, str]] = None,
    duration_s: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build per-span intent_map and calibrated trust_curve (both marked proposal)."""
    answers_by_qid = answers_by_qid or {}
    spans = span_diagnostics.get("spans", [])

    # Map candidates to spans by time overlap or span_id
    cand_by_span: Dict[int, List[Dict[str, Any]]] = {}
    for c in candidates:
        if "span_id" in c:
            cand_by_span.setdefault(int(c["span_id"]), []).append(c)
        else:
            tb = float(c.get("boundary_s", 0.0))
            for sp in spans:
                if float(sp["start_s"]) - 0.15 <= tb <= float(sp["end_s"]) + 0.15:
                    cand_by_span.setdefault(int(sp["span_id"]), []).append(c)

    intent_map: List[Dict[str, Any]] = []
    for sp in spans:
        sid = int(sp["span_id"])
        linked = cand_by_span.get(sid, [])
        p_split = 0.5
        p_merge = 0.5
        p_inflection = 0.85 if not sp.get("review_repair_candidate") else 0.5
        state = "confirmed_as_sung" if not sp.get("review_repair_candidate") else "unresolved"

        for c in linked:
            ans = answers_by_qid.get(c["qid"])
            if c["factor"] == "boundary":
                p_split = float(c.get("p_split", 0.5))
                p_merge = float(c.get("p_merge", 0.5))
                if ans in ("optA", "optB"):
                    chosen = c["option_a"] if ans == "optA" else c["option_b"]
                    state = "confirmed_boundary"
                    if chosen["action"] == "split":
                        p_split, p_merge = 0.90, 0.10
                    else:
                        p_split, p_merge = 0.10, 0.90
                elif ans == "skip":
                    state = "unresolved"
            elif c["factor"] == "pitch":
                p_inflection = float(c.get("p_inflection", 0.5))
                if ans in ("optA", "optB"):
                    chosen = c["option_a"] if ans == "optA" else c["option_b"]
                    if chosen["action"] == "preserve":
                        state = "confirmed_as_sung"
                        p_inflection = 0.90
                    else:
                        state = "confirmed_centred"
                        p_inflection = 0.10
                elif ans == "skip":
                    state = "unresolved"

        trust_span = (
            0.92 if state.startswith("confirmed_")
            else max(0.25, min(0.85, float(sp.get("mean_confidence", 0.7)) * (1.0 - 0.35 if linked else 1.0)))
        )

        intent_map.append({
            "span_id": sid,
            "t0": round(float(sp["start_s"]), 3),
            "t1": round(float(sp["end_s"]), 3),
            "note_name": sp.get("note_name", ""),
            "p_split": round(p_split, 4),
            "p_merge": round(p_merge, 4),
            "p_inflection": round(p_inflection, 4),
            "state": state,
            "span_trust": round(trust_span, 3),
            "status_tag": "proposal",
        })

    # Build 0.5s trust_curve across the recording
    dur = max(0.5, float(duration_s))
    times = np.arange(0.0, dur + 1e-6, 0.5)
    values = []
    for t_val in times:
        matching = [row["span_trust"] for row in intent_map if row["t0"] <= t_val <= row["t1"]]
        if matching:
            values.append(round(float(np.mean(matching)), 3))
        else:
            values.append(0.75)

    mean_trust = round(float(np.mean(values)), 3) if values else 0.75
    trust_curve = {
        "step_s": 0.5,
        "times_s": [round(float(v), 2) for v in times],
        "trust": values,
        "mean_trust": mean_trust,
        "status_tag": "proposal",
        "adoption_note": "Proposed downstream melody trust signal for Pasan/Rahul; optional until adopted.",
    }
    return intent_map, trust_curve


def apply_user_answers(
    candidates: Sequence[Dict[str, Any]],
    span_diagnostics: Dict[str, Any],
    answers: Sequence[Dict[str, Any]],
    duration_s: float,
    strategy: str = "eig",
) -> Dict[str, Any]:
    """Process submitted user answers and return updated intent_map, trust_curve, and repair targets."""
    answers_by_qid: Dict[str, str] = {}
    answer_log: List[Dict[str, Any]] = []
    for a in answers:
        qid = str(a.get("qid", ""))
        choice = str(a.get("choice", "skip"))
        if qid:
            answers_by_qid[qid] = choice
            answer_log.append({
                "qid": qid,
                "choice": choice,
                "timestamp_utc": a.get("timestamp_utc"),
            })

    intent_map, trust_curve = compute_intent_and_trust(
        span_diagnostics=span_diagnostics,
        candidates=candidates,
        answers_by_qid=answers_by_qid,
        duration_s=duration_s,
    )

    centred_span_ids: List[int] = []
    confirmed_boundaries: List[Dict[str, Any]] = []
    for c in candidates:
        choice = answers_by_qid.get(c["qid"])
        if choice not in ("optA", "optB"):
            continue
        chosen = c["option_a"] if choice == "optA" else c["option_b"]
        if c["factor"] == "pitch" and chosen["action"] == "centre":
            centred_span_ids.append(int(c["span_id"]))
        elif c["factor"] == "boundary":
            confirmed_boundaries.append({
                "boundary_s": c["boundary_s"],
                "action": chosen["action"],
                "start_s": c["start_s"],
                "end_s": c["end_s"],
            })

    return {
        "intent_map": intent_map,
        "trust_curve": trust_curve,
        "user_answers": {
            "strategy": strategy,
            "policy_hash": policy_hash(),
            "answers": answer_log,
        },
        "centred_span_ids": sorted(set(centred_span_ids)),
        "confirmed_boundaries": confirmed_boundaries,
        "needs_repair_review": len(centred_span_ids) > 0,
    }
