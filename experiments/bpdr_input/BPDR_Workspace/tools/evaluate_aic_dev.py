"""Declared simulated-singer development evaluation for Active Intent Confirmation (AIC).

Evaluates:
  - H-I1a: Boundary question budget curve (K=0..5) on 39 Vocadito training-role windows
           across 14 singers under oracle_A1 and noisy_A2 simulated users.
  - H-I1b: Pitch question budget curve (K=0..5) on 498 v64 development records
           (53 untouched originals + 445 detuned pairs, 20 groups).
  - H-I2: Leave-singer-out calibration (ECE & Brier score before vs after).
  - H-I3: Do-no-harm verification on 'as_sung' spans.
  - Section 11 Confound Ablation: 4-way decomposition of the +0.145 Note F1 gain
    (pesto_notes_from_track vs gap-only median vs extractor-boundary median vs studio_v1).

Data discipline: Reserved HumTrans M03-M05, VALID, and TEST splits are untouched.
"""
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks

WS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WS / "source"), str(WS)]

from bpdr_input.studio import boundary, clean, evidence as E, question_engine as qe
from tools.analyze_r1_e2e import N_RESAMPLES, RNG_SEED, bootstrap_ci, clips_to_singer_means, singer_metric_ci
from tools.benchmark_studio_vocadito import load_vocadito_windows, note_f1, notes_dicts_to_pairs, onset_f1

OUT_DIR = WS / "results" / "studio" / "aic"
V64_DIR = WS / "results" / "transformer_usability" / "2026-11-04_v64_reliable_melodic_pairs"
K_GRID = [0, 1, 2, 3, 4, 5]
STRATEGIES = ["none", "random", "lowest_confidence", "eig"]
PAIRED_K = [3, 5]
# Simulated noisy listener: probability that a pitch answer is wrong (matches qe.LISTENER_FLIP_EPS).
ANSWER_FLIP_P = qe.LISTENER_FLIP_EPS
# Residual pitch error after a rendered repair (cents), taken from the declared bpdr clean benchmark
# (results/studio/r1/bpdr_repair_benchmark.json, clean MAE 7.47 c). Replaces the earlier perfect-repair
# assumption; the perfect-repair figure is still reported for comparison.
REPAIR_RESIDUAL_CENTS = 7.47


def _paired_ci(a: List[Tuple[str, float]], b: List[Tuple[str, float]]) -> List[float]:
    """Paired unit-level bootstrap CI of mean(a) - mean(b) (units = singers or groups)."""
    ga, gb = _group_means(a), _group_means(b)
    common = sorted(set(ga) & set(gb))
    return [round(v, 4) for v in bootstrap_ci({u: ga[u] - gb[u] for u in common})]


def _fit_temp_bias(logits: np.ndarray, y: np.ndarray, x0, bounds) -> Tuple[float, float]:
    def nll(params):
        t_scale, bias = params
        z = np.clip((logits - bias) / max(0.25, t_scale), -20.0, 20.0)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1.0 - 1e-6)
        return -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize(nll, x0=x0, bounds=bounds, method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def _logit_arr(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.02, 0.98)
    return np.log(p / (1.0 - p))


def _group_means(tuples: List[Tuple[str, float]]) -> Dict[str, float]:
    by: Dict[str, List[float]] = {}
    for s, v in tuples:
        by.setdefault(s, []).append(float(v))
    return {s: float(np.mean(vs)) for s, vs in by.items()}


def _tuples_ci(tuples: List[Tuple[str, float]]) -> Tuple[float, float, float]:
    return singer_metric_ci(_group_means(tuples))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_declaration() -> Tuple[Path, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decl = {
        "experiment": "active_intent_confirmation_dev_evaluation",
        "date": "2026-11-23",
        "revision": "v0.2: measured harm (noisy user + auto-repair comparator), leave-one-out selection, paired CIs, recall ceilings, realistic repair residual",
        "data_scope": {
            "vocadito_dev": "39 training-role Vocadito windows across 14 singers (A1 oracle, A2 noisy user)",
            "v64_dev": "498 development records (53 originals + 445 detuned pairs, 20 groups)",
            "reserved_untouched": ["HumTrans M03", "HumTrans M04", "HumTrans M05", "VALID", "TEST"],
        },
        "hypotheses": ["H-I1a", "H-I1b", "H-I2", "H-I3", "Section11_NoteRule_Ablation"],
        "k_grid": K_GRID,
        "strategies": STRATEGIES,
        "metrics": {
            "H-I1a": ["onset_f1_50ms", "note_f1_50ms_50c"],
            "H-I1b": ["mean_abs_correction_error_cents", "harmful_edit_rate_pct"],
            "H-I2": ["ece", "brier_score"],
            "H-I3": ["harmful_edit_rate_pct_noisy_user", "auto_repair_comparator_pct"],
            "paired": ["eig_minus_lowest_confidence", "eig_minus_random", "eig_minus_none"],
            "ceilings": ["recall_ceiling_boundary", "recall_ceiling_pitch"],
        },
        "seed": RNG_SEED,
        "bootstrap_resamples": N_RESAMPLES,
        "win_condition": "Report curves with singer-level 95% CIs honestly, including any null results.",
    }
    decl_path = OUT_DIR / "declaration.json"
    raw = json.dumps(decl, indent=2).encode("utf-8")
    decl_path.write_bytes(raw)
    return decl_path, _sha256_bytes(raw)


def _compute_ece_brier(probs: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> Dict[str, Any]:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    labels = np.asarray(labels, dtype=float)
    brier = float(np.mean((probs - labels) ** 2)) if len(probs) else 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        if m.any():
            conf = float(np.mean(probs[m]))
            acc = float(np.mean(labels[m]))
            ece += (float(m.sum()) / len(probs)) * abs(acc - conf)
            bin_rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": int(m.sum()), "mean_pred": round(conf, 4), "empirical_freq": round(acc, 4)})
        else:
            bin_rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0, "mean_pred": None, "empirical_freq": None})
    return {"ece": round(float(ece), 4), "brier": round(float(brier), 4), "bins": bin_rows}


def _apply_boundary_answers(base_peaks: List[float], selected_qs: List[Dict[str, Any]], user_onsets: np.ndarray) -> List[float]:
    peaks = sorted(float(p) for p in base_peaks)
    for q in selected_qs:
        tb = float(q["boundary_s"])
        is_true_split = bool(len(user_onsets) > 0 and np.min(np.abs(user_onsets - tb)) <= 0.060)
        if is_true_split:
            if not any(abs(p - tb) <= 0.045 for p in peaks):
                peaks.append(tb)
                peaks.sort()
        else:
            peaks = [p for p in peaks if abs(p - tb) > 0.060]
    return peaks


def evaluate_vocadito_boundary_and_ablation() -> Dict[str, Any]:
    windows = load_vocadito_windows()
    prepared_windows = []

    # 4-way note-formation rule ablation accumulators
    ablation_raw: Dict[str, List[Tuple[str, float, float]]] = {
        "pesto_notes_from_track": [],
        "segment_median_gap_only": [],
        "segment_median_extractor_boundaries": [],
        "studio_v1_boundary_hints_plus_median_rule": [],
    }

    b_cal_records = []

    for w in windows:
        x, _ = E.load_mono(w["wav_path"], sr=E.SR, max_s=None)
        s0 = int(round(w["offset_s"] * E.SR))
        s1 = s0 + int(round(w["duration_s"] * E.SR))
        win_x = x[s0:s1]
        dur = len(win_x) / float(E.SR)
        xn = E.normalize(win_x)
        trk_o = E.pesto_track(xn)
        cleaned = clean.spectral_gate(win_x)
        trk_c = E.pesto_track(cleaned)
        use_clean, _ = clean.choose(trk_o, trk_c)
        prep_x = cleaned if use_clean else xn
        trk_p = trk_c if use_clean else trk_o

        b_hints = boundary.extract_boundary_hints(trk_p, x=prep_x, sr=E.SR)
        aligned = boundary.align_track_to_20ms(trk_p, sr=E.SR)
        prov_edges = boundary.provisional_boundary_edges(aligned, x=prep_x, sr=E.SR)

        a1_ref = w["refs"].get("A1", [])
        a2_ref = w["refs"].get("A2", a1_ref)
        a1_onsets = np.array([o for o, _ in a1_ref], dtype=float)
        a2_onsets = np.array([o for o, _ in a2_ref], dtype=float)

        # 4-way ablation against A1/A2 mean (matching benchmark_studio_vocadito.py)
        ablation_methods = {
            "pesto_notes_from_track": E.notes_from_track(trk_p, dur),
            "segment_median_gap_only": boundary.notes_from_track_with_boundary_hints(trk_p, {"peaks_s": []}, end_s=dur, sr=E.SR),
            "segment_median_extractor_boundaries": boundary.notes_from_track_with_boundary_hints(trk_p, {"peaks_s": prov_edges.tolist()}, end_s=dur, sr=E.SR),
            "studio_v1_boundary_hints_plus_median_rule": boundary.notes_from_track_with_boundary_hints(trk_p, b_hints, end_s=dur, sr=E.SR),
        }
        for mname, ndicts in ablation_methods.items():
            pairs = notes_dicts_to_pairs(ndicts)
            est_onsets = np.array([o for o, _ in pairs], dtype=float)
            nf1_vals = [note_f1(pairs, ref) for ref in w["refs"].values()]
            of1_vals = [onset_f1(np.array([o for o, _ in ref]), est_onsets) for ref in w["refs"].values()]
            ablation_raw[mname].append((w["person"], float(np.mean(nf1_vals)), float(np.mean(of1_vals))))

        # Extract boundary ambiguity candidates
        cands = qe.extract_ambiguity_candidates(
            trk_p, x=prep_x, sr=E.SR, boundary_hints=b_hints,
            span_diagnostics={"spans": []}, duration_s=dur,
        )
        b_cands = [c for c in cands if c["factor"] == "boundary"]
        b_cands = qe.calibrate_candidate_probabilities(b_cands)

        for c in b_cands:
            tb = float(c["boundary_s"])
            y_true = 1.0 if (len(a1_onsets) > 0 and np.min(np.abs(a1_onsets - tb)) <= 0.060) else 0.0
            b_cal_records.append({
                "singer": w["person"],
                "raw_p": float(c["raw_p_model"]),
                "y": y_true,
            })

        prepared_windows.append({
            "singer": w["person"],
            "dur": dur,
            "trk_p": trk_p,
            "base_peaks": list(b_hints.get("peaks_s", [])),
            "b_cands": b_cands,
            "a1_ref": a1_ref,
            "a1_onsets": a1_onsets,
            "a2_onsets": a2_onsets,
        })

    # Summarize 4-way confound ablation
    ablation_summary = {}
    singer_nf1_by_method: Dict[str, Dict[str, float]] = {}
    for mname, rows in ablation_raw.items():
        nf1_tuples = [(s, nf) for s, nf, _ in rows]
        of1_tuples = [(s, of) for s, _, of in rows]
        ablation_summary[mname] = {
            "note_f1": [round(v, 4) for v in _tuples_ci(nf1_tuples)],
            "onset_f1": [round(v, 4) for v in _tuples_ci(of1_tuples)],
        }
        singer_nf1_by_method[mname] = _group_means(nf1_tuples)

    singers = sorted(singer_nf1_by_method["pesto_notes_from_track"].keys())
    ablation_diffs = {}
    for a, b in [
        ("segment_median_gap_only", "pesto_notes_from_track"),
        ("segment_median_extractor_boundaries", "segment_median_gap_only"),
        ("studio_v1_boundary_hints_plus_median_rule", "segment_median_extractor_boundaries"),
        ("studio_v1_boundary_hints_plus_median_rule", "pesto_notes_from_track"),
    ]:
        diffs = {s: singer_nf1_by_method[a][s] - singer_nf1_by_method[b][s] for s in singers}
        ablation_diffs[f"{a}_minus_{b}"] = [round(v, 4) for v in bootstrap_ci(diffs)]

    # Leave-singer-out calibration for boundary candidates (H-I2)
    raw_probs = np.array([r["raw_p"] for r in b_cal_records], dtype=float)
    y_labels = np.array([r["y"] for r in b_cal_records], dtype=float)
    cal_probs_loo = np.zeros_like(raw_probs)
    unique_singers = sorted({r["singer"] for r in b_cal_records})
    loo_params: Dict[str, Tuple[float, float]] = {}
    for held_s in unique_singers:
        tr_idx = [i for i, r in enumerate(b_cal_records) if r["singer"] != held_s]
        te_idx = [i for i, r in enumerate(b_cal_records) if r["singer"] == held_s]
        t_hat, b_hat = _fit_temp_bias(_logit_arr(raw_probs[tr_idx]), y_labels[tr_idx], [1.3, 0.0], [(0.3, 4.0), (-2.0, 2.0)])
        loo_params[held_s] = (t_hat, b_hat)
        z_te = np.clip((_logit_arr(raw_probs[te_idx]) - b_hat) / max(0.25, t_hat), -20.0, 20.0)
        cal_probs_loo[te_idx] = 1.0 / (1.0 + np.exp(-z_te))

    boundary_calibration = {
        "n_candidates": len(b_cal_records),
        "raw": _compute_ece_brier(raw_probs, y_labels),
        "calibrated_loo": _compute_ece_brier(cal_probs_loo, y_labels),
    }

    # Selection for the headline curves uses the held-out singer's LOO calibration (not the default constants).
    # Default-constant selection is kept as a secondary curve for comparison.
    for pw in prepared_windows:
        t_hat, b_hat = loo_params.get(pw["singer"], (1.35, 0.0))
        pw["b_cands_loo"] = qe.calibrate_candidate_probabilities(
            pw["b_cands"], {"boundary_temp": t_hat, "boundary_bias": b_hat}
        )

    def _run_boundary(user_mode: str, strat: str, k: int, cand_key: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        nf1_t, of1_t = [], []
        for idx_w, pw in enumerate(prepared_windows):
            sel = qe.select_questions(pw[cand_key], k=k, max_k=max(5, k), strategy=strat, seed=RNG_SEED + idx_w)
            user_ons = pw["a1_onsets"] if user_mode == "oracle_A1" else pw["a2_onsets"]
            new_peaks = _apply_boundary_answers(pw["base_peaks"], sel, user_ons)
            ndicts = boundary.notes_from_track_with_boundary_hints(pw["trk_p"], {"peaks_s": new_peaks}, end_s=pw["dur"], sr=E.SR)
            pairs = notes_dicts_to_pairs(ndicts)
            est_ons = np.array([o for o, _ in pairs], dtype=float)
            nf1_t.append((pw["singer"], note_f1(pairs, pw["a1_ref"])))
            of1_t.append((pw["singer"], onset_f1(pw["a1_onsets"], est_ons)))
        return nf1_t, of1_t

    # H-I1a budget curve across K=0..5 and user models (oracle_A1, noisy_A2)
    curves: Dict[str, Any] = {}
    curves_default_selection: Dict[str, Any] = {}
    paired: Dict[str, Any] = {}
    raw_tuples: Dict[Tuple[str, str, int], List[Tuple[str, float]]] = {}
    for user_mode in ("oracle_A1", "noisy_A2"):
        curves[user_mode], curves_default_selection[user_mode] = {}, {}
        for strat in STRATEGIES:
            curves[user_mode][strat], curves_default_selection[user_mode][strat] = [], []
            for k in K_GRID:
                for cand_key, store in (("b_cands_loo", curves), ("b_cands", curves_default_selection)):
                    nf1_tuples, of1_tuples = _run_boundary(user_mode, strat, k, cand_key)
                    if cand_key == "b_cands_loo":
                        raw_tuples[(user_mode, strat, k)] = nf1_tuples
                    store[user_mode][strat].append({
                        "k": k,
                        "note_f1": [round(v, 4) for v in _tuples_ci(nf1_tuples)],
                        "onset_f1": [round(v, 4) for v in _tuples_ci(of1_tuples)],
                    })
        paired[user_mode] = {}
        for k in PAIRED_K:
            paired[user_mode][f"K={k}"] = {
                f"eig_minus_{other}": _paired_ci(raw_tuples[(user_mode, "eig", k)], raw_tuples[(user_mode, other, k)])
                for other in ("lowest_confidence", "random", "none")
            }

    # Recall ceilings: (a) every candidate asked and answered by the oracle, (b) exact A1 onsets as edges.
    ceil_all, ceil_exact = [], []
    for pw in prepared_windows:
        new_peaks = _apply_boundary_answers(pw["base_peaks"], pw["b_cands"], pw["a1_onsets"])
        pairs = notes_dicts_to_pairs(boundary.notes_from_track_with_boundary_hints(pw["trk_p"], {"peaks_s": new_peaks}, end_s=pw["dur"], sr=E.SR))
        ceil_all.append((pw["singer"], note_f1(pairs, pw["a1_ref"])))
        pairs_x = notes_dicts_to_pairs(boundary.notes_from_track_with_boundary_hints(pw["trk_p"], {"peaks_s": pw["a1_onsets"].tolist()}, end_s=pw["dur"], sr=E.SR))
        ceil_exact.append((pw["singer"], note_f1(pairs_x, pw["a1_ref"])))
    recall_ceiling = {
        "note_f1_all_candidates_answered_by_oracle": [round(v, 4) for v in _tuples_ci(ceil_all)],
        "note_f1_with_exact_reference_onsets_as_edges": [round(v, 4) for v in _tuples_ci(ceil_exact)],
        "mean_candidates_per_window": round(float(np.mean([len(pw["b_cands"]) for pw in prepared_windows])), 2),
        "note": "Upper bounds for boundary-only intervention given the extractor's candidates; shows how much of the gap questions could ever close.",
    }

    return {
        "n_windows": len(prepared_windows),
        "n_singers": len(singers),
        "ablation": {"methods": ablation_summary, "diffs": ablation_diffs},
        "boundary_calibration": boundary_calibration,
        "selection_calibration": "leave_one_singer_out",
        "curves": curves,
        "curves_default_selection": curves_default_selection,
        "paired_eig_comparisons_note_f1": paired,
        "recall_ceiling": recall_ceiling,
    }


def evaluate_v64_pitch_and_do_no_harm() -> Dict[str, Any]:
    records = []
    for fname, is_pair in [("train_originals.json", False), ("train_pairs.json", True)]:
        entries = json.loads((V64_DIR / fname).read_text(encoding="utf-8"))
        for r in entries:
            if "excluded" in r:
                continue
            npz_path = r["analysis_C"] if is_pair else r["analysis_A"]
            z = np.load(npz_path)
            raw = z["raw_views"][0]
            if len(z["fine"]) != len(raw):
                continue
            t = np.arange(len(raw), dtype=float) * 0.02
            ok = z["reliable"].astype(bool) & np.isfinite(z["fine"])
            f0 = np.where(ok, 440.0 * (2.0 ** (np.where(ok, z["fine"], 0.0) / 1200.0)), np.nan)
            conf = raw[:, 65].astype(float)
            pk, _ = find_peaks(raw[:, 70], height=0.5, distance=5)
            edges = t[pk].tolist()

            trk = {"t": t, "f0": f0, "voiced": ok, "conf": conf}
            dur = float(t[-1] + 0.02) if len(t) else 10.0
            span_diag = boundary.analyze_boundary_spans(trk, {"peaks_s": edges}, end_s=dur, sr=E.SR)
            cands = qe.extract_ambiguity_candidates(
                trk, sr=E.SR, boundary_hints={"times_s": t.tolist(), "probabilities": raw[:, 70].tolist(), "peaks_s": edges},
                span_diagnostics=span_diag, duration_s=dur,
            )
            p_cands = qe.calibrate_candidate_probabilities([c for c in cands if c["factor"] == "pitch"])

            sup = r.get("support_seconds") if is_pair else None
            inj_cents = abs(float(r.get("cents", 0.0))) if is_pair else 0.0

            # Annotate ground-truth intent per candidate
            for c in p_cands:
                if is_pair and sup is not None:
                    ov = max(0.0, min(float(c["end_s"]), float(sup[1])) - max(float(c["start_s"]), float(sup[0])))
                    frac = ov / max(1e-6, min(float(c["end_s"]) - float(c["start_s"]), float(sup[1]) - float(sup[0])))
                    is_err = frac >= 0.45
                else:
                    is_err = False
                c["gt_is_error"] = is_err
                c["gt_error_cents"] = inj_cents if is_err else 0.0

            records.append({
                "group": r["group"],
                "is_pair": is_pair,
                "gt_shift_cents": inj_cents,
                "p_cands": p_cands,
            })

    # Pitch H-I2 LOO calibration
    p_rows = [c for rec in records for c in rec["p_cands"]]
    p_groups = [rec["group"] for rec in records for _ in rec["p_cands"]]
    raw_p_err = np.array([c["p_error"] for c in p_rows], dtype=float)
    y_p_err = np.array([1.0 if c["gt_is_error"] else 0.0 for c in p_rows], dtype=float)
    cal_p_loo = np.zeros_like(raw_p_err)
    loo_pitch: Dict[str, Tuple[float, float]] = {}

    for held_g in sorted(set(p_groups)):
        tr = [i for i, g in enumerate(p_groups) if g != held_g]
        te = [i for i, g in enumerate(p_groups) if g == held_g]
        if not tr or not te:
            continue
        t_h, b_h = _fit_temp_bias(_logit_arr(raw_p_err[tr]), y_p_err[tr], [1.0, 0.0], [(0.3, 4.0), (-3.0, 3.0)])
        loo_pitch[held_g] = (t_h, b_h)
        z_te = np.clip((_logit_arr(raw_p_err[te]) - b_h) / max(0.25, t_h), -20.0, 20.0)
        cal_p_loo[te] = 1.0 / (1.0 + np.exp(-z_te))

    pitch_calibration = {
        "n_candidates": len(p_rows),
        "raw": _compute_ece_brier(raw_p_err, y_p_err),
        "calibrated_loo": _compute_ece_brier(cal_p_loo, y_p_err),
    }

    # Re-calibrate each record's candidates with the held-out group's LOO parameters (selection uses these).
    base_cal = qe.DEFAULT_CALIBRATION
    for rec in records:
        t_h, b_h = loo_pitch.get(rec["group"], (1.0, 0.0))
        tt = max(0.25, t_h)
        loo_cal = {
            "pitch_alpha0": (base_cal["pitch_alpha0"] - b_h) / tt,
            "pitch_alpha1": base_cal["pitch_alpha1"] / tt,
            "pitch_alpha2": base_cal["pitch_alpha2"] / tt,
            "pitch_alpha3": base_cal["pitch_alpha3"] / tt,
        }
        rec["p_cands_loo"] = qe.calibrate_candidate_probabilities(rec["p_cands"], loo_cal)

    def _flip(rec_idx: int, qid: str, strat: str, k: int) -> bool:
        key = f"{RNG_SEED}|{rec_idx}|{qid}|{strat}|{k}".encode("utf-8")
        seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
        return bool(np.random.default_rng(seed).random() < ANSWER_FLIP_P)

    def _run_pitch(strat: str, k: int, cand_key: str, user: str, residual: float):
        """Returns (err tuples over detuned pairs, harm tuples over untouched originals)."""
        err_t, harm_t = [], []
        for idx_r, rec in enumerate(records):
            sel = qe.select_questions(rec[cand_key], k=k, max_k=max(5, k), strategy=strat, seed=RNG_SEED + idx_r, pitch_budget=k)
            if rec["is_pair"]:
                fixed = False
                for c in sel:
                    wrong = user == "noisy" and _flip(idx_r, c["qid"], strat, k)
                    # A true error is repaired only when the answer is correct; a flipped answer leaves it.
                    if c["gt_is_error"] and not wrong:
                        fixed = True
                err_t.append((rec["group"], residual if fixed else rec["gt_shift_cents"]))
            else:
                # Untouched original: any pitch question wrongly answered "centre" triggers an edit that should not happen.
                edited = any(user == "noisy" and _flip(idx_r, c["qid"], strat, k) for c in sel if c["factor"] == "pitch")
                harm_t.append((rec["group"], 100.0 if edited else 0.0))
        return err_t, harm_t

    # H-I1b budget curve (K=0..5): headline = LOO selection, noisy user, realistic residual
    pitch_curves: Dict[str, Any] = {}
    pitch_curves_perfect_oracle: Dict[str, Any] = {}
    pitch_curves_default_selection: Dict[str, Any] = {}
    raw_pitch: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}
    for strat in STRATEGIES:
        pitch_curves[strat], pitch_curves_perfect_oracle[strat], pitch_curves_default_selection[strat] = [], [], []
        for k in K_GRID:
            err_t, harm_t = _run_pitch(strat, k, "p_cands_loo", "noisy", REPAIR_RESIDUAL_CENTS)
            raw_pitch[(strat, k)] = err_t
            e_perf, _ = _run_pitch(strat, k, "p_cands_loo", "oracle", 0.0)
            e_def, _ = _run_pitch(strat, k, "p_cands", "noisy", REPAIR_RESIDUAL_CENTS)
            pitch_curves[strat].append({
                "k": k,
                "pos_mae_cents": [round(v, 3) for v in _tuples_ci(err_t)],
                "harmful_edit_rate_pct": [round(v, 3) for v in _tuples_ci(harm_t)] if harm_t else [0.0, 0.0, 0.0],
            })
            pitch_curves_perfect_oracle[strat].append({"k": k, "pos_mae_cents": [round(v, 3) for v in _tuples_ci(e_perf)]})
            pitch_curves_default_selection[strat].append({"k": k, "pos_mae_cents": [round(v, 3) for v in _tuples_ci(e_def)]})

    paired_pitch = {
        f"K={k}": {
            f"eig_minus_{o}_mae_cents": _paired_ci(raw_pitch[("eig", k)], raw_pitch[(o, k)])
            for o in ("lowest_confidence", "random", "none")
        }
        for k in PAIRED_K
    }

    # Do-no-harm: measured. Compare AIC (K=3) against automatically repairing every candidate.
    _, harm_aic_noisy = _run_pitch("eig", 3, "p_cands_loo", "noisy", REPAIR_RESIDUAL_CENTS)
    _, harm_aic_oracle = _run_pitch("eig", 3, "p_cands_loo", "oracle", 0.0)
    auto_edit = [(rec["group"], 100.0 if rec["p_cands"] else 0.0) for rec in records if not rec["is_pair"]]
    n_orig = len([r for r in records if not r["is_pair"]])

    # Candidate recall ceiling for pitch
    pair_recs = [r for r in records if r["is_pair"]]
    has_err_cand = [(r["group"], 1.0 if any(c["gt_is_error"] for c in r["p_cands"]) else 0.0) for r in pair_recs]
    ceiling_mae = [
        (r["group"], REPAIR_RESIDUAL_CENTS if any(c["gt_is_error"] for c in r["p_cands"]) else r["gt_shift_cents"])
        for r in pair_recs
    ]

    return {
        "n_records": len(records),
        "n_groups": len({r["group"] for r in records}),
        "pitch_calibration": pitch_calibration,
        "selection_calibration": "leave_one_group_out",
        "answer_model": {
            "noisy_flip_probability": ANSWER_FLIP_P,
            "repair_residual_cents": REPAIR_RESIDUAL_CENTS,
            "residual_source": "bpdr clean MAE in results/studio/r1/bpdr_repair_benchmark.json (feature-level, not delivered-WAV)",
        },
        "curves": pitch_curves,
        "curves_perfect_repair_oracle_user": pitch_curves_perfect_oracle,
        "curves_default_selection": pitch_curves_default_selection,
        "paired_eig_comparisons": paired_pitch,
        "recall_ceiling": {
            "fraction_of_detuned_pairs_with_an_error_candidate": [round(v, 4) for v in _tuples_ci(has_err_cand)],
            "pos_mae_cents_if_every_candidate_answered_correctly": [round(v, 3) for v in _tuples_ci(ceiling_mae)],
            "uncorrected_pos_mae_cents": round(float(np.mean([r["gt_shift_cents"] for r in pair_recs])), 3),
        },
        "h_i3_do_no_harm": {
            "measured": True,
            "n_untouched_originals": n_orig,
            "definition": "Percent of untouched originals receiving at least one wrongly answered question that would trigger an edit (K=3, EIG, LOO selection).",
            "aic_oracle_user_pct": [round(v, 3) for v in _tuples_ci(harm_aic_oracle)],
            "aic_noisy_user_pct": [round(v, 3) for v in _tuples_ci(harm_aic_noisy)],
            "auto_repair_every_candidate_pct": [round(v, 3) for v in _tuples_ci(auto_edit)],
            "note": "Oracle user is zero by construction; the noisy-user and auto-repair rows are the informative measurements.",
        },
    }


def main():
    decl_path, decl_sha = write_declaration()
    print(f"Declaration written: {decl_path} (sha256={decl_sha[:12]}...)", flush=True)

    print("Running Vocadito H-I1a boundary curves & Section 11 ablation (39 windows)...", flush=True)
    voc_res = evaluate_vocadito_boundary_and_ablation()

    print("Running v64 H-I1b pitch curves & H-I2/H-I3 checks (498 records)...", flush=True)
    v64_res = evaluate_v64_pitch_and_do_no_harm()

    out = {
        "schema": "aic_dev_evaluation/0.2",
        "declaration_sha256": decl_sha,
        "k_grid": K_GRID,
        "strategies": STRATEGIES,
        "vocadito": voc_res,
        "v64_pitch": v64_res,
    }
    out_path = OUT_DIR / "aic_dev_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved AIC dev results to {out_path}", flush=True)
    print("Section 11 ablation:", json.dumps(voc_res["ablation"]["methods"], indent=2), flush=True)


if __name__ == "__main__":
    main()
