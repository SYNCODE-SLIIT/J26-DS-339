"""Phase 0 diagnostics for the Studio boundary model (studio_v1). Read-only: no training, no model or data edits.

Declared before running (model2_improvement_plan.md, Phase 0). Questions:
  Q1 On the 39 Vocadito windows (clean, 10 dB, 0 dB), what are precision/recall and the onset timing error of studio_v1?
  Q2 Which false positives sit inside a sustained/vibrato note, and which false negatives are same-pitch repeats?
  Q3 How much of the onset-F1 -> note-F1 gap comes from note formation (phantom pre-first-peak note, unvoiced gaps,
     boundary-glide frames in the median) rather than from boundary detection? (oracle-edge note F1 isolates this)
  Q4 How do 10 dB / 0 dB noise shift the 20 ms input features (confidence, voicing, energy range, flux, pitch steps)?
Also reports precision/recall/timing on the humming development clips (dev_M02) when the cached records load.
Output: results/studio/r1/v1_error_diagnostics.json. Run: ..\\.venv\\Scripts\\python.exe tools\\diagnose_boundary_errors.py
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WS / 'source'), str(WS)]

from bpdr_input.reporting.provenance import relpath, sha256_file
from bpdr_input.studio import boundary, clean, evidence as E
from tools.benchmark_studio_vocadito import (
    load_vocadito_windows, make_notes_from_aligned, note_f1, onset_f1,
)

NOISE = WS / 'datasets' / 'environmental_noise' / '1-100210-A-36.wav'
OUT = WS / 'results' / 'studio' / 'r1' / 'v1_error_diagnostics.json'
CONDS = (('clean', None), ('10dB', 10), ('0dB', 0))
TOL = 0.05


def mix(x, noise, snr):
    n = np.resize(noise, len(x))
    g = np.sqrt((x ** 2).mean() / ((n ** 2).mean() * 10 ** (snr / 10)))
    y = (x + g * n).astype(np.float32)
    return (y / (np.abs(y).max() / 0.9)).astype(np.float32)


def match(ref, est, tol=TOL):
    """One-to-one nearest matching. Returns list of (ref_idx, est_idx) pairs."""
    used, pairs = set(), []
    for i, r in enumerate(ref):
        cand = [k for k in range(len(est)) if k not in used and abs(est[k] - r) <= tol]
        if cand:
            k = min(cand, key=lambda k: abs(est[k] - r))
            used.add(k)
            pairs.append((i, k))
    return pairs


def notes_variants(edges, aligned, ref_notes):
    """Note F1 under several note-formation rules, same edges. Returns dict name -> list[(onset, midi)]."""
    t = np.asarray(aligned['t'], float)
    f0 = np.asarray(aligned['f0'], float)
    voiced = np.asarray(aligned['voiced'], bool)
    rel = voiced & np.isfinite(f0) & (f0 > 0)
    fine = np.zeros(len(t))
    if rel.any():
        fine[rel] = 1200.0 * np.log2(np.maximum(f0[rel], 1e-6) / 440.0)
    edges = np.sort(np.asarray(edges, float))

    def form(edge_list, drop_first=False, split_gaps=False, trim=False):
        seg = np.digitize(t, edge_list)
        out = []
        for s in np.unique(seg):
            if drop_first and s == 0 and len(edge_list) > 0:
                continue
            idx = np.flatnonzero((seg == s) & rel)
            groups = [idx]
            if split_gaps and len(idx) > 1:
                cut = np.flatnonzero(np.diff(t[idx]) > 0.08) + 1
                groups = np.split(idx, cut)
            for g in groups:
                if len(g) < 4:
                    continue
                core = g[1:-1] if (trim and len(g) >= 6) else g
                out.append((float(t[g[0]]), int(round(69 + np.median(fine[core]) / 100))))
        return out

    ref_edges = np.array([o for o, _ in ref_notes], float)
    return {
        'legacy': form(edges),
        'drop_phantom_first': form(edges, drop_first=True),
        'refined_gap_trim': form(edges, split_gaps=True, trim=True),
        'oracle_edges_legacy_pitch': form(ref_edges),
        'oracle_edges_refined': form(ref_edges, split_gaps=True, trim=True),
    }


def diagnose_window(w, model, noise, cond, snr):
    x_full, _ = E.load_mono(w['wav_path'], max_s=None)
    s0 = max(0, int(round(w['offset_s'] * E.SR)))
    s1 = min(len(x_full), s0 + int(round(w['duration_s'] * E.SR)))
    x = x_full[s0:s1]
    if snr is not None:
        x = mix(x, noise, snr)
    xn = E.normalize(x)
    trk_o = E.pesto_track(xn)
    cleaned = clean.spectral_gate(x)
    trk_c = E.pesto_track(cleaned)
    use_clean, _ = clean.choose(trk_o, trk_c)
    prepared = cleaned if use_clean else xn
    trk_p = trk_c if use_clean else trk_o
    aligned = boundary.align_track_to_20ms(trk_p, sr=E.SR)
    hints = boundary.extract_boundary_hints(trk_p, x=prepared, sr=E.SR, model=model)
    peaks = np.asarray(hints['peaks_s'], float)

    feats = boundary.extract_boundary_features(aligned, x=prepared, sr=E.SR, nin=5)
    ok = feats[:, 1] > 0.5
    conf, energy, flux, pitch = feats[:, 2], feats[:, 3], feats[:, 4], feats[:, 0]
    dp = np.abs(np.diff(pitch)) * (ok[1:] & ok[:-1])
    feat_stats = {
        'voiced_frac': float(ok.mean()),
        'mean_conf_voiced': float(conf[ok].mean()) if ok.any() else 0.0,
        'energy_range_p90_p10': float(np.percentile(energy, 90) - np.percentile(energy, 10)),
        'mean_flux': float(flux.mean()),
        'mean_abs_pitch_step_semitones': float(dp.mean()),
        'frac_pitch_step_gt_0p5st': float((dp > 0.5).mean()),
        'n_peaks': int(len(peaks)),
        'cleaned_chosen': bool(use_clean),
    }

    # per-annotator diagnostics against A1 (note F1 variants average A1/A2)
    refs = w['refs']
    a1 = refs.get('A1', [])
    ref_on = np.array([o for o, _ in a1], float)
    ref_midi = np.array([m for _, m in a1], float)
    pairs = match(ref_on, peaks)
    matched_ref = {i for i, _ in pairs}
    matched_est = {k for _, k in pairs}
    timing_ms = [float((peaks[k] - ref_on[i]) * 1000) for i, k in pairs]

    # FP classification
    t = np.asarray(aligned['t'], float)
    f0 = np.asarray(aligned['f0'], float)
    okt = np.asarray(aligned['voiced'], bool) & np.isfinite(f0) & (f0 > 0)
    cents = np.where(okt, 1200 * np.log2(np.maximum(f0, 1e-6) / 440.0), np.nan)
    fp = {'within_note_vibrato_like': 0, 'within_note_steady': 0, 'unvoiced_region': 0, 'near_ref_but_late_or_early': 0}
    for k in range(len(peaks)):
        if k in matched_est:
            continue
        i = int(np.argmin(np.abs(t - peaks[k])))
        if not okt[i]:
            fp['unvoiced_region'] += 1
            continue
        if len(ref_on) and np.min(np.abs(ref_on - peaks[k])) <= 0.15:
            fp['near_ref_but_late_or_early'] += 1
            continue
        lo, hi = max(0, i - 5), min(len(t), i + 6)
        seg = cents[lo:hi]
        std = float(np.nanstd(seg)) if np.isfinite(seg).sum() >= 3 else 0.0
        fp['within_note_vibrato_like' if std > 30 else 'within_note_steady'] += 1

    # FN classification
    fn = {'same_pitch_repeat': 0, 'pitch_step': 0, 'first_note': 0}
    for i in range(len(ref_on)):
        if i in matched_ref:
            continue
        if i == 0:
            fn['first_note'] += 1
        elif abs(ref_midi[i] - ref_midi[i - 1]) < 0.5:
            fn['same_pitch_repeat'] += 1
        else:
            fn['pitch_step'] += 1
    n_same = int(sum(1 for i in range(1, len(ref_on)) if abs(ref_midi[i] - ref_midi[i - 1]) < 0.5))

    # gap audit: note F1 averaged over annotators for each formation rule
    variants = {}
    for ann_refs in [r for r in refs.values() if r]:
        for name, notes in notes_variants(peaks, aligned, ann_refs).items():
            variants.setdefault(name, []).append(note_f1(notes, ann_refs))
    variants = {k: float(np.mean(v)) for k, v in variants.items()}

    # pitch-check failures among onset-matched notes (legacy formation)
    legacy = make_notes_from_aligned(peaks, aligned)
    pitch_fail = 0
    onset_hits = 0
    for ro, rm in a1:
        c = [(eo, em) for eo, em in legacy if abs(eo - ro) <= TOL]
        if c:
            onset_hits += 1
            if min(abs((em - rm) * 100) for _, em in c) > 50:
                pitch_fail += 1

    return {
        'id': w['id'], 'person': w['person'], 'cond': cond,
        'n_ref': len(ref_on), 'n_est': len(peaks), 'n_match': len(pairs),
        'precision': len(pairs) / len(peaks) if len(peaks) else 0.0,
        'recall': len(pairs) / len(ref_on) if len(ref_on) else 0.0,
        'onset_f1_a1': onset_f1(ref_on, peaks),
        'timing_ms': timing_ms, 'fp': fp, 'fn': fn, 'n_same_pitch_refs': n_same,
        'note_f1_variants': variants,
        'onset_hits_legacy': onset_hits, 'pitch_check_fail_among_onset_hits': pitch_fail,
        'features': feat_stats,
    }


def summarize(rows):
    def mean(key):
        return float(np.mean([r[key] for r in rows]))
    n_est = sum(r['n_est'] for r in rows)
    n_ref = sum(r['n_ref'] for r in rows)
    n_match = sum(r['n_match'] for r in rows)
    tm = np.array([v for r in rows for v in r['timing_ms']], float)
    fp = {k: sum(r['fp'][k] for r in rows) for k in rows[0]['fp']}
    fn = {k: sum(r['fn'][k] for r in rows) for k in rows[0]['fn']}
    nv = {k: float(np.mean([r['note_f1_variants'][k] for r in rows])) for k in rows[0]['note_f1_variants']}
    hits = sum(r['onset_hits_legacy'] for r in rows)
    fails = sum(r['pitch_check_fail_among_onset_hits'] for r in rows)
    feat = {k: float(np.mean([r['features'][k] for r in rows])) for k in rows[0]['features']}
    return {
        'windows': len(rows), 'pooled_precision': n_match / max(n_est, 1), 'pooled_recall': n_match / max(n_ref, 1),
        'mean_onset_f1_a1': mean('onset_f1_a1'),
        'timing_ms': {'n': int(len(tm)), 'median_signed': float(np.median(tm)) if len(tm) else None,
                      'median_abs': float(np.median(np.abs(tm))) if len(tm) else None,
                      'p90_abs': float(np.percentile(np.abs(tm), 90)) if len(tm) else None,
                      'frac_within_20ms': float((np.abs(tm) <= 20).mean()) if len(tm) else None},
        'false_positives': fp, 'false_negatives': fn,
        'ref_notes_same_pitch_as_previous': sum(r['n_same_pitch_refs'] for r in rows),
        'note_f1_by_formation_rule': nv,
        'onset_matched_but_pitch_check_failed': {'matched': hits, 'failed': fails, 'frac': fails / max(hits, 1)},
        'feature_means': feat,
    }


def dev_m02(model):
    """Precision/recall/timing on cached humming development clips (training-parity features). Best effort."""
    try:
        sys.path.insert(0, str(WS / 'tools'))
        import fit_r1_boundary as B
        recs = B.load()
        dev = [r for r in recs if r['role'] == 'development']
        n_est = n_ref = n_match = 0
        tm = []
        for r in dev:
            p = B.probs(model, r, False, r['edges'])
            pk = B.picks(p, r, model.default_threshold)
            pairs = match(np.sort(r['onsets']), np.sort(pk))
            n_est += len(pk); n_ref += len(r['onsets']); n_match += len(pairs)
            on, pks = np.sort(r['onsets']), np.sort(pk)
            tm += [float((pks[k] - on[i]) * 1000) for i, k in pairs]
        tm = np.array(tm)
        return {'clips': len(dev), 'precision': n_match / max(n_est, 1), 'recall': n_match / max(n_ref, 1),
                'timing_ms_median_abs': float(np.median(np.abs(tm))) if len(tm) else None,
                'timing_ms_median_signed': float(np.median(tm)) if len(tm) else None}
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return {'error': f'{type(exc).__name__}: {exc}'}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--output', default=str(OUT))
    args = ap.parse_args(argv)
    noise, _ = librosa.load(NOISE, sr=E.SR)
    model = boundary.load_boundary_model(boundary.V1_CHECKPOINT)
    windows = load_vocadito_windows()
    if args.limit:
        windows = windows[: args.limit]
    per_cond = {}
    for cond, snr in CONDS:
        rows = []
        for n, w in enumerate(windows, 1):
            r = diagnose_window(w, model, noise, cond, snr)
            rows.append(r)
            print(f"[{cond} {n}/{len(windows)}] {w['id']} P={r['precision']:.2f} R={r['recall']:.2f} "
                  f"v1note={r['note_f1_variants']['legacy']:.3f}", flush=True)
        per_cond[cond] = {'summary': summarize(rows), 'rows': rows}
    payload = {
        'model': 'studio_v1', 'checkpoint': relpath(boundary.V1_CHECKPOINT, WS), 'checkpoint_sha256': sha256_file(boundary.V1_CHECKPOINT), 'threshold': float(model.default_threshold),
        'noise_file': relpath(NOISE, WS), 'tolerance_s': TOL, 'n_windows': len(windows),
        'vocadito': {c: v['summary'] for c, v in per_cond.items()},
        'dev_M02': dev_m02(model),
        'rows': {c: v['rows'] for c, v in per_cond.items()},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=1, default=float), encoding='utf-8')
    print('\nWrote', args.output)
    for c, v in payload['vocadito'].items():
        print(f"\n== {c} ==")
        print(json.dumps({k: v[k] for k in ('pooled_precision', 'pooled_recall', 'timing_ms', 'false_positives',
                                           'false_negatives', 'note_f1_by_formation_rule',
                                           'onset_matched_but_pitch_check_failed')}, indent=1))
    print('\ndev_M02', json.dumps(payload['dev_M02']))


if __name__ == '__main__':
    main()
