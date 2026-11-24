"""Benchmark real Studio end-to-end inference on the 39 Vocadito training-role WAV windows.

Evaluates:
  - pyin_4096 (raw window & prepared window)
  - pesto_notes_from_track (Studio prepared PESTO track grouped by nearest semitone)
  - Studio boundary models via boundary.extract_boundary_hints:
      * v1 (V1_CHECKPOINT, nin=5, thr=0.80)
      * v2 (V2_CHECKPOINT, nin=5, thr=0.65)
      * v2_bpdr (V2_BPDR_CHECKPOINT, nin=6, thr=0.60, provisional edges)
      * v4 (boundary_transformer_v4.pt, nin=5)
      * v4_bpdr (boundary_transformer_v4_bpdr.pt, nin=6, provisional edges)
  - Optional pre-parity ablation: unaligned 100 Hz PESTO + onset_strength + thr=0.50 on V1_CHECKPOINT

Saves per-window metrics and 14-singer singer-macro 95% bootstrap CIs (1,000 resamples)
to results/studio/r1/studio_vocadito_benchmark.json.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import librosa
from scipy.signal import find_peaks

WS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WS / 'source'), str(WS)]

from bpdr_input.reporting.provenance import relpath, sha256_file
from bpdr_input.studio import boundary, clean, evidence as E
from tools.analyze_r1_e2e import (
    N_RESAMPLES,
    RNG_SEED,
    bootstrap_ci,
    clips_to_singer_means,
    singer_metric_ci,
)

TRAIN_ORIGINALS = WS / 'results' / 'transformer_usability' / '2026-11-04_v64_reliable_melodic_pairs' / 'train_originals.json'
VOC_AUDIO_DIR = WS / 'datasets' / 'vocadito' / 'Audio'
VOC_NOTES_DIR = WS / 'datasets' / 'vocadito' / 'Annotations' / 'Notes'
R1_DIR = WS / 'results' / 'studio' / 'r1'
DEFAULT_OUT = R1_DIR / 'studio_vocadito_benchmark.json'


def onset_f1(ref, est, tol=0.05):
    """One-to-one nearest onset F1 within tol seconds (matches fit_r1_boundary.f1)."""
    ref, est = np.sort(ref), np.sort(est)
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    used, hit = set(), 0
    for r in ref:
        cand = [k for k in range(len(est)) if k not in used and abs(est[k] - r) <= tol]
        if cand:
            used.add(min(cand, key=lambda k: abs(est[k] - r)))
            hit += 1
    if hit == 0:
        return 0.0
    p, rc = hit / len(est), hit / len(ref)
    return float(2 * p * rc / (p + rc))


def note_f1(est, ref):
    """One-to-one nearest note F1 within 50 ms onset and 50 cents pitch (matches fit_r1_e2e.note_f1)."""
    if len(est) == 0 or len(ref) == 0:
        return 0.0
    used, hit = set(), 0
    for ro, rm in ref:
        cand = [
            k for k, (eo, em) in enumerate(est)
            if k not in used and abs(eo - ro) <= 0.05 and abs((em - rm) * 100.0) <= 50.0
        ]
        if cand:
            used.add(min(cand, key=lambda k: abs(est[k][0] - ro)))
            hit += 1
    if hit == 0:
        return 0.0
    p, rc = hit / len(est), hit / len(ref)
    return float(2 * p * rc / (p + rc))


def make_notes_from_aligned(edges, aligned):
    """Segment aligned 20 ms track at boundary edges into (onset_s, midi_pitch) pairs."""
    t = np.asarray(aligned['t'], dtype=float)
    f0 = np.asarray(aligned['f0'], dtype=float)
    voiced = np.asarray(aligned['voiced'], dtype=bool)
    rel = voiced & np.isfinite(f0) & (f0 > 0)
    fine = np.zeros(len(t), dtype=float)
    if rel.any():
        fine[rel] = 1200.0 * np.log2(np.maximum(f0[rel], 1e-6) / 440.0)
    seg = np.digitize(t, np.sort(np.asarray(edges, dtype=float)))
    out = []
    for s in np.unique(seg):
        idx = np.flatnonzero((seg == s) & rel)
        if len(idx) >= 4:
            out.append((float(t[idx[0]]), int(round(69.0 + float(np.median(fine[idx])) / 100.0))))
    return out


def notes_dicts_to_pairs(note_dicts):
    """Convert evidence.notes_from_track dicts to (onset_s, midi_pitch) pairs."""
    return [(float(n['onset_s']), int(n['pitch_midi'])) for n in note_dicts]


def load_vocadito_windows():
    """Load the 39 Vocadito training-role windows and their A1/A2 note references."""
    entries = json.loads(TRAIN_ORIGINALS.read_text(encoding='utf-8'))
    windows = []
    for r in entries:
        if 'excluded' in r or not str(r.get('source_id', '')).startswith('vocadito'):
            continue
        parent = r['parent_source_id']
        wav_path = VOC_AUDIO_DIR / f'{parent}.wav'
        if not wav_path.exists():
            raise FileNotFoundError(f'Missing Vocadito WAV: {wav_path}')
        off = float(r['crop_offset_seconds'])
        dur = float(r.get('duration_seconds', 10.0))
        refs = {}
        for ann in ('A1', 'A2'):
            ann_path = VOC_NOTES_DIR / f'{parent}_notes{ann}.csv'
            if ann_path.exists():
                m = np.loadtxt(ann_path, delimiter=',', ndmin=2)
                on = m[:, 0] - off
                mask = (on >= 0.0) & (on < dur)
                refs[ann] = [(float(o), float(69.0 + 12.0 * np.log2(h / 440.0))) for o, h in zip(on[mask], m[mask, 1])]
        windows.append({
            'id': r['source_id'],
            'person': r['group'],
            'parent_source_id': parent,
            'wav_path': str(wav_path),
            'offset_s': off,
            'duration_s': dur,
            'refs': refs,
        })
    return windows


def pre_parity_v1_peaks(trk, x, sr=E.SR, model=None):
    """Simulate legacy unaligned 100 Hz boundary inference + onset_strength + thr=0.50 (read-only ablation)."""
    if model is None:
        model = boundary.load_boundary_model(boundary.V1_CHECKPOINT)
    t = np.asarray(trk['t'], dtype=float)
    f0 = np.asarray(trk['f0'], dtype=np.float32)
    voiced = np.asarray(trk['voiced'], dtype=bool)
    conf = np.asarray(trk['conf'], dtype=np.float32)
    ok = voiced & np.isfinite(f0) & (f0 > 0)
    N = len(t)
    pitch = np.zeros(N, dtype=np.float32)
    if ok.sum() > 0:
        fine = 1200.0 * np.log2(np.maximum(f0[ok], 1e-6) / 440.0)
        pitch[ok] = (fine - np.median(fine)) / 100.0
    hop = max(1, int(round(sr * float(np.median(np.diff(t)))))) if N > 1 else 220
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=hop)[0]
    log_e = 20.0 * np.log10(np.maximum(rms, 1e-8))
    t_e = np.arange(len(log_e)) * hop / float(sr)
    e_interp = np.interp(t, t_e, log_e)
    energy = ((e_interp - e_interp.mean()) / (e_interp.std() + 1e-6)).astype(np.float32)
    oenv = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop)
    t_o = np.arange(len(oenv)) * hop / float(sr)
    o_interp = np.interp(t, t_o, oenv)
    omax = float(o_interp.max())
    onset = (o_interp / omax if omax > 1e-8 else o_interp).astype(np.float32)
    feats = np.stack([pitch, ok.astype(np.float32), conf, energy, onset], axis=-1)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(feats[None, ...]).float()))[0].cpu().numpy()
    pk_idx, _ = find_peaks(probs, height=0.50, distance=5)
    return t[pk_idx].tolist() if len(pk_idx) > 0 else []


def candidate_checkpoints():
    """Return ordered list of (method_name, Path) for all available Studio checkpoints."""
    cands = [
        ('studio_v1', boundary.V1_CHECKPOINT),
        ('studio_v2', boundary.V2_CHECKPOINT),
    ]
    v2_bpdr = getattr(boundary, 'V2_BPDR_CHECKPOINT', R1_DIR / 'boundary_transformer_v2_bpdr.pt')
    cands.append(('studio_v2_bpdr', Path(v2_bpdr)))
    v4 = R1_DIR / 'boundary_transformer_v4.pt'
    v4_bpdr = R1_DIR / 'boundary_transformer_v4_bpdr.pt'
    cands.append(('studio_v4', v4))
    cands.append(('studio_v4_bpdr', v4_bpdr))
    v5 = getattr(boundary, 'V5_CHECKPOINT', R1_DIR / 'boundary_transformer_v5.pt')
    cands.append(('studio_v5', Path(v5)))
    return [(name, p) for name, p in cands if p.exists()]


NOISE_FILE = WS / 'datasets' / 'environmental_noise' / '1-100210-A-36.wav'
NOISY_OUT = R1_DIR / 'studio_vocadito_benchmark_noisy.json'
SNR_LEVELS = {'clean': None, '10db': 10, '0db': 0}


def mix_noise(x, noise, snr_db):
    """Add looped environmental noise at snr_db, then peak-normalise to 0.9 (same recipe as the Studio evidence probes)."""
    n = np.resize(noise, len(x))
    g = np.sqrt((x ** 2).mean() / ((n ** 2).mean() * 10 ** (snr_db / 10)))
    y = (x + g * n).astype(np.float32)
    return (y / (np.abs(y).max() / 0.9)).astype(np.float32)


def score_predictions(est_pairs, refs):
    """Compute onset_f1 (against A1) and note_f1 (mean across available A1/A2)."""
    a1_ref = refs.get('A1', [])
    a1_onsets = np.asarray([o for o, _ in a1_ref], dtype=float)
    est_onsets = np.asarray([o for o, _ in est_pairs], dtype=float)
    on_f1 = onset_f1(a1_onsets, est_onsets) if len(a1_onsets) > 0 else 0.0
    nt_scores = [note_f1(est_pairs, r) for r in refs.values() if r]
    nt_f1 = float(np.mean(nt_scores)) if nt_scores else 0.0
    return on_f1, nt_f1


def summarize_benchmark(rows, methods):
    """Compute singer-macro 95% bootstrap CIs and paired diffs for all benchmarked methods."""
    summary = {}
    for m in methods:
        nt_clips = {(r['person'], r['id']): r[f'{m}_note_f1'] for r in rows if f'{m}_note_f1' in r}
        on_clips = {(r['person'], r['id']): r[f'{m}_onset_f1'] for r in rows if f'{m}_onset_f1' in r}
        if not nt_clips:
            continue
        nt_ci = singer_metric_ci(clips_to_singer_means(nt_clips), n_resamples=N_RESAMPLES, seed=RNG_SEED)
        on_ci = singer_metric_ci(clips_to_singer_means(on_clips), n_resamples=N_RESAMPLES, seed=RNG_SEED)
        diffs = {}
        for base in ('pyin_4096', 'pesto_notes_from_track', 'studio_v1'):
            if base == m:
                continue
            b_nt = {(r['person'], r['id']): r[f'{base}_note_f1'] for r in rows if f'{base}_note_f1' in r}
            b_on = {(r['person'], r['id']): r[f'{base}_onset_f1'] for r in rows if f'{base}_onset_f1' in r}
            common = sorted(set(nt_clips) & set(b_nt))
            if common:
                d_nt = bootstrap_ci(
                    clips_to_singer_means({c: nt_clips[c] - b_nt[c] for c in common}),
                    n_resamples=N_RESAMPLES,
                    seed=RNG_SEED,
                )
                d_on = bootstrap_ci(
                    clips_to_singer_means({c: on_clips[c] - b_on[c] for c in common}),
                    n_resamples=N_RESAMPLES,
                    seed=RNG_SEED,
                )
                diffs[base] = {'note_f1': d_nt, 'onset_f1': d_on}
        summary[m] = {
            'note_f1': nt_ci,
            'onset_f1': on_ci,
            'diffs': diffs,
        }
    return summary


def run_benchmark(limit=None, include_pre_parity=False, snr_db=None):
    """Run the Studio Vocadito benchmark and return the JSON-serializable payload.

    snr_db: None for the original clean audio, or a number to mix environmental noise into every window first.
    """
    windows = load_vocadito_windows()
    if limit is not None:
        windows = windows[:limit]
    noise = None
    if snr_db is not None:
        noise, _ = librosa.load(NOISE_FILE, sr=E.SR)

    ckpts = candidate_checkpoints()
    loaded_models = {name: boundary.load_boundary_model(path) for name, path in ckpts}
    v1_model = loaded_models.get('studio_v1')

    methods = ['pyin_4096', 'pyin_4096_prepared', 'pesto_notes_from_track'] + [name for name, _ in ckpts]
    if include_pre_parity and v1_model is not None:
        methods.append('studio_v1_pre_parity_100hz_thr05')

    rows = []
    for idx, w in enumerate(windows, 1):
        x_full, _ = E.load_mono(w['wav_path'], max_s=None)
        s0 = max(0, int(round(w['offset_s'] * E.SR)))
        s1 = min(len(x_full), s0 + int(round(w['duration_s'] * E.SR)))
        x = x_full[s0:s1]
        if noise is not None:
            x = mix_noise(x, noise, snr_db)
        dur = len(x) / E.SR

        xn = E.normalize(x)
        trk_o = E.pesto_track(xn)
        cleaned = clean.spectral_gate(x)
        trk_c = E.pesto_track(cleaned)
        use_clean, _ = clean.choose(trk_o, trk_c)
        prepared = cleaned if use_clean else xn
        trk_p = trk_c if use_clean else trk_o
        aligned_p = boundary.align_track_to_20ms(trk_p, sr=E.SR)

        row = {
            'id': w['id'],
            'person': w['person'],
            'cleaned_chosen': bool(use_clean),
            'n_ref_a1': len(w['refs'].get('A1', [])),
        }

        pyin_raw_pairs = notes_dicts_to_pairs(E.notes_from_track(E.pyin_track(x), dur))
        pyin_prep_pairs = notes_dicts_to_pairs(E.notes_from_track(E.pyin_track(prepared), dur))
        pesto_pairs = notes_dicts_to_pairs(E.notes_from_track(trk_p, dur))

        for name, pairs in (
            ('pyin_4096', pyin_raw_pairs),
            ('pyin_4096_prepared', pyin_prep_pairs),
            ('pesto_notes_from_track', pesto_pairs),
        ):
            on_f1, nt_f1 = score_predictions(pairs, w['refs'])
            row[f'{name}_onset_f1'] = on_f1
            row[f'{name}_note_f1'] = nt_f1

        for name, model in loaded_models.items():
            hints = boundary.extract_boundary_hints(trk_p, x=prepared, sr=E.SR, model=model)
            pairs = make_notes_from_aligned(hints['peaks_s'], aligned_p)
            on_f1, nt_f1 = score_predictions(pairs, w['refs'])
            row[f'{name}_onset_f1'] = on_f1
            row[f'{name}_note_f1'] = nt_f1
            row[f'{name}_thr'] = hints['threshold']
            row[f'{name}_edges_source'] = hints.get('boundary_edges_source', 'n/a')

        if include_pre_parity and v1_model is not None:
            pre_peaks = pre_parity_v1_peaks(trk_p, prepared, sr=E.SR, model=v1_model)
            pairs = make_notes_from_aligned(pre_peaks, aligned_p)
            on_f1, nt_f1 = score_predictions(pairs, w['refs'])
            row['studio_v1_pre_parity_100hz_thr05_onset_f1'] = on_f1
            row['studio_v1_pre_parity_100hz_thr05_note_f1'] = nt_f1

        rows.append(row)
        print(
            f"[{idx}/{len(windows)}] {w['id']} ({w['person']}) | "
            f"pyin={row['pyin_4096_note_f1']:.3f} pesto={row['pesto_notes_from_track_note_f1']:.3f} "
            f"v1={row.get('studio_v1_note_f1', 0.0):.3f}",
            flush=True,
        )

    summary = summarize_benchmark(rows, methods)
    model_meta = {
        name: {
            'checkpoint': relpath(path, WS),
            'checkpoint_sha256': sha256_file(path),
            'nin': int(getattr(loaded_models[name], 'nin', 5)),
            'default_threshold': float(getattr(loaded_models[name], 'default_threshold', 0.8)),
        }
        for name, path in ckpts
    }
    return {
        'n_windows': len(rows),
        'n_singers': len({r['person'] for r in rows}),
        'methods': methods,
        'models': model_meta,
        'summary': summary,
        'rows': rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Benchmark real Studio end-to-end Vocadito accuracy.')
    parser.add_argument('--limit', type=int, default=None, help='Optional cap on number of windows.')
    parser.add_argument('--include-pre-parity', action='store_true', help='Include unaligned 100 Hz thr=0.50 pre-parity ablation.')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path (default: protected clean file, or the noisy file when --snr-levels adds noise).')
    parser.add_argument('--snr-levels', type=str, default='clean', help='Comma list from clean,10db,0db. Anything beyond clean writes a separate noisy results file.')
    args = parser.parse_args(argv)

    levels = [s.strip().lower() for s in args.snr_levels.split(',') if s.strip()]
    unknown = [s for s in levels if s not in SNR_LEVELS]
    if unknown:
        parser.error(f'unknown snr level(s): {unknown}')
    if levels == ['clean']:
        out_path = Path(args.output) if args.output else DEFAULT_OUT
        payload = run_benchmark(limit=args.limit, include_pre_parity=args.include_pre_parity)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'\nWrote Studio Vocadito benchmark to: {out_path}')
        payloads = {'clean': payload}
    else:
        out_path = Path(args.output) if args.output else NOISY_OUT
        payloads = {}
        for lv in levels:
            print(f'\n=== SNR level: {lv} ===', flush=True)
            payloads[lv] = run_benchmark(limit=args.limit, include_pre_parity=args.include_pre_parity, snr_db=SNR_LEVELS[lv])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({'levels': payloads}, indent=2), encoding='utf-8')
        print(f'\nWrote noisy Studio Vocadito benchmark to: {out_path}')
    for lv, payload in payloads.items():
        print(f'\n--- {lv} ---')
        for m, stats in payload['summary'].items():
            nt = stats['note_f1']
            on = stats['onset_f1']
            print(f"{m:<36} | Note F1: {nt[0]:.3f} [{nt[1]:.3f}, {nt[2]:.3f}] | Onset F1: {on[0]:.3f} [{on[1]:.3f}, {on[2]:.3f}]")


if __name__ == '__main__':
    main()
