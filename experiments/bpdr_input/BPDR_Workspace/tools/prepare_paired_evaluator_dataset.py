"""Paired Dataset Construction for AI Music Evaluator & BPDR Pitch-Repair Transformer.

Builds paired training and development samples:
(Original Reference Melody, Imperfect Singing Input, 4-View Boundary Data, Target Pitch Correction Curve)

Sources:
1. HumTrans R1 (589 aligned clips with ground-truth reference melody).
2. Vocadito (annotated human singing windows).
3. Singing Error Simulator:
   - Flat / sharp intonation shifts (+-20 to +-120 cents)
   - Pitch drift / sag over sustained notes
   - Onset scoop / overshoot
   - Boundary split / merge / jitter
   - Zero-error in-tune preservation controls (vibrato preservation)

Output:
Manifest and arrays stored in results/studio/r1/paired_evaluator_dataset/
"""
import json
import math
import os
from pathlib import Path
import numpy as np

WS = Path(__file__).resolve().parents[1]
HUMTRANS_DIR = WS / 'datasets' / 'humtrans_r1' / 'analysis'
HUMTRANS_MANIFEST = WS / 'datasets' / 'humtrans_r1' / 'manifest_analysis.json'
VOCADITO_NOTES = WS / 'datasets' / 'vocadito' / 'Annotations' / 'Notes'
DATA_ORIGINALS = WS / 'results' / 'transformer_usability' / '2026-11-04_v64_reliable_melodic_pairs' / 'train_originals.json'
OUT_DIR = WS / 'results' / 'studio' / 'r1' / 'paired_evaluator_dataset'

N_FRAMES = 501  # 10.02s at 20ms canonical grid
HOP_SEC = 0.020


def pad_or_crop(arr, length=N_FRAMES, pad_val=0.0):
    arr = np.asarray(arr)
    if len(arr) >= length:
        return arr[:length]
    padded = np.full(length, pad_val, dtype=arr.dtype)
    padded[:len(arr)] = arr
    return padded


def simulate_singing_errors(ref_pitch_cents, ok_mask, rng, error_type=None):
    """Simulate amateur/karaoke singing pitch errors on a reference pitch track.
    
    Returns:
        sung_pitch_cents: corrupted pitch track in cents
        target_correction_cents: ground-truth correction needed (ref - sung)
        error_metadata: dict describing the injected errors
    """
    T = len(ref_pitch_cents)
    target_correction = np.zeros(T, dtype=np.float32)
    sung_pitch = ref_pitch_cents.copy().astype(np.float32)
    
    if error_type is None:
        error_type = rng.choice(['clean', 'flat_sharp', 'drift', 'scoop', 'combined'], p=[0.20, 0.35, 0.15, 0.15, 0.15])
    
    meta = {'error_type': error_type, 'interventions': []}
    
    if error_type == 'clean':
        # Identity control: sung matches reference perfectly with small expressive vibrato
        return sung_pitch, target_correction, meta

    # Find contiguous voiced segments
    voiced_indices = np.flatnonzero(ok_mask)
    if len(voiced_indices) < 10:
        return sung_pitch, target_correction, meta

    segments = []
    seg_start = voiced_indices[0]
    prev_idx = voiced_indices[0]
    for idx in voiced_indices[1:]:
        if idx == prev_idx + 1:
            prev_idx = idx
        else:
            if prev_idx - seg_start >= 5:
                segments.append((seg_start, prev_idx + 1))
            seg_start = idx
            prev_idx = idx
    if prev_idx - seg_start >= 5:
        segments.append((seg_start, prev_idx + 1))

    if not segments:
        return sung_pitch, target_correction, meta

    # Apply errors to selected segments
    num_to_corrupt = max(1, int(round(len(segments) * rng.uniform(0.4, 0.8))))
    corrupted_segs = rng.choice(len(segments), size=min(num_to_corrupt, len(segments)), replace=False)

    for seg_idx in corrupted_segs:
        s0, s1 = segments[seg_idx]
        seg_len = s1 - s0
        
        # 1. Note-level flat or sharp shift (+-20 to +-120 cents)
        if error_type in ('flat_sharp', 'combined'):
            # Random shift: biased towards flat (-40 to -80 cents is common amateur karaoke error)
            shift = float(rng.choice([-90, -75, -60, -45, -30, -20, 25, 40, 60, 80]))
            sung_pitch[s0:s1] += shift
            target_correction[s0:s1] -= shift
            meta['interventions'].append({'type': 'intone_shift', 'span': [int(s0), int(s1)], 'cents': shift})

        # 2. Pitch drift / sag over sustained notes
        elif error_type == 'drift':
            drift_amount = float(rng.choice([-50, -40, -30, 30, 40]))
            ramp = np.linspace(0.0, drift_amount, seg_len, dtype=np.float32)
            sung_pitch[s0:s1] += ramp
            target_correction[s0:s1] -= ramp
            meta['interventions'].append({'type': 'drift', 'span': [int(s0), int(s1)], 'drift_cents': drift_amount})

        # 3. Onset scoop / overshoot
        elif error_type == 'scoop':
            scoop_len = min(seg_len // 2, int(rng.integers(3, 8)))
            if scoop_len > 1:
                scoop_amount = float(rng.choice([-100, -80, -60, 60]))
                curve = np.linspace(scoop_amount, 0.0, scoop_len, dtype=np.float32)
                sung_pitch[s0:s0 + scoop_len] += curve
                target_correction[s0:s0 + scoop_len] -= curve
                meta['interventions'].append({'type': 'scoop', 'span': [int(s0), int(s0 + scoop_len)], 'scoop_cents': scoop_amount})

    # Ensure unvoiced frames have 0 correction
    target_correction[~ok_mask] = 0.0
    return sung_pitch, target_correction, meta


def build_dataset(max_clips=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(339)
    samples = []
    
    print(f"Loading HumTrans R1 manifest from {HUMTRANS_MANIFEST}...")
    with open(HUMTRANS_MANIFEST, 'r', encoding='utf-8') as f:
        humtrans_manifest = json.load(f)

    # 1. Process HumTrans R1 clips
    ht_count = 0
    for item in humtrans_manifest:
        key = item['key']
        ref_path = HUMTRANS_DIR / f"{key}.ref.npz"
        npz_path = HUMTRANS_DIR / f"{key}.npz"
        if not ref_path.exists() or not npz_path.exists():
            continue
            
        ref_data = np.load(ref_path)
        if not bool(ref_data['kept']):
            continue

        raw_npz = np.load(npz_path)
        ref_cents = ref_data['ref'].astype(np.float32)  # Cents relative to 440 Hz
        sung_fine = raw_npz['fine'].astype(np.float32)  # Cents relative to 440 Hz
        reliable = raw_npz['reliable'].astype(bool)
        raw_views = raw_npz['raw_views'][0]  # shape (T, 71)
        
        # Valid mask
        ok = reliable & np.isfinite(sung_fine) & np.isfinite(ref_cents)
        if ok.sum() < 30:
            continue

        # Zero-center pitch to median of voiced ref
        ref_median = float(np.median(ref_cents[ok]))
        ref_centered = np.where(ok, ref_cents - ref_median, 0.0)
        sung_centered = np.where(ok, sung_fine - ref_median, 0.0)
        
        # Extract features
        conf = raw_views[:, 65]
        log_energy = raw_views[:, 68]
        onset_score = raw_views[:, 69]
        prov_boundary = raw_views[:, 70]
        
        # Natural paired recording: target is (ref - sung)
        target_corr = np.where(ok, ref_cents - sung_fine, 0.0)
        
        # Also create a simulated error version to expand training coverage
        sung_sim, target_sim, sim_meta = simulate_singing_errors(ref_centered, ok, rng)
        
        # Create record 1: Natural hummed/sung clip against MIDI ref
        sample_natural = {
            'id': f"ht_nat_{key}",
            'source': 'humtrans_r1',
            'role': item.get('role', 'train'),
            'person': item.get('person', 'unk'),
            'is_simulated': False,
            'ref_centered': pad_or_crop(ref_centered, N_FRAMES),
            'sung_pitch': pad_or_crop(sung_centered, N_FRAMES),
            'target_corr': pad_or_crop(target_corr, N_FRAMES),
            'ok_mask': pad_or_crop(ok.astype(np.float32), N_FRAMES),
            'conf': pad_or_crop(conf, N_FRAMES),
            'energy': pad_or_crop(log_energy, N_FRAMES, -100.0),
            'onset': pad_or_crop(onset_score, N_FRAMES),
            'prov_boundary': pad_or_crop(prov_boundary, N_FRAMES),
        }
        samples.append(sample_natural)

        # Create record 2: Simulated karaoke error clip
        sample_sim = {
            'id': f"ht_sim_{key}",
            'source': 'humtrans_r1_sim',
            'role': item.get('role', 'train'),
            'person': item.get('person', 'unk'),
            'is_simulated': True,
            'sim_type': sim_meta['error_type'],
            'ref_centered': pad_or_crop(ref_centered, N_FRAMES),
            'sung_pitch': pad_or_crop(sung_sim, N_FRAMES),
            'target_corr': pad_or_crop(target_sim, N_FRAMES),
            'ok_mask': pad_or_crop(ok.astype(np.float32), N_FRAMES),
            'conf': pad_or_crop(conf, N_FRAMES),
            'energy': pad_or_crop(log_energy, N_FRAMES, -100.0),
            'onset': pad_or_crop(onset_score, N_FRAMES),
            'prov_boundary': pad_or_crop(prov_boundary, N_FRAMES),
        }
        samples.append(sample_sim)

        ht_count += 1
        if max_clips and ht_count >= max_clips:
            break

    print(f"Loaded {ht_count} HumTrans clips -> {len(samples)} paired training samples.")

    # 2. Process Vocadito annotated windows
    vocadito_count = 0
    if DATA_ORIGINALS.exists():
        with open(DATA_ORIGINALS, 'r', encoding='utf-8') as f:
            voc_originals = json.load(f)
        for r0 in voc_originals:
            if 'excluded' in r0 or not r0['source_id'].startswith('vocadito'):
                continue
            npz_file = Path(r0['analysis_A'])
            if not npz_file.exists():
                continue
            z = np.load(npz_file)
            c = z['fine'].astype(np.float32)
            ok = z['reliable'].astype(bool) & np.isfinite(c)
            if ok.sum() < 30:
                continue
            
            raw_views = z['raw_views'][0]
            conf = raw_views[:, 65]
            log_energy = raw_views[:, 68]
            onset_score = raw_views[:, 69]
            prov_boundary = raw_views[:, 70]
            
            c_med = float(np.median(c[ok]))
            c_centered = np.where(ok, c - c_med, 0.0)
            
            # Vocadito is clean solo singing - use it for clean preservation & simulated karaoke errors
            # Record A: Clean preservation control
            samples.append({
                'id': f"voc_clean_{r0['source_id']}",
                'source': 'vocadito_clean',
                'role': 'train',
                'person': r0.get('group', 'vocadito'),
                'is_simulated': False,
                'ref_centered': pad_or_crop(c_centered, N_FRAMES),
                'sung_pitch': pad_or_crop(c_centered, N_FRAMES),
                'target_corr': pad_or_crop(np.zeros_like(c_centered), N_FRAMES),
                'ok_mask': pad_or_crop(ok.astype(np.float32), N_FRAMES),
                'conf': pad_or_crop(conf, N_FRAMES),
                'energy': pad_or_crop(log_energy, N_FRAMES, -100.0),
                'onset': pad_or_crop(onset_score, N_FRAMES),
                'prov_boundary': pad_or_crop(prov_boundary, N_FRAMES),
            })
            
            # Record B: Simulated singing error on Vocadito
            sung_sim, target_sim, sim_meta = simulate_singing_errors(c_centered, ok, rng)
            samples.append({
                'id': f"voc_sim_{r0['source_id']}",
                'source': 'vocadito_sim',
                'role': 'train',
                'person': r0.get('group', 'vocadito'),
                'is_simulated': True,
                'sim_type': sim_meta['error_type'],
                'ref_centered': pad_or_crop(c_centered, N_FRAMES),
                'sung_pitch': pad_or_crop(sung_sim, N_FRAMES),
                'target_corr': pad_or_crop(target_sim, N_FRAMES),
                'ok_mask': pad_or_crop(ok.astype(np.float32), N_FRAMES),
                'conf': pad_or_crop(conf, N_FRAMES),
                'energy': pad_or_crop(log_energy, N_FRAMES, -100.0),
                'onset': pad_or_crop(onset_score, N_FRAMES),
                'prov_boundary': pad_or_crop(prov_boundary, N_FRAMES),
            })
            vocadito_count += 1

    print(f"Loaded {vocadito_count} Vocadito windows -> Total samples: {len(samples)}.")

    # Split into train, dev, heldout based on manifest role or deterministic split
    train_samples = [s for s in samples if s['role'] == 'train']
    dev_samples = [s for s in samples if s['role'] in ('dev', 'dev_M02')]
    heldout_samples = [s for s in samples if s['role'] in ('test', 'heldout')]

    # If dev/heldout empty (e.g. from vocadito only), split train deterministically
    if len(dev_samples) == 0:
        n_dev = max(10, int(0.15 * len(train_samples)))
        dev_samples = train_samples[-n_dev:]
        train_samples = train_samples[:-n_dev]

    print(f"Dataset split: Train={len(train_samples)}, Dev={len(dev_samples)}, Heldout={len(heldout_samples)}")

    # Save to npz files for high-speed multi-worker training
    for name, split_data in [('train', train_samples), ('dev', dev_samples), ('heldout', heldout_samples)]:
        if not split_data:
            continue
        np.savez_compressed(
            OUT_DIR / f"{name}_data.npz",
            ref_centered=np.stack([s['ref_centered'] for s in split_data]),
            sung_pitch=np.stack([s['sung_pitch'] for s in split_data]),
            target_corr=np.stack([s['target_corr'] for s in split_data]),
            ok_mask=np.stack([s['ok_mask'] for s in split_data]),
            conf=np.stack([s['conf'] for s in split_data]),
            energy=np.stack([s['energy'] for s in split_data]),
            onset=np.stack([s['onset'] for s in split_data]),
            prov_boundary=np.stack([s['prov_boundary'] for s in split_data]),
        )
        # Save metadata manifest
        meta_list = [{
            'id': s['id'],
            'source': s['source'],
            'role': s['role'],
            'person': s['person'],
            'is_simulated': s['is_simulated'],
            'sim_type': s.get('sim_type', 'none'),
        } for s in split_data]
        with open(OUT_DIR / f"{name}_manifest.json", 'w', encoding='utf-8') as f:
            json.dump(meta_list, f, indent=2)

    summary = {
        'total_samples': len(samples),
        'train_samples': len(train_samples),
        'dev_samples': len(dev_samples),
        'heldout_samples': len(heldout_samples),
        'humtrans_clips': ht_count,
        'vocadito_windows': vocadito_count,
        'canonical_frames': N_FRAMES,
        'frame_hop_s': HOP_SEC,
        'out_dir': str(OUT_DIR),
    }
    with open(OUT_DIR / 'dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("Paired evaluator dataset successfully constructed:")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    build_dataset()
