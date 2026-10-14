"""One isolated BPDR crop: technical audio, crop GAME decode, PESTO features, screen."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

from .analysis import extract
from .crop import choose, CROP_POLICY_VERSION


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def game_decode(wav: Path, folder: Path, game_python: Path, game_root: Path, game_model: Path) -> tuple[list[dict], str | None]:
    folder.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([str(game_python), str(game_root / 'infer.py'), 'extract',
                             str(wav), '-m', str(game_model), '--output-formats', 'mid,csv',
                             '--pitch-format', 'number', '--output-dir', str(folder)],
                            cwd=game_root, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'crop GAME exited {result.returncode}: {(result.stderr or result.stdout)[-900:]}')
    csv_path = folder / (wav.stem + '.csv')
    if not csv_path.exists():
        return [], None
    with csv_path.open(newline='', encoding='utf-8') as stream:
        notes = [{'onset_sec': float(row['onset']), 'offset_sec': float(row['offset']),
                  'model_midi_pitch': float(row['pitch'])} for row in csv.DictReader(stream)]
    return notes, digest(csv_path)


def yin_pitch(audio: np.ndarray, rate: int, length: int) -> np.ndarray:
    import librosa
    raw = librosa.yin(audio, sr=rate, fmin=50, fmax=1500, frame_length=2048, hop_length=441)
    if len(raw) < length:
        raw = np.pad(raw, (0, length - len(raw)), constant_values=np.nan)
    return raw[:length]


def screen(features: dict, notes: list[dict], yin: np.ndarray) -> tuple[dict, np.ndarray]:
    times = features['times_seconds']
    voiced = features['voiced_mask']
    pesto = features['f0_hz']
    note_pitch = np.zeros(len(times), dtype=np.float64)
    inside = np.zeros(len(times), dtype=bool)
    overlap = np.zeros(len(times), dtype=np.uint8)
    for note in notes:
        a, b = note['onset_sec'], note['offset_sec']
        overlap += ((times >= a) & (times < b)).astype(np.uint8)
        mask = (times >= a + .10) & (times <= b - .10)
        inside |= mask
        note_pitch[mask] = 440.0 * 2 ** ((note['model_midi_pitch'] - 69) / 12)
    reliable = inside & (overlap == 1) & voiced & np.isfinite(yin) & (yin > 0)
    reliable &= np.isfinite(pesto) & (pesto > 0) & (note_pitch > 0)
    ix = np.flatnonzero(reliable)
    reliable[ix] &= np.abs(1200 * np.log2(yin[ix] / pesto[ix])) <= 50
    ix = np.flatnonzero(reliable)
    reliable[ix] &= np.abs(1200 * np.log2(pesto[ix] / note_pitch[ix])) <= 100
    share = float(reliable.sum() / inside.sum()) if inside.any() else 0.0
    voiced_fraction = float(voiced.mean()) if len(voiced) else 0.0
    overlap_fraction = float((overlap > 1).mean()) if len(overlap) else 0.0
    reasons = []
    if len(notes) < 2:
        reasons.append('TOO_FEW_PREDICTED_NOTES')
    if voiced_fraction < .25:
        reasons.append('LOW_VOICING')
    if reliable.sum() < 10 or share < .35:
        reasons.append('INSUFFICIENT_RELIABLE_PITCH')
    if overlap_fraction > .05:
        reasons.append('OVERLAPPING_PREDICTED_NOTES')
    return {'status': 'review_candidate' if not reasons else 'withheld', 'reasons': reasons,
            'note_count': len(notes), 'frame_count': len(times), 'inside_note_frames': int(inside.sum()),
            'voiced_frames': int(voiced.sum()), 'voiced_fraction': voiced_fraction,
            'reliable_frames': int(reliable.sum()), 'reliable_share_inside_notes': share,
            'overlap_frames': int((overlap > 1).sum()), 'predicted_overlap_fraction': overlap_fraction,
            'policy': 'preparation-screen-exploratory-0.1',
            'manual_checks_required': ['single_voice', 'accompaniment_leakage',
                                       'musical_acceptability', 'ambiguous_transitions']}, reliable


def run(request: dict) -> dict:
    stem = Path(request['stem'])
    destination = Path(request['destination'])
    destination.mkdir(parents=True, exist_ok=False)
    track = request['track']
    note_json = Path(request['note_json'])
    full = json.loads(note_json.read_text(encoding='utf-8'))
    if full['track_id'] != track['track_id'] or full['method'] != 'game_vocal':
        raise ValueError('Full-song GAME prediction identity mismatch')
    info = sf.info(stem)
    if info.channels not in (1, 2) or info.frames < 1:
        raise ValueError('Unsupported or empty vocal stem')
    duration = info.frames / info.samplerate
    crop = choose(full['notes'], duration)
    if crop is None:
        return {'status': 'withheld', 'reasons': ['NO_REST_ALIGNED_CROP_WITH_FIVE_SUNG_SECONDS'],
                'training_eligible': False, 'reference_type': 'model_estimate'}
    start = round(crop.start_sec * info.samplerate)
    end = min(info.frames, round(crop.end_sec * info.samplerate))
    with sf.SoundFile(stem) as stream:
        stream.seek(start)
        channels = stream.read(end - start, dtype='float32', always_2d=True)
    if not np.isfinite(channels).all():
        return {'status': 'technical_failure', 'reasons': ['NONFINITE_AUDIO'], 'training_eligible': False}
    energies = np.sqrt(np.mean(channels.astype(np.float64) ** 2, axis=0))
    channel = int(np.argmax(energies))
    mono = channels[:, channel]
    if info.samplerate != 22050:
        from scipy.signal import resample_poly
        common = math.gcd(info.samplerate, 22050)
        mono = resample_poly(mono, 22050 // common, info.samplerate // common).astype(np.float32)
    if not .25 <= len(mono) / 22050 <= 20.0 + 1 / 22050:
        return {'status': 'technical_failure', 'reasons': ['DURATION_OUT_OF_RANGE'], 'training_eligible': False}
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    if rms < 1e-5 or peak >= 1.0 or float(np.mean(np.abs(mono) >= .999)) > .01:
        return {'status': 'rerecord_required', 'reasons': ['SILENCE_OR_CLIPPING'],
                'training_eligible': False}
    wav = destination / 'vocal.wav'
    sf.write(wav, mono, 22050, subtype='PCM_24')
    accepted, accepted_rate = sf.read(wav, dtype='float32')
    if accepted_rate != 22050 or abs(len(accepted) / 22050 - (end - start) / info.samplerate) > 1 / 22050 + 1e-9:
        raise ValueError('Prepared WAV duration mapping failed')
    game_root = Path(request['game_root'])
    game_model = Path(request['game_model'])
    raw_dir = destination / '_game_tmp'
    notes, raw_hash = game_decode(wav, raw_dir, Path(request['game_python']), game_root, game_model)
    for note in notes:
        if not (0 <= note['onset_sec'] < note['offset_sec'] <= len(accepted) / 22050 + .05
                and 0 <= note['model_midi_pitch'] <= 127):
            raise ValueError('Invalid crop GAME note')
    # Release GAME process before loading PESTO; each song's worker exits after export.
    features, provenance = extract(accepted, 22050, request['pesto_cache'])
    times = features['times_seconds']
    boundary = np.zeros(len(times), dtype=np.float32)
    for note in notes:
        for edge in (note['onset_sec'], note['offset_sec']):
            boundary = np.maximum(boundary, np.exp(-.5 * ((times - edge) / .04) ** 2).astype(np.float32))
    matrix = features['boundary_features_71'].copy()
    matrix[:, 70] = boundary
    if matrix.shape != (len(times), 71) or not np.isfinite(matrix).all():
        raise ValueError('Invalid BPDR model input')
    np.savez_compressed(destination / 'features.npz', boundary_features_71=matrix,
                        times_seconds=times, f0_hz=features['f0_hz'],
                        pitch_change_valid=features['pitch_change_valid'])
    yin = yin_pitch(accepted, 22050, len(times))
    screening, reliable = screen(features, notes, yin)
    edge_warning = bool(crop.warnings) or any(n['onset_sec'] < .10 or
                          n['offset_sec'] > len(accepted) / 22050 - .10 for n in notes)
    if edge_warning:
        screening['reasons'].append('UNCERTAIN_CROP_EDGE')
        screening['status'] = 'withheld'
    np.savez_compressed(destination / 'masks.npz', reliable_pitch_screen=reliable,
                        voiced=features['voiced_mask'],
                        heuristic_boundary_score=features['provisional_boundary_score'].astype(np.float32),
                        crop_edge_protected=
                        (times < .10) | (times > len(accepted) / 22050 - .10))
    boundaries = {'condition': 'game_estimated_crop_decode', 'reference_type': 'model_estimate',
                  'time_origin': 'crop_start', 'pitch_field': 'model_midi_pitch',
                  'pitch_shift_semitones': 0, 'notes': notes, 'raw_crop_csv_sha256': raw_hash,
                  'game_seed': 'not exposed by extraction CLI',
                  'feature_column_70': 'Gaussian GAME note-edge score, sigma=0.04 seconds',
                  'other_boundary_condition': 'audio-derived heuristic in masks.npz; not column 70'}
    (destination / 'boundaries.json').write_text(json.dumps(boundaries, indent=2) + '\n')
    if raw_dir.exists():
        for temp in raw_dir.iterdir():
            temp.unlink()
        raw_dir.rmdir()
    return {'status': 'exported', 'technical_status': 'ACCEPT_UNCHANGED',
            'crop_policy': CROP_POLICY_VERSION, 'crop': {'start_sample': start, 'end_sample': end,
            'original_sample_rate_hz': info.samplerate, 'offset_seconds': start / info.samplerate,
            'duration_seconds': (end - start) / info.samplerate, 'sung_seconds_from_full_game': crop.sung_sec,
            'full_game_note_count': crop.note_count, 'warnings': list(crop.warnings)},
            'preparation': {'selected_channel': channel, 'original_channels': info.channels,
                            'original_rate_hz': info.samplerate, 'export_rate_hz': 22050,
                            'subtype': 'PCM_24', 'denoising': False, 'musical_correction': False,
                            'peak': peak, 'rms': rms},
            'feature_provenance': provenance, 'boundary_condition': boundaries['condition'],
            'screen': screening, 'training_eligible': False, 'review_status': 'pending'
            if screening['status'] == 'review_candidate' else 'not_queued',
            'reference_type': 'model_estimate'}


def main() -> None:
    request = json.load(sys.stdin)
    result = run(request)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
