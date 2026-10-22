"""Gate J5: does separating only an excerpt of a song give the same vocal as separating the whole song?

Separates [start-margin, start+20+margin] of one track with the pipeline's separator settings and compares the
20-second window with the same window of the stem kept from the full-song run, in level and in GAME notes.
GAME samples its boundaries, so a second decode of the full-song crop is the noise floor. Needs a stem kept by
run_local.py --keep-stems. Exploratory, one window per call. Run with the Intel-GPU launcher environment (.venv-xpu).
"""
import argparse
import csv
import json
import logging
import os
from pathlib import Path
import subprocess
import time
import uuid
import mir_eval
import numpy as np
import soundfile as sf

COMPONENT = next(p for p in Path(__file__).resolve().parents if p.name == 'bpdr_input' and (p/'pyproject.toml').is_file())
ASSETS = Path(os.environ.get('MELODY_MODEL_ASSETS', COMPONENT/'checkpoints'/'melody-extraction'))
DATASET = COMPONENT/'data'/'mtg-jamendo'
MODEL = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'


def game(inputs, directory):
    python = ASSETS/'.venv-game'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    command = [str(python), 'infer.py', 'extract', str(inputs), '-m', str(ASSETS/'models'/'GAME-1.0-medium'/'model.pt'),
               '--glob', '*.wav', '--output-formats', 'mid,csv', '--pitch-format', 'number', '--output-dir', str(directory)]
    result = subprocess.run(command, cwd=ASSETS/'tools'/'GAME', capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'GAME exited {result.returncode}: {(result.stderr or result.stdout)[-1500:]}')


def notes(path):
    if not path.exists():  # GAME writes nothing when it finds no note
        return np.zeros((0, 2)), np.zeros(0)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return np.array([[float(r['onset']), float(r['offset'])] for r in rows]).reshape(-1, 2), np.array([float(r['pitch']) for r in rows])


def agreement(reference, estimate):
    (ri, rp), (ei, ep) = reference, estimate
    row = dict(reference_notes=int(len(ri)), estimate_notes=int(len(ei)))
    if not len(ri) or not len(ei):
        return row
    hz = lambda midi: 440.*2**((midi-69)/12)
    _, _, onset = mir_eval.transcription.onset_precision_recall_f1(ri, ei)
    _, _, pitched, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, hz(rp), ei, hz(ep), offset_ratio=None)
    cents = np.array([100*(ep[e]-rp[r]) for r, e in mir_eval.transcription.match_note_onsets(ri, ei)])
    return dict(row, onset_f1=float(onset), onset_pitch_f1=float(pitched),
                median_abs_pitch_difference_cents=float(np.median(abs(cents))) if len(cents) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track-id', required=True)
    parser.add_argument('--start', type=float, required=True, help='Window start in the full song, seconds')
    parser.add_argument('--margin', type=float, default=10., help='Extra audio separated on each side of the window, seconds')
    parser.add_argument('--device', default='xpu')
    args = parser.parse_args()
    local_ffmpeg = ASSETS/'tools'/'ffmpeg'/'bin'
    if local_ffmpeg.is_dir():
        os.environ['PATH'] = str(local_ffmpeg)+os.pathsep+os.environ['PATH']
    import torch
    from audio_separator.separator import Separator
    kept = COMPONENT/'data'/'jamendo-vocal-stems'/(args.track_id+'.wav')
    source = next((DATASET/'vocal-tagged-audio').rglob(args.track_id+'.low.mp3'))
    directory = COMPONENT/'runs'/'extraction_checks'/uuid.uuid4().hex
    for name in ('full_song_stem', 'full_song_stem_again', 'excerpt_stem'):
        (directory/'input'/name).mkdir(parents=True)
    first = max(args.start-args.margin, 0.)
    mix = directory/'excerpt_mix.wav'  # the pipeline's own decode settings, cut while decoding
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source), '-ss', f'{first:.3f}', '-t', f'{args.start-first+20.+args.margin:.3f}',
                    '-ar', '44100', '-ac', '2', '-c:a', 'pcm_s16le', str(mix)], check=True)
    separator = Separator(output_dir=str(directory), model_file_dir=str(ASSETS/'models'/'mlx-audio-separator'), output_format='WAV',
                          output_single_stem='Vocals', log_level=logging.WARNING)
    # The package selects only CUDA, Apple or DirectML devices itself; an Intel GPU is set here, as in run_local.py.
    separator.torch_device = torch.device(args.device)
    separator.load_model(MODEL)
    started = time.perf_counter()
    separator.separate(str(mix), custom_output_names={'Vocals': 'excerpt_stem'})
    seconds = time.perf_counter()-started
    info = sf.info(kept)
    rate = info.samplerate
    a, _ = sf.read(kept, start=int(args.start*rate), stop=int((args.start+20.)*rate), dtype='float64', always_2d=True)
    b, rate_b = sf.read(directory/'excerpt_stem.wav', start=int((args.start-first)*rate), stop=int((args.start-first+20.)*rate), dtype='float64', always_2d=True)
    length = min(len(a), len(b))
    a, b = a[:length], b[:length]
    gain = float(np.sum(a*b)/np.sum(b*b))
    db = lambda residual: float(10*np.log10(np.sum(a**2)/np.sum(residual**2)))
    crop = f'{args.track_id}_t{int(args.start):04d}'
    for name, samples in (('full_song_stem', a), ('full_song_stem_again', a), ('excerpt_stem', b)):
        sf.write(directory/'input'/name/(crop+'.wav'), samples, rate, subtype='PCM_16')
    found = {}
    for name in ('full_song_stem', 'full_song_stem_again', 'excerpt_stem'):
        game(directory/'input'/name, directory/name)
        found[name] = notes(directory/name/(crop+'.csv'))
    summary = dict(track_id=args.track_id, window_seconds=[args.start, args.start+20.], excerpt_seconds=[first, args.start+20.+args.margin],
                   device=args.device, sample_rates=[rate, rate_b], separation_seconds=seconds, separation_seconds_per_excerpt_second=seconds/sf.info(mix).duration,
                   full_song_seconds=info.duration, work_share_of_full_song=sf.info(mix).duration/info.duration,
                   stem_difference_db=db(a-b), gain_of_best_match=gain, stem_difference_db_after_gain_match=db(a-gain*b),
                   game_noise_floor_full_song_stem_twice=agreement(found['full_song_stem'], found['full_song_stem_again']),
                   full_song_stem_against_excerpt_stem=agreement(found['full_song_stem'], found['excerpt_stem']),
                   reading='Excerpt separation costs accuracy only where its agreement falls clearly below the noise floor',
                   limitations=['One window of one track', 'Both stems come from the PyTorch separator on this machine'])
    (directory/'excerpt_comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('CHECK_RUN:', directory)


if __name__ == '__main__':
    main()
