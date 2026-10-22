"""Measure what a lower separator chunk overlap costs in accuracy for one song.

Separates the song again with the given overlap, then runs GAME on that stem and once more on the kept
default stem. GAME's segmentation is sampled, so the second default run is the noise floor against which
the faster stem is judged. Needs a stem kept by run_local.py --keep-stems. Run with the launcher environment.
"""
import argparse
import csv
import inspect
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
PIPELINE = COMPONENT/'data'/'mtg-jamendo'/'melody-extraction-pipeline'
MODEL = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'


def game(stem, directory):
    python = ASSETS/'.venv-game'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    command = [str(python), 'infer.py', 'extract', str(stem), '-m', str(ASSETS/'models'/'GAME-1.0-medium'/'model.pt'),
               '--output-formats', 'mid,csv', '--pitch-format', 'number', '--output-dir', str(directory)]
    result = subprocess.run(command, cwd=ASSETS/'tools'/'GAME', capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'GAME exited {result.returncode}: {(result.stderr or result.stdout)[-1500:]}')
    path = directory/(stem.stem+'.csv')
    if not path.exists():
        return np.zeros((0, 2)), np.zeros(0)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return np.array([[float(r['onset']), float(r['offset'])] for r in rows]).reshape(-1, 2), np.array([float(r['pitch']) for r in rows])


def agreement(reference, estimate):
    """Note agreement of two transcriptions of the same song, plus pitch difference on onset-matched notes."""
    (ri, rp), (ei, ep) = reference, estimate
    if not len(ri) or not len(ei):
        return dict(reference_notes=len(ri), estimate_notes=len(ei))
    hz = lambda midi: 440.*2**((midi-69)/12)
    _, _, onset = mir_eval.transcription.onset_precision_recall_f1(ri, ei)
    _, _, pitched, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, hz(rp), ei, hz(ep), offset_ratio=None)
    cents = np.array([100*(ep[e]-rp[r]) for r, e in mir_eval.transcription.match_note_onsets(ri, ei)])
    return dict(reference_notes=len(ri), estimate_notes=len(ei), onset_f1=float(onset), onset_pitch_f1=float(pitched),
                median_abs_pitch_difference_cents=float(np.median(abs(cents))) if len(cents) else None,
                p90_abs_pitch_difference_cents=float(np.quantile(abs(cents), .9)) if len(cents) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('track_id')
    parser.add_argument('--overlap', type=int, default=2)
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()
    local_ffmpeg = ASSETS/'tools'/'ffmpeg'/'bin'
    if local_ffmpeg.is_dir():
        os.environ['PATH'] = str(local_ffmpeg)+os.pathsep+os.environ['PATH']
    import torch
    from audio_separator.separator import Separator
    if args.threads:
        torch.set_num_threads(args.threads)
    default_stem = COMPONENT/'data'/'jamendo-vocal-stems'/(args.track_id+'.wav')
    source = next((PIPELINE.parent/'vocal-tagged-audio').rglob(args.track_id+'.low.mp3'))
    directory = COMPONENT/'runs'/'extraction_checks'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    mix = directory/'mix.wav'  # the pipeline's own decode settings
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source), '-ar', '44100', '-ac', '2', '-c:a', 'pcm_s16le', str(mix)], check=True)
    chunking = dict(inspect.signature(Separator.__init__).parameters['mdxc_params'].default, overlap=args.overlap)
    separator = Separator(output_dir=str(directory), model_file_dir=str(ASSETS/'models'/'mlx-audio-separator'), output_format='WAV',
                          output_single_stem='Vocals', mdxc_params=chunking, log_level=logging.WARNING)
    separator.load_model(MODEL)
    started = time.perf_counter()
    separator.separate(str(mix), custom_output_names={'Vocals': f'{args.track_id}_overlap{args.overlap}'})
    seconds = time.perf_counter()-started
    faster_stem = directory/f'{args.track_id}_overlap{args.overlap}.wav'
    a, rate = sf.read(default_stem, dtype='float64', always_2d=True)
    b, _ = sf.read(faster_stem, dtype='float64', always_2d=True)
    length = min(len(a), len(b))
    residual = float(np.sum((a[:length]-b[:length])**2))
    payload = json.loads((PIPELINE/'outputs'/'game_vocal'/(args.track_id+'.json')).read_text(encoding='utf-8'))
    pipeline_notes = (np.array([[n['onset_sec'], n['offset_sec']] for n in payload['notes']]).reshape(-1, 2),
                      np.array([n['model_midi_pitch'] for n in payload['notes']]))
    (directory/'default_again').mkdir()
    (directory/'faster').mkdir()
    repeat = game(default_stem, directory/'default_again')
    faster = game(faster_stem, directory/'faster')
    summary = dict(track_id=args.track_id, audio_seconds=length/rate, overlap=args.overlap, threads=torch.get_num_threads(),
                   separation_seconds=seconds, separation_seconds_per_audio_second=seconds/(length/rate),
                   stem_difference_db=float(10*np.log10(np.sum(a[:length]**2)/residual)) if residual else None,
                   game_noise_floor_default_stem_twice=agreement(pipeline_notes, repeat),
                   default_stem_against_faster_stem=agreement(pipeline_notes, faster),
                   reading='The faster setting costs accuracy only where its agreement falls clearly below the noise floor')
    (directory/'overlap_comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('CHECK_RUN:', directory)


if __name__ == '__main__':
    main()
