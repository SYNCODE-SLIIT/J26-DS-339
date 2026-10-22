"""Gate J2: run the pinned GAME model on vocadito recordings and score its notes against A1 and A2.

Only training and development singers are read, so the PP1 evaluation and final holdout stay untouched.
Run this with the launcher environment (it needs mir_eval); GAME runs in its own environment.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import mir_eval
import numpy as np

COMPONENT = next(p for p in Path(__file__).resolve().parents if p.name == 'bpdr_input' and (p/'pyproject.toml').is_file())
ASSETS = Path(os.environ.get('MELODY_MODEL_ASSETS', COMPONENT/'checkpoints'/'melody-extraction'))
# Alternative decodings of one waveform, run on a subset because GAME is slow on CPU. A repeat of the default
# shows run-to-run variation; the other two move the boundary threshold.
VARIATIONS = {'default_repeat': [], 'seg_threshold_0.1': ['--seg-threshold', '0.1'], 'seg_threshold_0.4': ['--seg-threshold', '0.4']}


def run_game(inputs, directory, extra):
    """The pipeline's own GAME command, pointed at a folder of solo recordings."""
    python = ASSETS/'.venv-game'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    command = [str(python), 'infer.py', 'extract', str(inputs), '-m', str(ASSETS/'models'/'GAME-1.0-medium'/'model.pt'),
               '--glob', '*.wav', '--output-formats', 'mid,csv', '--pitch-format', 'number', '--output-dir', str(directory), *extra]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ASSETS/'tools'/'GAME', capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'GAME exited {result.returncode}: {(result.stderr or result.stdout)[-1500:]}')
    return time.perf_counter()-started


def reference_notes(path):
    rows = np.loadtxt(path, delimiter=',', ndmin=2)  # start seconds, Hz, duration seconds
    return np.column_stack([rows[:, 0], rows[:, 0]+rows[:, 2]]), rows[:, 1]


def game_notes(path):
    if not path.exists():  # GAME writes nothing when it finds no note
        return np.zeros((0, 2)), np.zeros(0)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    intervals = np.array([[float(r['onset']), float(r['offset'])] for r in rows]).reshape(-1, 2)
    return intervals, np.array([float(r['pitch']) for r in rows])


def hz(midi):
    return 440.*2**((np.asarray(midi)-69)/12)


def scores(reference, reference_hz, estimate, estimate_hz):
    if not len(estimate):
        return {name: dict(precision=0., recall=0., f1=0.) for name in ('onset_only', 'onset_pitch', 'onset_pitch_offset')}
    p, r, f = mir_eval.transcription.onset_precision_recall_f1(reference, estimate)
    result = dict(onset_only=dict(precision=p, recall=r, f1=f))
    for name, ratio in (('onset_pitch', None), ('onset_pitch_offset', .2)):
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(reference, reference_hz, estimate, estimate_hz, offset_ratio=ratio)
        result[name] = dict(precision=p, recall=r, f1=f)
    return result


def mean_scores(rows):
    return {name: {key: float(np.mean([row[name][key] for row in rows])) for key in ('precision', 'recall', 'f1')} for name in rows[0]}


def link_inputs(sources, directory):
    directory.mkdir(parents=True)
    for source in sources:
        target = directory/(source['source_id']+'.wav')
        try:
            os.link(source['paths']['audio'], target)
        except OSError:
            target.write_bytes(Path(source['paths']['audio']).read_bytes())
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-run', type=Path, help='Phase 2 data run; default is runs/phase2/latest.json')
    parser.add_argument('--output-root', type=Path, default=COMPONENT/'runs'/'extraction_checks')
    parser.add_argument('--variation-count', type=int, default=8, help='Shortest recordings also decoded with the alternative settings')
    args = parser.parse_args()
    data_run = args.data_run or Path(json.loads((COMPONENT/'runs'/'phase2'/'latest.json').read_text())['path'])
    manifest = json.loads((data_run/'manifest.json').read_text())
    sources = [s for s in manifest['sources'] if s['split'] in ('train', 'development') and not s['errors']]
    varied = sorted(sources, key=lambda s: (s['duration_seconds'], s['source_id']))[:args.variation_count]
    varied_ids = {s['source_id'] for s in varied}
    directory = args.output_root.resolve()/uuid.uuid4().hex
    audio_seconds = sum(s['duration_seconds'] for s in sources)
    timing = dict(default=run_game(link_inputs(sources, directory/'input'), directory/'default', []))
    subset = link_inputs(varied, directory/'input_variation')
    timing.update({name: run_game(subset, directory/name, extra) for name, extra in VARIATIONS.items()})
    rows = []
    for source in sources:
        sid = source['source_id']
        a1, a1_hz = reference_notes(source['paths']['A1'])
        a2, a2_hz = reference_notes(source['paths']['A2'])
        notes = {name: game_notes(directory/name/(sid+'.csv')) for name in ['default']+(list(VARIATIONS) if sid in varied_ids else [])}
        intervals, midi = notes['default']
        row = dict(source_id=sid, singer_id=source['singer_id'], split=source['split'],
                   note_counts=dict(A1=len(a1), A2=len(a2), **{name: len(value[0]) for name, value in notes.items()}),
                   A2_against_A1=scores(a1, a1_hz, a2, a2_hz),
                   raw_pitch_against_A1=scores(a1, a1_hz, intervals, hz(midi)),
                   raw_pitch_against_A2=scores(a2, a2_hz, intervals, hz(midi)),
                   published_pitch_against_A1=scores(a1, a1_hz, intervals, hz(midi+12)))
        # Pitch convention: compare raw model pitch with annotated pitch on onset-matched notes.
        matched = mir_eval.transcription.match_note_onsets(a1, intervals) if len(intervals) else []
        row['raw_minus_A1_cents'] = [float(1200*np.log2(hz(midi[e])/a1_hz[r])) for r, e in matched]
        # Natural boundary variation: how far do other decodings of the same waveform move the onsets?
        row['onset_f1_against_default'] = {name: scores(intervals, hz(midi), notes[name][0], hz(notes[name][1]))['onset_only']['f1']
                                           for name in notes if name != 'default'} if len(intervals) else {}
        rows.append(row)
    cents = np.array([c for row in rows for c in row['raw_minus_A1_cents']])
    octave = np.round(cents/1200).astype(int)
    summary = dict(
        scope='GAME-1.0-medium on full vocadito recordings of training and development singers only',
        tolerances='mir_eval defaults: 50 ms onset, 50 cents pitch, offset within 20% of duration or 50 ms',
        recordings=len(rows), singers=len({r['singer_id'] for r in rows}), audio_seconds=audio_seconds,
        game_wall_seconds=timing, default_wall_seconds_per_audio_second=timing['default']/audio_seconds,
        variation_recordings=[s['source_id'] for s in varied],
        mean={key: mean_scores([row[key] for row in rows]) for key in
              ('A2_against_A1', 'raw_pitch_against_A1', 'raw_pitch_against_A2', 'published_pitch_against_A1')},
        raw_minus_A1=dict(matched_notes=int(len(cents)), octave_histogram={str(k): int((octave == k).sum()) for k in sorted(set(octave))},
                          median_cents_within_octave=float(np.median(cents-1200*octave)) if len(cents) else None,
                          p90_abs_cents_within_octave=float(np.quantile(abs(cents-1200*octave), .9)) if len(cents) else None),
        mean_onset_f1_against_default={name: float(np.mean([row['onset_f1_against_default'][name] for row in rows if name in row['onset_f1_against_default']]))
                                       for name in VARIATIONS},
        mean_note_count_on_variation_recordings={name: float(np.mean([row['note_counts'][name] for row in rows if name in row['note_counts'] and 'default_repeat' in row['note_counts']]))
                                                 for name in ('A1', 'A2', 'default', *VARIATIONS)},
        limitations=[f'Model predictions scored against two annotators on {len(rows)} recordings; a pilot, not a benchmark',
                     'No BPDR preparation is involved; this characterizes the extractor only'])
    (directory/'game_on_vocadito.json').write_text(json.dumps(dict(summary=summary, recordings=rows), indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('CHECK_RUN:', directory)


if __name__ == '__main__':
    main()
