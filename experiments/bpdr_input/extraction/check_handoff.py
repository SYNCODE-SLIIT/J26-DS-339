"""Gate H1: does the team's GAME extractor read BPDR's accepted audio as well as the audio it was checked on?

Three GAME decodes of the crops already measured by check_stems.py: the crop as given to Phase 1 (twice, the
second is the noise floor because GAME samples its boundaries) and the Phase 1 accepted WAV. They are compared
with each other, with GAME on the full recording or full-song stem cut to the same window, and, for vocadito,
with annotators A1 and A2. Training singers only; exploratory. Run with the launcher environment (.venv-run).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
import mir_eval
import numpy as np
import soundfile as sf

COMPONENT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get('MELODY_MODEL_ASSETS', COMPONENT/'checkpoints'/'melody-extraction'))
PIPELINE = COMPONENT/'data'/'mtg-jamendo'/'melody-extraction-pipeline'
EDGE = .5  # seconds; notes starting this close to a crop edge are also reported with them left out


def run_game(inputs, directory):
    python = ASSETS/'.venv-game'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    command = [str(python), 'infer.py', 'extract', str(inputs), '-m', str(ASSETS/'models'/'GAME-1.0-medium'/'model.pt'),
               '--glob', '*.wav', '--output-formats', 'mid,csv', '--pitch-format', 'number', '--output-dir', str(directory)]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ASSETS/'tools'/'GAME', capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'GAME exited {result.returncode}: {(result.stderr or result.stdout)[-1500:]}')
    return time.perf_counter()-started


def read_game(path):
    if not path.exists():  # GAME writes nothing when it finds no note
        return np.zeros((0, 2)), np.zeros(0)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return np.array([[float(r['onset']), float(r['offset'])] for r in rows]).reshape(-1, 2), np.array([float(r['pitch']) for r in rows])


def window(notes, start, length):
    """Notes wholly inside the crop, moved to the crop's time origin."""
    intervals, pitch = notes
    keep = (intervals[:, 0] >= start) & (intervals[:, 1] <= start+length) if len(intervals) else np.zeros(0, bool)
    return intervals[keep]-start, pitch[keep]


def hz(midi):
    return 440.*2**((np.asarray(midi)-69)/12)


def agreement(reference, estimate, reference_is_hz=False):
    (ri, rp), (ei, ep) = reference, estimate
    row = dict(reference_notes=int(len(ri)), estimate_notes=int(len(ei)))
    if not len(ri) or not len(ei):
        return dict(row, onset_f1=0., onset_pitch_f1=0.)
    rh = rp if reference_is_hz else hz(rp)
    _, _, onset = mir_eval.transcription.onset_precision_recall_f1(ri, ei)
    _, _, pitched, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rh, ei, hz(ep), offset_ratio=None)
    cents = np.array([1200*np.log2(hz(ep[e])/rh[r]) for r, e in mir_eval.transcription.match_note_onsets(ri, ei)])
    return dict(row, onset_f1=float(onset), onset_pitch_f1=float(pitched),
                median_abs_pitch_difference_cents=float(np.median(abs(cents))) if len(cents) else None)


def interior(notes, length):
    intervals, pitch = notes
    keep = (intervals[:, 0] >= EDGE) & (intervals[:, 0] <= length-EDGE) if len(intervals) else np.zeros(0, bool)
    return intervals[keep], pitch[keep]


def annotation(path):
    rows = np.loadtxt(path, delimiter=',', ndmin=2)  # start seconds, Hz, duration seconds
    return np.column_stack([rows[:, 0], rows[:, 0]+rows[:, 2]]), rows[:, 1]


def mean(rows, *keys):
    values = []
    for row in rows:
        for key in keys:
            row = row.get(key) if isinstance(row, dict) else None
        if row is not None:
            values.append(row)
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stems-run', type=Path, required=True, help='check_stems.py run directory with the crop folders')
    parser.add_argument('--game-run', type=Path, required=True, help='check_game.py run directory with GAME on full vocadito recordings')
    parser.add_argument('--data-run', type=Path, help='Phase 2 data run; default is runs/phase2/latest.json')
    args = parser.parse_args()
    data_run = args.data_run or Path(json.loads((COMPONENT/'runs'/'phase2'/'latest.json').read_text())['path'])
    manifest = json.loads((data_run/'manifest.json').read_text())
    phrases = {p['phrase_id']: p for p in manifest['phrases']}
    sources = {s['source_id']: s for s in manifest['sources']}
    stems_run, game_run = args.stems_run.resolve(), args.game_run.resolve()
    measured = json.loads((stems_run/'stems_vs_vocadito.json').read_text())['crops']
    directory = COMPONENT/'runs'/'extraction_checks'/uuid.uuid4().hex
    for name in ('given', 'given_again', 'accepted'):
        (directory/'input'/name).mkdir(parents=True)
    crops = []
    for row in measured:
        if row.get('status') not in ('ACCEPT_UNCHANGED', 'ACCEPT_PREPARED'):
            continue
        saved = next((stems_run/row['source']/row['crop_id']/'analysis').iterdir())
        if row['source'] == 'vocadito' and phrases[row['crop_id']]['split'] != 'train':
            raise SystemExit(f"{row['crop_id']} is not a training crop")
        given, accepted = sf.info(saved/'original.wav'), sf.info(saved/'accepted.wav')
        for name, path in (('given', saved/'original.wav'), ('given_again', saved/'original.wav'), ('accepted', saved/'accepted.wav')):
            shutil.copyfile(path, directory/'input'/name/(row['crop_id']+'.wav'))
        crops.append(dict(source=row['source'], crop_id=row['crop_id'], group=row['group'], seconds=given.duration,
                          given=f'{given.samplerate} Hz, {given.channels} ch, {given.subtype}',
                          accepted=f'{accepted.samplerate} Hz, {accepted.channels} ch, {accepted.subtype}',
                          offset=row.get('original_offset_seconds')))
    timing = {name: run_game(directory/'input'/name, directory/name) for name in ('given', 'given_again', 'accepted')}
    for crop in crops:
        cid, length = crop['crop_id'], crop['seconds']
        notes = {name: read_game(directory/name/(cid+'.csv')) for name in timing}
        if crop['source'] == 'vocadito':
            phrase = phrases[cid]
            start, source = phrase['original_offset_seconds'], sources[phrase['source_id']]
            full = window(read_game(game_run/'default'/(phrase['source_id']+'.csv')), start, length)
            for annotator in ('A1', 'A2'):
                reference = window(annotation(source['paths'][annotator]), start, length)
                crop[f'against_{annotator}'] = {name: agreement(reference, value, reference_is_hz=True) for name, value in dict(notes, full_recording=full).items()}
        else:
            start = crop['offset']
            payload = json.loads((PIPELINE/'outputs'/'game_vocal'/(cid.split('_')[0]+'.json')).read_text(encoding='utf-8'))
            song = (np.array([[n['onset_sec'], n['offset_sec']] for n in payload['notes']]).reshape(-1, 2),
                    np.array([n['model_midi_pitch'] for n in payload['notes']]))
            full = window(song, start, length)
        crop['note_counts'] = dict({name: int(len(value[0])) for name, value in notes.items()}, full_cut_to_window=int(len(full[0])))
        crop['noise_floor_given_twice'] = agreement(notes['given'], notes['given_again'])
        crop['accepted_against_given'] = agreement(notes['given'], notes['accepted'])
        crop['full_against_given'] = agreement(notes['given'], full)
        crop['interior'] = dict(noise_floor_given_twice=agreement(interior(notes['given'], length), interior(notes['given_again'], length)),
                                accepted_against_given=agreement(interior(notes['given'], length), interior(notes['accepted'], length)),
                                full_against_given=agreement(interior(notes['given'], length), interior(full, length)))
    summary = {}
    for kind in ('vocadito', 'jamendo'):
        rows = [c for c in crops if c['source'] == kind]
        part = dict(crops=len(rows), groups=len({c['group'] for c in rows}), audio_seconds=float(sum(c['seconds'] for c in rows)),
                    formats=sorted({c['given']+' -> '+c['accepted'] for c in rows}))
        for key in ('noise_floor_given_twice', 'accepted_against_given', 'full_against_given'):
            part[key] = dict(onset_f1=mean(rows, key, 'onset_f1'), onset_pitch_f1=mean(rows, key, 'onset_pitch_f1'),
                             interior_onset_f1=mean(rows, 'interior', key, 'onset_f1'), interior_onset_pitch_f1=mean(rows, 'interior', key, 'onset_pitch_f1'),
                             median_abs_pitch_difference_cents=mean(rows, key, 'median_abs_pitch_difference_cents'))
        if kind == 'vocadito':
            for annotator in ('A1', 'A2'):
                part[f'against_{annotator}'] = {name: dict(onset_f1=mean(rows, f'against_{annotator}', name, 'onset_f1'),
                                                           onset_pitch_f1=mean(rows, f'against_{annotator}', name, 'onset_pitch_f1'))
                                                for name in ('given', 'given_again', 'accepted', 'full_recording')}
        summary[kind] = part
    summary.update(game_wall_seconds=timing, edge_seconds=EDGE, tolerances='mir_eval defaults: 50 ms onset, 50 cents pitch; offsets not scored',
                   limitations=['Crops already measured by check_stems.py; vocadito training singers only, one crop per singer',
                                'A handful of Jamendo crops cannot represent 257 artists',
                                'Accepted audio is whatever Phase 1 returned for these crops; repeat the check once a repair exists'])
    (directory/'handoff_and_crop_check.json').write_text(json.dumps(dict(summary=summary, crops=crops), indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('CHECK_RUN:', directory)


if __name__ == '__main__':
    main()
