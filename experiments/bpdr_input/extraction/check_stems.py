"""Gate J3: can separated Jamendo vocal stems support BPDR's reliable masks and shift audit?

`plan` names training-partition tracks to extract. `measure` cuts kept stems into crops of at most 20 seconds,
runs the Phase 1 path on each, and measures tracker agreement inside GAME notes plus the Phase 2 realized-shift
audit. The same measurements are made on vocadito training crops with the same GAME-based mask. Exploratory:
the stems have no human reference and nothing produced here is training eligible. Run with the component environment.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.analysis import extract
from bpdr_input.dataset import DataPolicy, singer_splits
from bpdr_input.interventions import AuditPolicy, audit_shift, independent_pitch, longest_span, render_local, select_support
from bpdr_input.pipeline import process

COMPONENT = next(p for p in Path(__file__).resolve().parents if p.name == 'bpdr_input' and (p/'pyproject.toml').is_file())
DATASET = COMPONENT/'data'/'mtg-jamendo'
STEMS = COMPONENT/'data'/'jamendo-vocal-stems'
CHECKPOINTS = COMPONENT/'checkpoints'
GAME_PITCH_OFFSET = 0.  # semitones added to GAME's own pitch; fixed by gate J2


def licenses(dataset):
    result, current = {}, None
    with (dataset/'audio_licenses.txt').open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            line = line.strip()
            match = re.fullmatch(r'\d+/(\d+)\.mp3', line)
            if match:
                current = match.group(1)
            elif current and 'creativecommons.org/licenses/' in line:
                result.setdefault(current, re.search(r'licenses/([a-z\-]+)/', line).group(1))
    return result


def eligible_tracks(dataset):
    """Training artists under the component's own grouping, and licenses that permit derivatives."""
    with (dataset/'melody-extraction-pipeline'/'manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    kinds = licenses(dataset)
    seed = DataPolicy().split_seed
    splits = singer_splits([r['artist_id'] for r in rows], seed)
    order = lambda artist: hashlib.sha256(f'{seed}:{artist}'.encode()).hexdigest()
    keep = [dict(r, license=kinds[r['track_id']]) for r in rows
            if splits[r['artist_id']] == 'train' and 'nd' not in kinds.get(r['track_id'], 'nd').split('-')]
    return sorted(keep, key=lambda r: (order(r['artist_id']), float(r['duration_annotation_sec']), int(r['track_id'])))


def plan(count, minimum_seconds=60.):
    chosen = {}
    for row in eligible_tracks(DATASET):
        if float(row['duration_annotation_sec']) >= minimum_seconds:
            chosen.setdefault(row['artist_id'], row)
    return list(chosen.values())[:count]


def note_frames(times, notes, guard):
    inside, pitch = np.zeros(len(times), dtype=bool), np.zeros(len(times))
    for onset, offset, midi in notes:
        frames = (times >= onset+guard) & (times <= offset-guard)
        inside |= frames
        pitch[frames] = 440.*2**((midi+GAME_PITCH_OFFSET-69)/12)
    return inside, pitch


def share(mask, within):
    return float(mask.sum()/within.sum()) if within.any() else 0.


def measure(audio_path, notes, directory, data_policy, policy):
    result, _ = process(audio_path, directory/'analysis', CHECKPOINTS)
    row = dict(status=result.status.value, reasons=list(result.reason_codes), note_count=len(notes))
    if result.accepted_audio is None:
        return row
    audio, sr = sf.read(result.accepted_audio.path, dtype='float32')
    before, _ = extract(audio, sr, CHECKPOINTS)
    times, pesto, voiced = before['times_seconds'], before['f0_hz'], before['voiced_mask']
    yin = independent_pitch(audio, sr)
    inside, note_hz = note_frames(times, notes, data_policy.boundary_guard_seconds)
    agree = inside & voiced & (yin > 0)
    agree[agree] = abs(1200*np.log2(yin[agree]/pesto[agree])) <= policy.tracker_agreement_cents
    # No human reference exists here: the model's note pitch takes the place of the annotated pitch.
    reliable = agree.copy()
    reliable[agree] = abs(1200*np.log2(pesto[agree]/note_hz[agree])) <= data_policy.reference_pitch_tolerance_cents
    span, hop = longest_span(reliable), float(times[1]-times[0])
    row.update(duration_seconds=len(audio)/sr, voiced_fraction=float(voiced.mean()), seconds_inside_notes=float(inside.sum()*hop),
               voiced_inside_notes=share(inside & voiced, inside), trackers_agree_inside_notes=share(agree, inside),
               reliable_inside_notes=share(reliable, inside), reliable_seconds=float(reliable.sum()*hop),
               longest_reliable_span_seconds=float((span[1]-span[0])*hop) if span else 0.)
    try:
        support = select_support(times, reliable, policy)
    except ValueError as exc:
        row['audit'] = dict(support_seconds=None, reason=str(exc), shifts_passed=0)
        return row
    shifts = {}
    for cents in policy.shifts_cents:
        try:
            detuned, blend = render_local(audio, sr, cents, support, policy)
            after, _ = extract(detuned, sr, CHECKPOINTS)
            audit, _, _ = audit_shift(before, after, yin, independent_pitch(detuned, sr), reliable, support, cents, policy)
            exact = len(detuned) == len(audio) and bool(np.array_equal(detuned[blend == 0], audio[blend == 0]))
            shifts[f'{cents:+g}'] = dict(passed=bool(audit['passed'] and exact), yin=audit['independent_yin'], pesto=audit['pesto'],
                                         plateau_coverage=audit['plateau_coverage_fraction'], outside_coverage=audit['outside_coverage_fraction'],
                                         outside_p90_drift_cents=audit['outside_p90_drift_cents'])
        except (ValueError, RuntimeError) as exc:
            shifts[f'{cents:+g}'] = dict(passed=False, reason=str(exc))
    row['audit'] = dict(support_seconds=support, shifts_passed=sum(s['passed'] for s in shifts.values()), shifts=shifts)
    return row


def best_windows(notes, duration, seconds=20., step=5., count=2, minimum=5.):
    """Up to `count` non-overlapping windows with the most predicted singing; a fixed rule, not a listening choice."""
    starts = np.arange(0., max(duration-seconds, 0.)+1e-9, step)
    cover = np.array([sum(max(0., min(off, s+seconds)-max(on, s)) for on, off, _ in notes) for s in starts])
    chosen = []
    for index in np.argsort(-cover, kind='stable'):
        if cover[index] < minimum or len(chosen) == count:
            break
        if all(abs(starts[index]-other) >= seconds for other in chosen):
            chosen.append(float(starts[index]))
    return sorted(chosen)


def jamendo_rows(directory, data_policy, policy):
    eligible = {r['track_id']: r for r in eligible_tracks(DATASET)}
    rows = []
    for stem in sorted(STEMS.glob('*.wav'), key=lambda p: int(p.stem)):
        track = eligible.get(stem.stem)
        if track is None:  # another partition or a NoDerivatives license: never cropped or detuned
            rows.append(dict(source='jamendo', track_id=stem.stem, skipped='not a training-partition track with a derivative-permitting license'))
            continue
        payload = json.loads((DATASET/'melody-extraction-pipeline'/'outputs'/'game_vocal'/(stem.stem+'.json')).read_text(encoding='utf-8'))
        notes = [(n['onset_sec'], n['offset_sec'], n['model_midi_pitch']) for n in payload['notes']]
        info = sf.info(stem)
        for start in best_windows(notes, info.duration):
            crop_id = f'{stem.stem}_t{int(start):04d}'
            out = directory/'jamendo'/crop_id
            out.mkdir(parents=True)
            samples, sr = sf.read(stem, start=int(start*info.samplerate), stop=int((start+20.)*info.samplerate), dtype='float32', always_2d=True)
            sf.write(out/'audio.wav', samples, sr, subtype='FLOAT')
            local = [(on-start, off-start, midi) for on, off, midi in notes if on >= start and off <= start+len(samples)/sr]
            rows.append(dict(source='jamendo', group=track['artist_id'], track_id=stem.stem, license=track['license'], crop_id=crop_id,
                             original_offset_seconds=start, stem_channels=info.channels, stem_sample_rate_hz=info.samplerate,
                             **measure(out/'audio.wav', local, out, data_policy, policy)))
    return rows


def vocadito_rows(directory, data_run, game_run, data_policy, policy):
    manifest = json.loads((data_run/'manifest.json').read_text())
    rows, seen = [], set()
    for phrase in sorted(manifest['phrases'], key=lambda p: p['phrase_id']):
        if phrase['split'] != 'train' or phrase['singer_id'] in seen:
            continue
        seen.add(phrase['singer_id'])
        path = game_run/'default'/(phrase['source_id']+'.csv')
        notes = []
        if path.exists():
            with path.open(newline='') as stream:
                notes = [(float(r['onset']), float(r['offset']), float(r['pitch'])) for r in csv.DictReader(stream)]
        offset, length = phrase['original_offset_seconds'], phrase['duration_seconds']
        local = [(on-offset, off-offset, midi) for on, off, midi in notes if on >= offset and off <= offset+length]
        out = directory/'vocadito'/phrase['phrase_id']
        out.mkdir(parents=True)
        rows.append(dict(source='vocadito', group=phrase['singer_id'], crop_id=phrase['phrase_id'],
                         **measure(phrase['audio_path'], local, out, data_policy, policy)))
    return rows


def summarize(rows):
    done = [r for r in rows if 'audit' in r]
    middle = lambda key: float(np.median([r[key] for r in done])) if done else None
    return dict(crops=len([r for r in rows if 'status' in r]), groups=len({r['group'] for r in rows if 'group' in r}),
                accepted_by_phase1=len(done), skipped=len([r for r in rows if 'skipped' in r]),
                median_voiced_inside_notes=middle('voiced_inside_notes'), median_trackers_agree_inside_notes=middle('trackers_agree_inside_notes'),
                median_reliable_inside_notes=middle('reliable_inside_notes'), median_longest_reliable_span_seconds=middle('longest_reliable_span_seconds'),
                crops_with_audit_support=sum(r['audit']['support_seconds'] is not None for r in done),
                shifts_attempted=sum(len(r['audit'].get('shifts', {})) for r in done), shifts_passed=sum(r['audit']['shifts_passed'] for r in done),
                crops_passing_all_shifts=sum(r['audit']['shifts_passed'] == len(AuditPolicy().shifts_cents) for r in done))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('plan').add_argument('--count', type=int, default=6)
    probe = commands.add_parser('measure')
    probe.add_argument('--game-run', type=Path, required=True, help='check_game.py run directory with GAME notes for vocadito')
    probe.add_argument('--data-run', type=Path, help='Phase 2 data run; default is runs/phase2/latest.json')
    args = parser.parse_args()
    if args.command == 'plan':
        chosen = plan(args.count)
        for row in chosen:
            print(row['track_id'], row['artist_id'], row['license'], row['duration_annotation_sec'], sep='\t')
        print('TRACKS:', ','.join(row['track_id'] for row in chosen))
        return 0
    data_run = args.data_run or Path(json.loads((COMPONENT/'runs'/'phase2'/'latest.json').read_text())['path'])
    directory = COMPONENT/'runs'/'extraction_checks'/uuid.uuid4().hex
    data_policy, policy = DataPolicy(), AuditPolicy()
    rows = jamendo_rows(directory, data_policy, policy)+vocadito_rows(directory, data_run, args.game_run.resolve(), data_policy, policy)
    summary = dict(mask='GAME note interior, PESTO voiced, YIN within 50 cents of PESTO, PESTO within 100 cents of the GAME note',
                   game_pitch_offset_semitones=GAME_PITCH_OFFSET, audit_policy=policy.version,
                   jamendo=summarize([r for r in rows if r['source'] == 'jamendo']),
                   vocadito=summarize([r for r in rows if r['source'] == 'vocadito']),
                   limitations=['Exploratory; no human reference for stems; nothing here is training eligible',
                                'vocadito rows use training singers only, one crop per singer',
                                'A handful of stems cannot represent 257 artists'])
    (directory/'stems_vs_vocadito.json').write_text(json.dumps(dict(summary=summary, crops=rows), indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('CHECK_RUN:', directory)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
