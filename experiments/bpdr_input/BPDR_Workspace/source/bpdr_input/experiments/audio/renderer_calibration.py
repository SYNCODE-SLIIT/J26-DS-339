"""Frozen training-only renderer calibration; no policy adoption or training pairs."""
from bpdr_input import COMPONENT_ROOT, module_source
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import uuid

import numpy as np
import soundfile as sf

from bpdr_input.audio.analysis import extract
from bpdr_input.experiments.audio.audit_diagnostics import failure_reasons, pitch_metrics
from bpdr_input.preparation.diagnostics.coverage import load_reference, support_record, training_phrases
from bpdr_input.preparation.sources.data_builder import validate_run, write_json
from bpdr_input.preparation.sources.dataset import DataPolicy, reference_mask
from bpdr_input.repair.interventions import AuditPolicy, audit_shift, envelope, independent_pitch, render_local
from bpdr_input.audio.pipeline import checksum, process
from bpdr_input.experiments.audio.renderer_candidates import candidate_audio


# Declared before measurement. Never choose a different renderer per crop/shift.
CONDITIONS = {'active_fft2048_hop512': None,
              'candidate_fft1024_hop128': {'n_fft': 1024, 'hop_length': 128},
              'candidate_fft1024_hop256': {'n_fft': 1024, 'hop_length': 256}}
LISTENING_SHIFTS = (-30., 30.)
PROFILES = [dict(name=f'{kind}_{hz}', kind=kind, base_hz=hz, depth_cents=35., rate_hz=5.)
            for kind in ('constant', 'vibrato') for hz in (110., 220., 440., 880.)]
PROFILES += [dict(name=f'vibrato_wide_{rate}', kind='vibrato', base_hz=220., depth_cents=70., rate_hz=rate)
             for rate in (3., 7.)]
PROFILES += [dict(name=f'glide_{direction}', kind='glide', base_hz=220., direction=direction)
             for direction in (-1., 1.)]


def select_vocadito(manifest, coverage):
    """Select all active-policy reference-eligible training crops, before rendering."""
    eligible = {row['phrase_id'] for row in coverage['geometry'] if row['policies']['baseline_086']['eligible']}
    selected = sorted((r for r in training_phrases(manifest) if r['phrase_id'] in eligible),
                      key=lambda r: r['phrase_id'])
    if len({r['phrase_id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate training phrase')
    return selected


def controlled_signal(profile, cents=0., sr=22050):
    t = np.arange(4*sr)/sr
    if profile['kind'] == 'constant':
        contour = np.zeros(len(t))
    elif profile['kind'] == 'vibrato':
        contour = profile['depth_cents']*np.sin(2*np.pi*profile['rate_hz']*t)
    elif profile['kind'] == 'glide':
        contour = profile['direction']*(-100.+50.*t)
    else:
        raise ValueError('Unknown constructed contour')
    f0 = profile['base_hz']*2**((contour+cents)/1200.)
    phase = 2*np.pi*np.cumsum(f0)/sr
    # Same amplitude envelope and harmonics for every analytic oracle.
    amplitude = .10*(1.+.20*np.sin(2*np.pi*2.*t))
    return (amplitude*(np.sin(phase)+.4*np.sin(2*phase)+.2*np.sin(3*phase))).astype(np.float32)


def fine_pass(metrics, policy):
    return bool(metrics['frames'] >= policy.min_audit_frames and metrics['median_error_cents'] is not None
                and metrics['median_error_cents'] <= policy.median_error_tolerance_cents
                and metrics['p90_error_cents'] <= policy.p90_error_tolerance_cents)


def summarize(rows):
    result = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        selected = [r for r in rows if r['dataset'] == dataset]
        baseline = {(r['record_id'], r['cents']) for r in selected
                    if r['condition'] == 'active_fft2048_hop512' and r['audit']['passed']}
        for name in CONDITIONS:
            group = [r for r in selected if r['condition'] == name]
            passed = [r for r in group if r['audit']['passed']]
            keys = {(r['record_id'], r['cents']) for r in passed}
            complete = {r['record_id'] for r in group if
                        sum(p['record_id'] == r['record_id'] for p in passed) == len(AuditPolicy().shifts_cents)}
            result[f'{dataset}/{name}'] = dict(tested_variants=len(group), passing_variants=len(passed),
                tested_crops=len({r['record_id'] for r in group}), tested_sources=len({r['source_id'] for r in group}),
                tested_groups=len({r['group_id'] for r in group}), group_kind='singer' if dataset == 'vocadito' else 'artist_proxy',
                any_passing_crops=len({r['record_id'] for r in passed}), all_six_crops=len(complete),
                retained_active_passes=len(keys & baseline), lost_active_passes=len(baseline-keys),
                added_passes=len(keys-baseline), failures=dict(Counter(reason for r in group for reason in r['reasons'])))
    return result


def run(coverage_path, jamendo_path):
    import librosa
    import torch
    component = COMPONENT_ROOT
    policy = AuditPolicy()
    tracked = {}

    def track(path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(component):
            raise ValueError('Calibration input must remain inside BPDR')
        digest = checksum(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Changed input: {path}')
        tracked[str(path)] = digest
        return path

    coverage_path, jamendo_path = track(coverage_path), track(jamendo_path)
    coverage = json.loads(coverage_path.read_text())
    jamendo = json.loads(jamendo_path.read_text())
    policy_json = json.loads(json.dumps(asdict(policy)))
    if coverage['run_status'] != 'complete' or jamendo['run_status'] != 'complete':
        raise ValueError('Completed input evidence required')
    if coverage['policies']['baseline_086'] != policy_json or jamendo['policy'] != policy_json:
        raise ValueError('Active policy mismatch')
    data_run = Path(coverage['data_run'])
    manifest_path = track(data_run/'manifest.json', coverage['source_manifest_sha256'])
    track(data_run/'reviews.csv', coverage['review_sheet_sha256'])
    manifest = json.loads(manifest_path.read_text())
    phrases = select_vocadito(manifest, coverage)
    prepared = Path(jamendo['preparation_path'])
    validate_run(prepared)
    track(prepared/'manifest.json')
    track(prepared/'reviews.csv')
    records = {r['record_id']: r for r in json.loads((prepared/'manifest.json').read_text())['records']}
    jam_probes = [r for r in jamendo['probes'] if r['shifts']]
    for probe in jam_probes:
        record = records[probe['record_id']]
        if record['split'] != 'train' or record['training_eligible']:
            raise ValueError('Unsafe Jamendo partition/eligibility')
        track(record['accepted_audio_path'], probe['accepted_sha256'])
        track(record['screen_masks']['path'], record['screen_masks']['sha256'])
    for phrase in phrases:
        load_reference(phrase)
        track(phrase['audio_path'], phrase['sha256'])
        track(phrase['f0_path'], phrase['reference_sha256']['f0'])
        track(phrase['notes_path'], phrase['reference_sha256']['notes'])

    root = component/'runs/renderer_calibration'/uuid.uuid4().hex
    root.mkdir(parents=True)
    snapshots = root/'source_code_at_run'
    snapshots.mkdir()
    for name in ('renderer_calibration', 'renderer_candidates', 'audit_diagnostics', 'coverage',
                 'analysis', 'interventions', 'dataset', 'pipeline', 'data_builder'):
        source = track(module_source(name+'.py'))
        (snapshots/source.name).write_bytes(source.read_bytes())
    track(Path(librosa.effects.__file__)) if Path(librosa.effects.__file__).is_relative_to(component) else None
    result = dict(run_status='running', version='renderer-calibration-exploratory-0.1',
        data_status='exploratory_not_research', training_ready_examples=0, adopted=False,
        policy=asdict(policy), conditions=CONDITIONS, declared_vocadito_ids=[r['phrase_id'] for r in phrases],
        declared_jamendo_ids=[r['record_id'] for r in jam_probes], declared_control_profiles=PROFILES,
        listening_shifts=LISTENING_SHIFTS, tracked_inputs=tracked, librosa_version=librosa.__version__,
        torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
        sources=[], variants=[], controls=[], artifacts=[], limitations=[
            'All three reference-geometry eligible Vocadito crops belong to one singer; not a diverse voice benchmark',
            'Jamendo cases were inspected earlier; this is repeated calibration, not held-out evaluation',
            'Constructed controls have known contours; real-audio shift estimates are not human correction ground truth',
            'Listening comparisons are labelled diagnostics and have no human ratings yet',
            'No geometry/threshold/model/default changes, training pairs or automatic winner selection'])
    def save(): write_json(root/'calibration.json', result)
    save()
    print('CALIBRATION_RUN:', root, flush=True)
    listening = []

    def measure(record_id, dataset, source_id, group_id, path, reliable_provider, frozen_support=None):
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim != 1 or sr != 22050 or len(audio)/sr > 20.:
            raise ValueError('Invalid accepted crop')
        before, analyzer = extract(audio, sr, component/'checkpoints')
        yin = independent_pitch(audio, sr)
        reliable = reliable_provider(before, yin)
        geometry = support_record(before['times_seconds'], reliable, policy)
        support = frozen_support if frozen_support is not None else geometry['support_seconds']
        entry = dict(record_id=record_id, dataset=dataset, source_id=source_id, group_id=group_id,
                     accepted_path=str(path), accepted_sha256=checksum(path), geometry=geometry, analyzer=analyzer,
                     reference_kind='human_f0_A1_A2' if dataset == 'vocadito' else 'model_estimate')
        result['sources'].append(entry)
        if not geometry['eligible']:
            entry['exclusion'] = geometry['reason']
            save()
            return
        if frozen_support is not None and not np.allclose(support, geometry['support_seconds'], rtol=0, atol=1e-12):
            raise ValueError('Frozen Jamendo support changed')
        outside = envelope(np.arange(len(audio))/sr, support, policy.ramp_seconds) == 0
        for name, settings in CONDITIONS.items():
            def render(cents):
                return render_local(audio, sr, cents, support, policy)[0] if settings is None else candidate_audio(audio, sr, cents, support, settings, policy)
            identity_pass = bool(np.array_equal(render(0.), audio))
            for cents in policy.shifts_cents:
                print('MEASURE:', record_id, name, cents, flush=True)
                shifted = render(cents)
                after, _ = extract(shifted, sr, component/'checkpoints')
                metrics, _, _ = audit_shift(before, after, yin, independent_pitch(shifted, sr), reliable, support, cents, policy)
                metrics.update(requested_cents=cents, identity_pass=identity_pass,
                    same_sample_count=len(shifted) == len(audio), outside_support_exact=bool(np.array_equal(audio[outside], shifted[outside])))
                reasons = failure_reasons(metrics, policy)
                if metrics['passed'] != (not reasons):
                    raise ValueError('Diagnostic and active audit decisions differ')
                row = dict(record_id=record_id, dataset=dataset, source_id=source_id, group_id=group_id,
                    condition=name, cents=cents, audit=metrics, reasons=reasons,
                    waveform_sha256=hashlib.sha256(shifted.tobytes()).hexdigest())
                if cents in LISTENING_SHIFTS:
                    folder = root/'listening'/record_id
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder/f'{name}_{cents:+g}.wav'
                    sf.write(target, shifted, sr, subtype='FLOAT')
                    roundtrip, loaded_sr = sf.read(target, dtype='float32')
                    if loaded_sr != sr or not np.array_equal(roundtrip, shifted):
                        raise ValueError('Listening artifact changed samples')
                    row['listening_artifact'] = dict(path=str(target), sha256=checksum(target))
                    result['artifacts'].append(row['listening_artifact'])
                    listening.append((record_id, path, name, cents, target, metrics['passed'], reasons))
                result['variants'].append(row)
                save()

    try:
        previous = {r['phrase_id']: r for r in coverage['probes']}
        for phrase in phrases:
            f0, notes = load_reference(phrase)
            if phrase['phrase_id'] in previous:
                old = previous[phrase['phrase_id']]
                diagnostic = json.loads(track(Path(old['preparation_run'])/'diagnostics.json').read_text())
                path = track(diagnostic['result']['accepted_audio']['path'], old['accepted_sha256'])
            else:
                technical, run_path = process(phrase['audio_path'], root/'preparation'/phrase['phrase_id'], component/'checkpoints')
                if technical.accepted_audio is None:
                    result['sources'].append(dict(record_id=phrase['phrase_id'], dataset='vocadito', exclusion=technical.reason_codes))
                    save()
                    continue
                path = track(technical.accepted_audio.path, technical.accepted_audio.checksum)
                track(run_path/'diagnostics.json')
            def reliable_voc(before, yin):
                mask, reference = reference_mask(before['times_seconds'], f0, notes, DataPolicy(**manifest['policy']))
                mask &= before['voiced_mask'] & (reference > 0) & np.isfinite(yin) & (yin > 0)
                valid = mask.copy()
                mask[valid] &= abs(1200*np.log2(yin[valid]/before['f0_hz'][valid])) <= policy.tracker_agreement_cents
                return mask
            measure(phrase['phrase_id'], 'vocadito', phrase['source_id'], phrase['singer_id'], path, reliable_voc)
        for probe in jam_probes:
            record = records[probe['record_id']]
            with np.load(record['screen_masks']['path']) as masks:
                reliable, times = masks['reliable_mask'], masks['times_seconds']
            def reliable_jam(before, yin):
                if not np.array_equal(before['times_seconds'], times):
                    raise ValueError('Screen grid changed')
                return reliable
            measure(record['record_id'], 'jamendo', record['source_id'], record['group_id'],
                    Path(record['accepted_audio_path']), reliable_jam, probe['geometry']['support_seconds'])
        sr = 22050
        for profile in PROFILES:
            print('CONTROL:', profile['name'], flush=True)
            original = controlled_signal(profile)
            yin = independent_pitch(original, sr)
            times = np.arange(len(yin))*.02
            mask = (times >= 1.) & (times <= 3.)
            for cents in policy.shifts_cents:
                oracle = pitch_metrics(yin, independent_pitch(controlled_signal(profile, cents), sr), mask, cents)
                for name, settings in CONDITIONS.items():
                    full = librosa.effects.pitch_shift(original, sr=sr, n_steps=cents/100, scale=False, **(settings or {}))
                    measured = pitch_metrics(yin, independent_pitch(full, sr), mask, cents)
                    result['controls'].append(dict(profile=profile['name'], condition=name, cents=cents,
                        analytic_oracle=oracle, renderer=measured, fine_pass=fine_pass(measured, policy),
                        oracle_fine_pass=fine_pass(oracle, policy), same_sample_count=len(full) == len(original)))
            save()
        result['summary'] = summarize(result['variants'])
        result['control_summary'] = {name: dict(tested=sum(r['condition'] == name for r in result['controls']),
            fine_passes=sum(r['condition'] == name and r['fine_pass'] for r in result['controls']),
            oracle_failures=sum(r['condition'] == name and not r['oracle_fine_pass'] for r in result['controls']),
            max_p90_error_cents=max(r['renderer']['p90_error_cents'] for r in result['controls'] if r['condition'] == name))
            for name in CONDITIONS}
        body = ['<!doctype html><meta charset="utf-8"><title>BPDR renderer comparison</title>',
                '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto}td,th{padding:12px;border-bottom:1px solid #ccc}audio{width:230px}</style>',
                '<h1>Renderer listening comparison</h1><p>Exploratory labelled diagnostics. No human ratings, approved references or production repairs. ',
                'All clips are matched training crops. Compare naturalness, vibrato, timing and edit boundaries against the original. ',
                'Automated pass status measures the shift audit only. Complete each comparison; do not choose different settings per crop.</p>',
                '<table><tr><th>Source and unchanged original</th><th>Renderer / shift</th><th>Experimental audio</th><th>Audit</th></tr>']
        for rid, original, name, cents, target, passed, reasons in listening:
            body.append(f'<tr><td>{html.escape(rid)}<br><audio controls preload="none" src="{html.escape(original.as_uri())}"></audio></td>'
                        f'<td>{html.escape(name)} / {cents:+g} cents</td><td><audio controls preload="none" src="{target.relative_to(root).as_posix()}"></audio></td>'
                        f'<td>{"PASS" if passed else "FAIL"}<br>{html.escape(", ".join(reasons))}</td></tr>')
        body.append('</table>')
        (root/'listen.html').write_text('\n'.join(body), encoding='utf-8')
        result['artifacts'].append(dict(path=str(root/'listen.html'), sha256=checksum(root/'listen.html')))
        validate_run(prepared)
        if any(checksum(path) != digest for path, digest in tracked.items()):
            raise ValueError('Inputs, reviews or source code changed during calibration')
        result.update(run_status='complete', inputs_unchanged=True)
        save()
        print(json.dumps(dict(summary=result['summary'], controls=result['control_summary']), indent=2), flush=True)
    except BaseException as exc:
        result.update(run_status='failed', failure=dict(type=type(exc).__name__, reason=str(exc)))
        save()
        raise
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coverage', type=Path)
    parser.add_argument('jamendo', type=Path)
    args = parser.parse_args()
    run(args.coverage, args.jamendo)


if __name__ == '__main__':
    main()
