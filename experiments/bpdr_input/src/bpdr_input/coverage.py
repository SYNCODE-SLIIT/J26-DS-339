"""Training-only geometry and realized-shift probes; never changes the active audit policy."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from .dataset import DataPolicy, reference_mask
from .interventions import AuditPolicy, select_support, longest_span, independent_pitch, render_local, audit_shift
from .analysis import extract
from .pipeline import process, checksum


def candidate_policies():
    baseline = AuditPolicy()
    return {
        'baseline_086': baseline,
        'candidate_066': replace(baseline, version='coverage-candidate-066', analysis_guard_seconds=.05),
        'candidate_052': replace(baseline, version='coverage-candidate-052', analysis_guard_seconds=.05,
                                 ramp_seconds=.06, min_plateau_seconds=.20),
        'candidate_050': replace(baseline, version='coverage-candidate-050', analysis_guard_seconds=.05,
                                 ramp_seconds=.05, min_plateau_seconds=.20),
    }


def training_phrases(manifest):
    """Filter before any crop/reference file is opened."""
    return [phrase for phrase in manifest['phrases'] if phrase['split'] == 'train']


def load_reference(phrase):
    expected = {'audio_path': phrase['sha256'], 'f0_path': phrase['reference_sha256']['f0'],
                'notes_path': phrase['reference_sha256']['notes']}
    for key, digest in expected.items():
        if checksum(phrase[key]) != digest:
            raise ValueError(f'{phrase["phrase_id"]}: changed {key}')
    return np.loadtxt(phrase['f0_path'], delimiter=',', ndmin=2), json.loads(Path(phrase['notes_path']).read_text())


def support_record(times, mask, policy):
    span = longest_span(mask)
    result = dict(longest_frame_count=int(span[1]-span[0]) if span else 0,
                  longest_endpoint_span_seconds=float(times[span[1]-1]-times[span[0]]) if span else 0.)
    try:
        start, end = select_support(times, mask, policy)
        plateau = mask & (times >= start+policy.ramp_seconds+policy.analysis_guard_seconds) & (times <= end-policy.ramp_seconds-policy.analysis_guard_seconds)
        outside = mask & ((times < start-policy.analysis_guard_seconds) | (times > end+policy.analysis_guard_seconds))
        enough = int(plateau.sum()) >= policy.min_audit_frames and int(outside.sum()) >= policy.min_outside_audit_frames
        result.update(support_seconds=[start, end], plateau_frames=int(plateau.sum()), outside_frames=int(outside.sum()),
                      eligible=enough, reason=None if enough else 'Insufficient plateau/outside frame counts')
    except ValueError as exc:
        result.update(support_seconds=None, eligible=False, reason=str(exc))
    return result


def probe(phrase, f0, notes, data_policy, policies, output, component):
    result, prepared = process(phrase['audio_path'], output/'preparation', component/'checkpoints')
    row = dict(phrase_id=phrase['phrase_id'], singer_id=phrase['singer_id'], source_id=phrase['source_id'],
               status=result.status.value, policies={}, training_eligible=False)
    if result.accepted_audio is None:
        row['reason'] = str(result.reason_codes)
        return row
    audio, sr = sf.read(result.accepted_audio.path, dtype='float32')
    before, analyzer = extract(audio, sr, component/'checkpoints')
    times = before['times_seconds']
    yin = independent_pitch(audio, sr)
    reliable, reference = reference_mask(times, f0, notes, data_policy)
    reliable &= before['voiced_mask'] & (reference > 0) & np.isfinite(yin) & (yin > 0)
    valid = reliable.copy()
    reliable[valid] &= abs(1200*np.log2(yin[valid]/before['f0_hz'][valid])) <= AuditPolicy().tracker_agreement_cents
    row.update(accepted_sha256=result.accepted_audio.checksum, analyzer=analyzer,
               yin_window_half_seconds=2048/(2*sr), preparation_run=str(prepared))
    for name, policy in policies.items():
        screen = support_record(times, reliable, policy)
        row['policies'][name] = dict(screen=screen, shifts={}, passed_shifts=0)
        if not screen['eligible']:
            continue
        folder = output/name
        folder.mkdir(parents=True)
        support = screen['support_seconds']
        identity, _ = render_local(audio, sr, 0., support, policy)
        identity_features, _ = extract(identity, sr, component/'checkpoints')
        identity_pass = (np.array_equal(identity, audio) and
                         np.allclose(before['f0_hz'], identity_features['f0_hz'], equal_nan=True, atol=1e-6, rtol=0))
        row['policies'][name]['identity_pass'] = bool(identity_pass)
        for cents in policy.shifts_cents:
            shifted, blend = render_local(audio, sr, cents, support, policy)
            after, _ = extract(shifted, sr, component/'checkpoints')
            measured, _, _ = audit_shift(before, after, yin, independent_pitch(shifted, sr), reliable, support, cents, policy)
            measured.update(same_sample_count=len(shifted) == len(audio),
                            outside_support_exact=bool(np.array_equal(shifted[blend == 0], audio[blend == 0])),
                            identity_pass=bool(identity_pass))
            measured['passed'] = bool(measured['passed'] and measured['same_sample_count'] and
                                      measured['outside_support_exact'] and identity_pass)
            row['policies'][name]['shifts'][f'{cents:+g}'] = measured
            sf.write(folder/f'shift_{cents:+g}.wav', shifted, sr, subtype='FLOAT')
        row['policies'][name]['passed_shifts'] = sum(value['passed'] for value in row['policies'][name]['shifts'].values())
    return row


def run(data_run, output_root, probe_singers=6):
    if probe_singers < 0:
        raise ValueError('probe-singers must be nonnegative')
    component = Path(__file__).resolve().parents[2]
    data_run = Path(data_run).resolve()
    manifest_path = data_run/'manifest.json'
    manifest_hash, review_hash = checksum(manifest_path), checksum(data_run/'reviews.csv')
    manifest = json.loads(manifest_path.read_text())
    data_policy = DataPolicy(**manifest['policy'])
    policies = candidate_policies()
    output = Path(output_root).resolve()/uuid.uuid4().hex
    if not output.is_relative_to(component.resolve()):
        raise ValueError('Coverage evidence must remain inside the BPDR component')
    output.mkdir(parents=True)
    geometry, references = [], {}
    for phrase in training_phrases(manifest):
        f0, notes = load_reference(phrase)
        references[phrase['phrase_id']] = (f0, notes)
        times = np.arange(0., phrase['duration_seconds'], .02)
        mask, _ = reference_mask(times, f0, notes, data_policy)
        geometry.append(dict(phrase_id=phrase['phrase_id'], source_id=phrase['source_id'], singer_id=phrase['singer_id'],
                             policies={name: support_record(times, mask, policy) for name, policy in policies.items()}))
    counts = {}
    for name in policies:
        eligible = [row for row in geometry if row['policies'][name]['eligible']]
        counts[name] = dict(crops=len(eligible), sources=len({row['source_id'] for row in eligible}),
                            singers=len({row['singer_id'] for row in eligible}))
    payload = dict(run_status='running', data_status='exploratory_not_research', data_run=str(data_run),
                   source_manifest_sha256=manifest_hash, review_sheet_sha256=review_hash,
                   policies={name: asdict(policy) for name, policy in policies.items()},
                   training_crops=len(geometry), training_singers=len({row['singer_id'] for row in geometry}),
                   geometry_counts=counts, geometry=geometry, probes=[],
                   untouched_partitions=['development', 'pp1_evaluation', 'final_holdout'],
                   training_eligible_examples=0, production_policy_changed=False,
                   limitations=['Reference coverage is an upper bound before tracker agreement and actual reviews',
                                'One selected crop per singer is a bounded engineering probe, not a benchmark',
                                'Candidate guards/ramps are hypotheses; tolerances and review gates unchanged'])
    def save():
        (output/'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    save()
    print('Reference geometry:', json.dumps(counts), flush=True)
    # Declared selection: best reference support per singer, sorted by descending
    # span, then phrase ID. Never use held-out sources or choose after shift results.
    lookup = {phrase['phrase_id']: phrase for phrase in training_phrases(manifest)}
    selected = []
    seen = set()
    for row in sorted(geometry, key=lambda row: (-row['policies']['candidate_050']['longest_endpoint_span_seconds'], row['phrase_id'])):
        if row['singer_id'] in seen or not row['policies']['candidate_050']['eligible']:
            continue
        seen.add(row['singer_id'])
        selected.append(row['phrase_id'])
    selected = selected[:probe_singers]
    payload['predeclared_probe_phrases'] = selected
    save()
    for pid in selected:
        print('Probing', pid, flush=True)
        f0, notes = references[pid]
        try:
            payload['probes'].append(probe(lookup[pid], f0, notes, data_policy, policies, output/'probes'/pid, component))
        except Exception as exc:
            payload.update(run_status='failed', failure=dict(phrase_id=pid, type=type(exc).__name__, reason=str(exc)))
            save()
            raise
        save()
    payload['realized_shift_counts'] = {}
    for name in policies:
        usable = [row for row in payload['probes'] if row['policies'].get(name, {}).get('passed_shifts', 0) > 0]
        complete = [row for row in usable if row['policies'][name]['passed_shifts'] == len(policies[name].shifts_cents)]
        payload['realized_shift_counts'][name] = dict(any_shift_singers=len({row['singer_id'] for row in usable}),
                                                   all_six_shift_singers=len({row['singer_id'] for row in complete}))
    for phrase in training_phrases(manifest):
        load_reference(phrase)
    assert checksum(manifest_path) == manifest_hash and checksum(data_run/'reviews.csv') == review_hash
    payload['inputs_unchanged'] = True
    payload['run_status'] = 'complete'
    save()
    print('Realized shifts:', json.dumps(payload['realized_shift_counts']), flush=True)
    print('COVERAGE_RUN:', output, flush=True)
    return output, payload


def main():
    component = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-run', type=Path)
    parser.add_argument('--output-root', type=Path, default=component/'runs/coverage')
    parser.add_argument('--probe-singers', type=int, default=6)
    args = parser.parse_args()
    data_run = args.data_run or Path(json.loads((component/'runs/phase2/latest.json').read_text())['path'])
    run(data_run, args.output_root, args.probe_singers)


if __name__ == '__main__':
    main()
