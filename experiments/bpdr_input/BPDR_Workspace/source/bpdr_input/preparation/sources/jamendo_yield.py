"""Measure the frozen sample and v0.3 intervention support without changing eligibility."""
from bpdr_input import module_source
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.audio.analysis import extract
from bpdr_input.preparation.diagnostics.coverage import support_record
from bpdr_input.preparation.sources.data_builder import validate_run, write_json
from bpdr_input.repair.interventions import AuditPolicy, audit_shift, independent_pitch, render_local
from bpdr_input.audio.pipeline import checksum


def counts(rows):
    return dict(crops=len(rows), songs=len({r['source_id'] for r in rows}),
                artists=len({r['group_id'] for r in rows}))


def run(batch_root):
    batch_root = Path(batch_root).resolve()
    component = batch_root.parents[2]
    batch = json.loads((batch_root/'batch.json').read_text())
    if batch['run_status'] != 'complete' or len(batch['track_ids']) != 5:
        raise ValueError('A completed frozen five-track batch is required')
    if checksum(batch['selection_path']) != batch['selection_sha256']:
        raise ValueError('Frozen selection changed')
    prepared = Path(batch['preparation_run'])
    verification = validate_run(prepared)
    manifest = json.loads((prepared/'manifest.json').read_text())
    selected = set(batch['track_ids'])
    rows = [r for r in manifest['records'] if r['dataset'] == 'jamendo']
    if any(r['source_id'] not in selected or r['split'] != 'train' or r['training_eligible'] for r in rows):
        raise ValueError('Preparation contains a different sample or unsafe eligibility')
    for track in batch['tracks']:
        if track['status'] != 'complete' or checksum(track['receipt_path']) != track['receipt_sha256']:
            raise ValueError('Missing or altered separation receipt')
    root = component/'runs/jamendo_yield'/uuid.uuid4().hex
    root.mkdir(parents=True)
    policy = AuditPolicy()
    evidence = dict(run_status='running', data_status='exploratory_not_research',
                    batch_path=str(batch_root), batch_sha256=checksum(batch_root/'batch.json'),
                    preparation_path=str(prepared), manifest_sha256=checksum(prepared/'manifest.json'),
                    preparation_verification=verification, policy=asdict(policy), probes=[], training_ready_examples=0,
                    limitations=['Five selected training artists are not a population benchmark',
                                 'Artist identity is a singer proxy; independent singer identity is unverified',
                                 'Synthetic realized-shift checks do not approve the original performance or model notes',
                                 'Reference masks use model estimates; musical review and leakage checks remain pending'])
    evidence['probe_code_sha256'] = {name: checksum(module_source(name+'.py'))
                                     for name in ('jamendo_yield', 'coverage', 'interventions', 'analysis')}
    for name in evidence['probe_code_sha256']:
        target = root/'source_code_at_run'
        target.mkdir(exist_ok=True)
        (target/(name+'.py')).write_bytes((module_source(name+'.py')).read_bytes())
    def save():
        write_json(root/'evidence.json', evidence)
    save()
    print('YIELD_RUN:', root, flush=True)
    try:
        for row in rows:
            probe = dict(record_id=row['record_id'], source_id=row['source_id'], group_id=row['group_id'],
                         screen_status=row['screen']['status'], shifts={}, passed_shifts=0,
                         training_eligible=False, support_basis='provisional model-estimated note/PESTO/YIN mask')
            evidence['probes'].append(probe)
            if row['screen']['status'] != 'review_candidate':
                probe['skip_reason'] = 'Withheld by unchanged preparation screen'
                save()
                continue
            with np.load(row['screen_masks']['path']) as archive:
                times, reliable = archive['times_seconds'], archive['reliable_mask']
            geometry = support_record(times, reliable, policy)
            probe['geometry'] = geometry
            if not geometry['eligible']:
                save()
                continue
            audio, rate = sf.read(row['accepted_audio_path'], dtype='float32')
            before, analyzer = extract(audio, rate, component/'checkpoints')
            if not np.array_equal(times, before['times_seconds']):
                raise ValueError('Probe and saved preparation grids differ')
            yin = independent_pitch(audio, rate)
            support = geometry['support_seconds']
            identity, _ = render_local(audio, rate, 0., support, policy)
            repeated, _ = extract(identity, rate, component/'checkpoints')
            identity_pass = bool(np.array_equal(identity, audio) and
                                 np.allclose(before['f0_hz'], repeated['f0_hz'], equal_nan=True, atol=1e-6, rtol=0))
            probe.update(identity_pass=identity_pass, analyzer=analyzer, accepted_sha256=checksum(row['accepted_audio_path']))
            folder = root/'probes'/row['record_id']
            folder.mkdir(parents=True)
            for cents in policy.shifts_cents:
                print('PROBE:', row['record_id'], cents, flush=True)
                shifted, blend = render_local(audio, rate, cents, support, policy)
                after, _ = extract(shifted, rate, component/'checkpoints')
                measured, _, _ = audit_shift(before, after, yin, independent_pitch(shifted, rate), reliable, support, cents, policy)
                measured.update(identity_pass=identity_pass, same_sample_count=len(shifted) == len(audio),
                                outside_support_exact=bool(np.array_equal(shifted[blend == 0], audio[blend == 0])))
                measured['passed'] = bool(measured['passed'] and identity_pass and measured['same_sample_count'] and measured['outside_support_exact'])
                path = folder/f'shift_{cents:+g}.wav'
                sf.write(path, shifted, rate, subtype='FLOAT')
                measured['artifact'] = dict(path=str(path), sha256=checksum(path))
                probe['shifts'][f'{cents:+g}'] = measured
                probe['passed_shifts'] = sum(s['passed'] for s in probe['shifts'].values())
                save()
        candidates = [r for r in rows if r['screen']['status'] == 'review_candidate']
        potential = [p for p in evidence['probes'] if p.get('geometry', {}).get('eligible')]
        any_pass = [p for p in potential if p['passed_shifts'] > 0]
        all_pass = [p for p in potential if p['passed_shifts'] == len(policy.shifts_cents)]
        evidence['summary'] = dict(selected_songs=5, selected_artists=5, prepared=counts(rows),
                                   review_candidates=counts(candidates), geometry_eligible=counts(potential),
                                   any_audited_shift=counts(any_pass), all_six_audited_shifts=counts(all_pass),
                                   audited_variants=sum(p['passed_shifts'] for p in potential),
                                   technical_statuses=dict(Counter(r['technical_result']['status'] for r in rows)),
                                   pending_jamendo_review_rows=2*len(candidates), training_ready_examples=0)
        validate_run(prepared)
        if checksum(batch_root/'batch.json') != evidence['batch_sha256']:
            raise ValueError('Batch changed during yield measurement')
        if any(checksum(module_source(name+'.py')) != digest for name, digest in evidence['probe_code_sha256'].items()):
            raise ValueError('Probe implementation changed during measurement')
        evidence['run_status'] = 'complete'
        save()
        print(json.dumps(evidence['summary'], indent=2), flush=True)
    except BaseException as exc:
        evidence.update(run_status='failed', failure=dict(type=type(exc).__name__, reason=str(exc)))
        save()
        raise
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    run(args.batch)


if __name__ == '__main__':
    main()
