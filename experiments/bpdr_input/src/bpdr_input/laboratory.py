"""Phase 2 commands: immutable source/crop manifest, review gate and four-view pilot."""
import argparse
import csv
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from .dataset import DataPolicy, build_dataset, dump, reference_mask, reviewed_intervals, verify_manifest
from .pipeline import process, checksum
from .analysis import extract
from .interventions import (AuditPolicy, independent_pitch, select_support, render_local, audit_shift,
                            nominal_boundaries, boundary_track, corrupt_boundaries, longest_span)


SOURCE_MODULES = ('laboratory', 'interventions', 'dataset', 'analysis', 'pipeline', 'contracts')


def source_code_sha256():
    module_dir = Path(__file__).resolve().parent
    return {name: checksum(module_dir/(name+'.py')) for name in SOURCE_MODULES}


def _read_npz(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _rank(phrase, policy):
    times = np.arange(0, phrase['duration_seconds'], .02)
    f0 = np.loadtxt(phrase['f0_path'], delimiter=',', ndmin=2)
    notes = json.loads(Path(phrase['notes_path']).read_text())
    safe, _ = reference_mask(times, f0, notes, policy)
    span = longest_span(safe)
    return span[1]-span[0] if span else 0


def run_interventions(directory, exploratory=False, limit=2):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest_digest = checksum(directory/'manifest.json')
    verify_manifest(manifest)
    data_policy = DataPolicy(**manifest['policy'])
    policy = AuditPolicy()
    source_hashes = source_code_sha256()
    with (directory/'reviews.csv').open(encoding='utf-8-sig', newline='') as stream:
        reviews = list(csv.DictReader(stream))
    review_digest = checksum(directory/'reviews.csv')
    approvals = []
    for phrase in manifest['phrases']:
        intervals, state = reviewed_intervals(phrase, reviews)
        approvals.append(dict(phrase_id=phrase['phrase_id'], status=state, approved_intervals_seconds=intervals))
    approval_map = {r['phrase_id']: r for r in approvals}
    candidates = [p for p in manifest['phrases'] if p['split'] == 'train' and
                  (exploratory or approval_map[p['phrase_id']]['status'] == 'approved')]
    candidates.sort(key=lambda p: (-_rank(p, data_policy), p['phrase_id']))
    experiment_root = directory/('exploratory' if exploratory else 'reviewed')/uuid.uuid4().hex
    experiment_root.mkdir(parents=True)
    dump(experiment_root/'locked_audit_policy.json', policy.__dict__)
    records, exclusions = [], []
    component = Path(__file__).resolve().parents[2]
    successful_phrases = 0
    selected_singers = set()
    for phrase in candidates:
        if successful_phrases >= limit:
            break
        if phrase['singer_id'] in selected_singers:
            continue
        pid = phrase['phrase_id']
        pair_count_before = len(records)
        try:
            if checksum(phrase['audio_path']) != phrase['sha256']:
                raise ValueError('Crop changed after manifest/review creation')
            result, run = process(phrase['audio_path'], experiment_root/'analysis', component/'checkpoints')
            if result.accepted_audio is None:
                raise ValueError(f'Baseline analysis stopped: {result.status.value} {result.reason_codes}')
            audio, sr = sf.read(result.accepted_audio.path, dtype='float32')
            # Analyze the decoded export itself, so identity and candidate share its encoding.
            before, baseline_metadata = extract(audio, sr, component/'checkpoints')
            times = before['times_seconds']
            f0 = np.loadtxt(phrase['f0_path'], delimiter=',', ndmin=2)
            notes = json.loads(Path(phrase['notes_path']).read_text())
            reliable, reference = reference_mask(times, f0, notes, data_policy)
            yin_before = independent_pitch(audio, sr)
            reliable &= before['voiced_mask'] & (reference > 0) & (yin_before > 0)
            candidates_mask = reliable.copy()
            reliable[candidates_mask] &= abs(1200*np.log2(yin_before[candidates_mask]/before['f0_hz'][candidates_mask])) <= policy.tracker_agreement_cents
            # Human-approved support is mandatory in reviewed runs; no empty-review bypass.
            if not exploratory:
                human = np.zeros(len(times), dtype=bool)
                for start, end in approval_map[pid]['approved_intervals_seconds']:
                    human |= (times >= start) & (times < end)
                reliable &= human
            support = select_support(times, reliable, policy)
            phrase_out = experiment_root/pid
            phrase_out.mkdir()
            identity, _ = render_local(audio, sr, 0., support, policy)
            identity_features, _ = extract(identity, sr, component/'checkpoints')
            same_pitch = np.allclose(before['f0_hz'], identity_features['f0_hz'], equal_nan=True, atol=1e-6, rtol=0)
            identity_audit = dict(sample_count_preserved=len(identity) == len(audio),
                                  exact_samples_preserved=bool(np.array_equal(identity, audio)),
                                  finite=bool(np.isfinite(identity).all()), no_clipping=bool(abs(identity).max() < 1),
                                  measured_pitch_preserved=bool(same_pitch), method='exact passthrough identity')
            dump(phrase_out/'identity_audit.json', identity_audit)
            if not all(identity_audit[k] for k in ('sample_count_preserved', 'exact_samples_preserved', 'finite', 'no_clipping', 'measured_pitch_preserved')):
                raise ValueError('Identity audit failed')
            sf.write(phrase_out/'original.wav', audio, sr, subtype='FLOAT')
            np.savez_compressed(phrase_out/'original_acoustics.npz', **before)
            endpoints = nominal_boundaries(times, notes)
            nominal = boundary_track(times, endpoints)
            masks = dict(reliable_mask=reliable, reference_f0_hz=reference, times_seconds=times)
            np.savez_compressed(phrase_out/'reference_masks_NOT_MODEL_INPUT.npz', **masks)
            dump(phrase_out/'baseline_analyzer.json', baseline_metadata)
            for cents in policy.shifts_cents:
                shift_out = phrase_out/f'shift_{cents:+g}'
                shift_out.mkdir()
                try:
                    detuned, blend = render_local(audio, sr, cents, support, policy)
                    sf.write(shift_out/'detuned.wav', detuned, sr, subtype='FLOAT')
                    after, metadata = extract(detuned, sr, component/'checkpoints')
                    yin_after = independent_pitch(detuned, sr)
                    audit, anchors, target = audit_shift(before, after, yin_before, yin_after, reliable, support, cents, policy)
                    sample_mask = blend > 0
                    audit['same_sample_count'] = len(detuned) == len(audio)
                    audit['outside_support_exact_samples'] = bool(np.array_equal(detuned[~sample_mask], audio[~sample_mask]))
                    audit['passed'] &= audit['same_sample_count'] and audit['outside_support_exact_samples']
                    dump(shift_out/'audit.json', audit)
                    dump(shift_out/'analyzer.json', metadata)
                    np.savez_compressed(shift_out/'detuned_acoustics.npz', **after)
                    # These remain private targets. Crossfade cents are not treated as known.
                    np.savez_compressed(shift_out/'targets_NOT_MODEL_INPUT.npz', anchor_mask=anchors,
                                        common_reliable_mask=anchors, detuned_plateau_mask=anchors & (target != 0),
                                        zero_anchor_mask=anchors & (target == 0), correction_target_cents=target, times_seconds=times)
                    for kind in ('split', 'merge', 'jitter'):
                        altered = corrupt_boundaries(endpoints, times, kind, data_policy.split_seed, split_time=sum(support)/2)
                        corrupt = boundary_track(times, altered)
                        views_dir = shift_out/kind
                        views_dir.mkdir()
                        for name, acoustics, track in [('A', before, nominal), ('B', before, corrupt), ('C', after, nominal), ('D', after, corrupt)]:
                            # Only runtime-like acoustics and timing-only boundary metadata are inputs.
                            np.savez_compressed(views_dir/(name+'_input.npz'), times_seconds=times,
                                                frame_features_70=acoustics['frame_features_70'],
                                                boundary_features_71=np.column_stack([acoustics['frame_features_70'], track]).astype(np.float32))
                        record = dict(phrase_id=pid, source_id=phrase['source_id'], singer_id=phrase['singer_id'], split=phrase['split'],
                                      cents=cents, corruption=kind, support_seconds=support, audit_passed=audit['passed'],
                                      review_status=approval_map[pid]['status'], exploratory=exploratory,
                                      training_eligible=bool(not exploratory and audit['passed'] and approval_map[pid]['status'] == 'approved'),
                                      waveform_paths={'A': str(phrase_out/'original.wav'), 'B': str(phrase_out/'original.wav'),
                                                      'C': str(shift_out/'detuned.wav'), 'D': str(shift_out/'detuned.wav')},
                                      waveform_sha256={'A': checksum(phrase_out/'original.wav'), 'B': checksum(phrase_out/'original.wav'),
                                                       'C': checksum(shift_out/'detuned.wav'), 'D': checksum(shift_out/'detuned.wav')},
                                      input_directory=str(views_dir), target_path=str(shift_out/'targets_NOT_MODEL_INPUT.npz'),
                                      input_sha256={name: checksum(views_dir/(name+'_input.npz')) for name in ('A', 'B', 'C', 'D')},
                                      manifest_path=str(directory/'manifest.json'), manifest_sha256=manifest_digest,
                                      crop_sha256=phrase['sha256'],
                                      audit_policy_path=str(experiment_root/'locked_audit_policy.json'),
                                      audit_policy_sha256=checksum(experiment_root/'locked_audit_policy.json'),
                                      review_file_path=str(directory/'reviews.csv'), review_file_sha256=review_digest,
                                      targets_sha256=checksum(shift_out/'targets_NOT_MODEL_INPUT.npz'), audit_sha256=checksum(shift_out/'audit.json'),
                                      common_mask_path=str(shift_out/'targets_NOT_MODEL_INPUT.npz'),
                                      reference_mask_path=str(phrase_out/'reference_masks_NOT_MODEL_INPUT.npz'), audit_path=str(shift_out/'audit.json'),
                                      nominal_boundary_seconds=endpoints.tolist(), corrupted_boundary_seconds=altered.tolist(),
                                      boundary_provenance='A1 onset and offset timings; controlled laboratory input, not runtime note inference')
                        validate_pair_record(record)
                        dump(views_dir/'pair_manifest.json', record)
                        records.append(record)
                except (ValueError, RuntimeError) as exc:
                    exclusions.append(dict(phrase_id=pid, cents=cents, reason=str(exc)))
        except (ValueError, RuntimeError) as exc:
            exclusions.append(dict(phrase_id=pid, reason=str(exc)))
        if any(record['audit_passed'] for record in records[pair_count_before:]):
            successful_phrases += 1
            selected_singers.add(phrase['singer_id'])
    if source_code_sha256() != source_hashes:
        raise RuntimeError('Component source code changed while generating the experiment')
    if checksum(directory/'manifest.json') != manifest_digest or checksum(directory/'reviews.csv') != review_digest:
        raise RuntimeError('Source manifest or review sheet changed while generating the experiment')
    summary = dict(mode='exploratory_not_training_ready' if exploratory else 'review_gated',
                   independent_phrase_count=successful_phrases,
                   independent_singer_count=len({r['singer_id'] for r in records if r['audit_passed']}),
                   paired_examples=len(records), audited_pass_pairs=sum(r['audit_passed'] for r in records),
                   training_eligible_pairs=sum(r['training_eligible'] for r in records), exclusions=exclusions,
                   source_code_sha256=source_hashes, data_manifest_sha256=manifest_digest,
                   review_sheet_sha256=review_digest,
                   untouched_partitions=['development', 'pp1_evaluation', 'final_holdout'],
                   limitations=['No model training/novelty result', 'Human reviews pending unless recorded',
                                'Crossfade ramp targets excluded', 'Repeated views are not independent samples'],
                   examples=records, review_states=approvals)
    dump(experiment_root/'summary.json', summary)
    dump(directory/('latest_exploratory.json' if exploratory else 'latest_reviewed.json'), {'path': str(experiment_root)})
    return experiment_root, summary


def validate_pair_record(record):
    streams = {name: _read_npz(Path(record['input_directory'])/(name+'_input.npz')) for name in ('A', 'B', 'C', 'D')}
    for values in streams.values():
        if set(values) != {'times_seconds', 'frame_features_70', 'boundary_features_71'}:
            raise ValueError('Reference or intervention target leaked into model inputs')
        n = len(values['times_seconds'])
        if values['frame_features_70'].shape != (n, 70) or values['boundary_features_71'].shape != (n, 71) or not np.isfinite(values['boundary_features_71']).all():
            raise ValueError('Invalid model input shape or values')
        if not np.array_equal(values['frame_features_70'], values['boundary_features_71'][:, :70]):
            raise ValueError('70/71 acoustic feature mismatch')
    for a, b in [('A', 'B'), ('C', 'D')]:
        if not np.array_equal(streams[a]['frame_features_70'], streams[b]['frame_features_70']):
            raise ValueError('Boundary-only pair changed acoustics')
        if record['waveform_sha256'][a] != record['waveform_sha256'][b]:
            raise ValueError('Boundary-only pair changed waveform')
    for a, b in [('A', 'C'), ('B', 'D')]:
        if not np.array_equal(streams[a]['boundary_features_71'][:, -1], streams[b]['boundary_features_71'][:, -1]):
            raise ValueError('Paired detuning changed boundary metadata')
    for values in streams.values():
        if not np.array_equal(values['times_seconds'], streams['A']['times_seconds']):
            raise ValueError('Four-view frame grids differ')
    if np.array_equal(streams['A']['boundary_features_71'][:, -1], streams['B']['boundary_features_71'][:, -1]):
        raise ValueError('Boundary corruption made no observable change')
    if np.array_equal(streams['A']['frame_features_70'], streams['C']['frame_features_70']):
        raise ValueError('Detuned acoustics were not recomputed')
    targets = _read_npz(record['target_path'])
    common = targets['common_reliable_mask']
    if common.shape != streams['A']['times_seconds'].shape or not np.array_equal(common, targets['anchor_mask']):
        raise ValueError('Paired reliable/anchor masks differ')
    voiced = (streams['A']['frame_features_70'][:,66] > .5) & (streams['C']['frame_features_70'][:,66] > .5)
    if np.any(common & ~voiced):
        raise ValueError('Common target mask contains unreliable voicing')
    if not np.isfinite(targets['correction_target_cents']).all():
        raise ValueError('Invalid correction target')
    if not np.all(targets['correction_target_cents'][targets['detuned_plateau_mask']] == -record['cents']) or not np.all(targets['correction_target_cents'][targets['zero_anchor_mask']] == 0):
        raise ValueError('Correction target sign or zero anchor is inconsistent')
    return True


def require_training_example(record):
    """Loader gate: refuse pilots, failed audits or missing human approval."""
    if not record.get('training_eligible') or record.get('exploratory') or not record.get('audit_passed') or record.get('review_status') != 'approved':
        raise ValueError('Example is not reviewed and audit-approved for training')
    if record.get('split') != 'train':
        raise ValueError('Only the training partition may feed model fitting')
    if checksum(record['manifest_path']) != record['manifest_sha256']:
        raise ValueError('Source/crop manifest changed after example generation')
    manifest = json.loads(Path(record['manifest_path']).read_text())
    verify_manifest(manifest)
    phrase = next((p for p in manifest['phrases'] if p['phrase_id'] == record['phrase_id']), None)
    if phrase is None or any(phrase[key] != record[key] for key in ('source_id', 'singer_id', 'split')) or phrase['sha256'] != record['crop_sha256']:
        raise ValueError('Example does not match a training crop in the source manifest')
    data_run = Path(record['manifest_path']).parent.resolve()
    if Path(record['review_file_path']).resolve() != data_run/'reviews.csv':
        raise ValueError('Review sheet is not attached to the source manifest')
    if checksum(record['review_file_path']) != record['review_file_sha256']:
        raise ValueError('Review sheet changed after example generation')
    with Path(record['review_file_path']).open(encoding='utf-8-sig', newline='') as stream:
        approved, state = reviewed_intervals(phrase, list(csv.DictReader(stream)))
    if state != 'approved':
        raise ValueError('Two current, independent approvals are required')
    start, end = record['support_seconds']
    if not any(review_start <= start < end <= review_end for review_start, review_end in approved):
        raise ValueError('Intervention extends outside the two-reviewer-approved interval')
    if checksum(record['audit_policy_path']) != record['audit_policy_sha256']:
        raise ValueError('Locked audit policy changed after example generation')
    policy = json.loads(Path(record['audit_policy_path']).read_text())
    if record['cents'] not in policy['shifts_cents']:
        raise ValueError('Shift is not in the locked intervention policy')
    for name, path in record['waveform_paths'].items():
        if checksum(path) != record['waveform_sha256'][name]:
            raise ValueError('Paired waveform changed after audit')
    for path_key, hash_key in [('review_file_path', 'review_file_sha256'), ('target_path', 'targets_sha256'), ('audit_path', 'audit_sha256')]:
        if checksum(record[path_key]) != record[hash_key]:
            raise ValueError('Review, target or audit changed after approval')
    if record['waveform_sha256']['A'] != record['waveform_sha256']['B'] or record['waveform_sha256']['C'] != record['waveform_sha256']['D']:
        raise ValueError('Four-view waveform identity failed')
    if not json.loads(Path(record['audit_path']).read_text()).get('passed'):
        raise ValueError('Saved renderer audit did not pass')
    audit = json.loads(Path(record['audit_path']).read_text())
    if audit['policy'] != policy:
        raise ValueError('Renderer audit used a different policy')
    if any(checksum(Path(record['input_directory'])/(name+'_input.npz')) != record['input_sha256'][name] for name in ('A', 'B', 'C', 'D')):
        raise ValueError('Model inputs changed after audit')
    validate_pair_record(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--dataset-root', type=Path, required=True)
    component = Path(__file__).resolve().parents[2]
    build.add_argument('--output-root', type=Path, default=component/'runs'/'phase2')
    pilot = commands.add_parser('interventions')
    pilot.add_argument('--data-run', type=Path, required=True)
    pilot.add_argument('--exploratory', action='store_true')
    pilot.add_argument('--limit', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'build':
        run = args.output_root.resolve()/uuid.uuid4().hex
        manifest = build_dataset(args.dataset_root, run)
        dump(args.output_root/'latest.json', {'path': str(run)})
        print((run/'data_summary.json').read_text())
        print('DATA_RUN:', run)
        return 0 if not manifest['source_errors'] else 1
    if args.limit < 1:
        parser.error('--limit must be positive')
    run, summary = run_interventions(args.data_run, args.exploratory, args.limit)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('examples', 'review_states')}, indent=2))
    print('EXPERIMENT_RUN:', run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
