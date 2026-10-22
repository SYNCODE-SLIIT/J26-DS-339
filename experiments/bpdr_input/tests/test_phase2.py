import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from bpdr_input.dataset import (singer_splits, crop_ranges, crop_notes, reference_mask,
                               validate_annotations, reviewed_intervals, build_dataset, DataPolicy, verify_manifest)
from bpdr_input.interventions import (render_local, envelope, corrupt_boundaries, boundary_track,
                                     audit_shift, AuditPolicy, independent_pitch, nominal_boundaries)
from bpdr_input.laboratory import require_training_example
from bpdr_input.pipeline import checksum


class Phase2DataTests(unittest.TestCase):
    def test_split_is_stable_and_singer_disjoint(self):
        singers = ['S'+str(i) for i in range(29)]
        a = singer_splits(singers)
        self.assertEqual(a, singer_splits(list(reversed(singers))+singers))
        self.assertEqual(set(a.values()), {'train', 'development', 'pp1_evaluation', 'final_holdout'})
        self.assertEqual(sum(v == 'train' for v in a.values()), 14)

    def test_bounds_order_nonfinite_and_overlap(self):
        grid = np.arange(0, 1., .01)
        f0 = np.column_stack([grid, np.full(len(grid), 220.)])
        notes = {'A1': np.array([[0., 220., .5], [.3, 220., .5]]), 'A2': np.array([[0., 220., .5]])}
        errors, warnings = validate_annotations(f0, notes, 1., 1/22050)
        self.assertFalse(errors)
        self.assertIn('A1_OVERLAPPING_NOTES_MASKED', warnings)
        errors, _ = validate_annotations(f0[::-1], notes, .4, 1/22050)
        self.assertIn('F0_TIMESTAMPS_INVALID', errors)
        self.assertIn('A1_VALUES_OR_BOUNDS_INVALID', errors)

    def test_missing_f0_coverage_cannot_extend_reference(self):
        f0 = np.array([[0., 220.], [.01, 220.], [.02, 220.]])
        notes = {'A1': np.array([[0., 220., 1.]]), 'A2': np.array([[0., 220., 1.]])}
        errors, _ = validate_annotations(f0, notes, 1., 1/22050)
        self.assertIn('F0_COVERAGE_INCOMPLETE', errors)
        note = dict(start_seconds=0., duration_seconds=1., pitch_hz=220., partial=False)
        mask, _ = reference_mask(np.arange(0, 1, .01), f0, {'A1': [note], 'A2': [note.copy()]})
        self.assertFalse(mask[np.arange(0, 1, .01) > .04].any())

    def test_crops_cover_source_and_flag_partial_notes(self):
        values = np.array([[0., 220., 44.]])
        ranges = crop_ranges(44000, 1000, {'A1': values, 'A2': values})
        self.assertEqual(ranges, [(0, 20000), (20000, 40000), (40000, 44000)])
        middle = crop_notes(values, 20, 40)
        self.assertTrue(middle[0]['partial'])
        self.assertEqual(middle[0]['start_seconds'], 0)
        self.assertEqual(middle[0]['duration_seconds'], 20)

    def test_reference_mask_protects_disagreement_partial_and_unvoiced(self):
        times = np.arange(0, 1, .02)
        f0 = np.column_stack([times, np.where((times>.4)&(times<.6), 0, 220)])
        note = dict(start_seconds=0., duration_seconds=1., pitch_hz=220., partial=False)
        notes = {'A1': [note], 'A2': [note.copy()]}
        safe, ref = reference_mask(times, f0, notes)
        self.assertFalse(safe[times < .1].any())
        self.assertFalse(safe[ref == 0].any())
        notes['A2'][0]['pitch_hz'] = 440
        self.assertFalse(reference_mask(times, f0, notes)[0].any())
        notes['A2'][0]['pitch_hz'] = 220
        notes['A1'][0]['partial'] = True
        self.assertFalse(reference_mask(times, f0, notes)[0].any())

    def test_review_gate_two_people_intersection_and_stale_hash(self):
        phrase = dict(phrase_id='p', sha256='abc', duration_seconds=2.)
        def row(reviewer, interval):
            return dict(phrase_id='p', crop_sha256='abc', reviewer_id=reviewer, musical_experience='singing',
                        decision='approve', approved_intervals_json=json.dumps(interval), reason='listened and checked', reviewed_at_utc='2026-09-27T00:00:00Z')
        rows = [row('r1', [[0., 1.5]]), row('r2', [[.5, 2.]])]
        self.assertEqual(reviewed_intervals(phrase, rows), ([[.5, 1.5]], 'approved'))
        self.assertFalse(reviewed_intervals(phrase, rows[:1])[0])
        rows[1]['reviewer_id'] = 'r1'
        self.assertFalse(reviewed_intervals(phrase, rows)[0])
        rows[1]['reviewer_id'] = 'r2'
        rows[1]['crop_sha256'] = 'stale'
        self.assertFalse(reviewed_intervals(phrase, rows)[0])
        rows[1]['crop_sha256'] = 'abc'
        rows[1]['decision'] = 'reject'
        self.assertFalse(reviewed_intervals(phrase, rows)[0])
        rows[1]['decision'] = 'approve'
        rows[1]['reviewed_at_utc'] = 'not a UTC timestamp'
        self.assertEqual(reviewed_intervals(phrase, rows)[1], 'review_timestamp_invalid')

    def test_generated_crop_offset_and_pending_reviews(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw'
            for folder in ['Audio', 'Annotations/F0', 'Annotations/Notes']:
                (root/folder).mkdir(parents=True)
            with (root/'vocadito_metadata.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['track_id', 'singer_id', 'language'])
                writer.writeheader()
                for i in range(4):
                    writer.writerow(dict(track_id=str(i), singer_id='S'+str(i), language='test'))
                    sr = 1000
                    audio = np.arange(25000, dtype='float32')/100000
                    sf.write(root/'Audio'/f'vocadito_{i}.wav', audio, sr, subtype='FLOAT')
                    f0 = np.column_stack([np.arange(0,25,.02), np.full(1250,220.)])
                    np.savetxt(root/'Annotations/F0'/f'vocadito_{i}_f0.csv', f0, delimiter=',')
                    for n in ('A1', 'A2'):
                        np.savetxt(root/'Annotations/Notes'/f'vocadito_{i}_notes{n}.csv', [[0,220,25]], delimiter=',')
            manifest = build_dataset(root, Path(tmp)/'generated')
            self.assertEqual(len(manifest['phrases']), 8)
            phrase = manifest['phrases'][1]
            self.assertEqual(phrase['original_offset_seconds'], 20)
            cropped, _ = sf.read(phrase['audio_path'], dtype='float32')
            np.testing.assert_array_equal(cropped, audio[20000:])
            f0 = np.loadtxt(phrase['f0_path'], delimiter=',')
            self.assertAlmostEqual(f0[0,0], 0)
            self.assertEqual(phrase['review_status'], 'pending')
            self.assertTrue(verify_manifest(manifest))
            Path(phrase['f0_path']).write_text('0,100\n')
            with self.assertRaisesRegex(ValueError, 'references changed'):
                verify_manifest(manifest)


class Phase2InterventionTests(unittest.TestCase):
    def test_identity_and_local_samples(self):
        sr = 22050
        audio = (.2*np.sin(2*np.pi*220*np.arange(sr*2)/sr)).astype('float32')
        identity, _ = render_local(audio, sr, 0, (.5,1.5))
        np.testing.assert_array_equal(identity, audio)
        shifted, weight = render_local(audio, sr, 20, (.5,1.5))
        np.testing.assert_array_equal(shifted[weight==0], audio[weight==0])
        self.assertEqual(len(shifted),len(audio))
        self.assertTrue(np.isfinite(shifted).all())

    def test_split_merge_jitter_and_input_identity(self):
        times = np.arange(0,2,.02)
        endpoints = np.array([.2,.6,1.,1.8])
        split = corrupt_boundaries(endpoints, times, 'split')
        merge = corrupt_boundaries(endpoints, times, 'merge')
        jitter = corrupt_boundaries(endpoints, times, 'jitter')
        self.assertEqual(len(split),len(endpoints)+1)
        self.assertEqual(len(merge),len(endpoints)-1)
        np.testing.assert_array_equal(jitter, corrupt_boundaries(endpoints,times,'jitter'))
        self.assertFalse(np.array_equal(boundary_track(times,endpoints),boundary_track(times,split)))

    def test_nominal_boundaries_include_note_offsets_before_rests(self):
        times = np.arange(0, 2, .02)
        notes = {'A1': [dict(start_seconds=.2, duration_seconds=.3),
                        dict(start_seconds=.8, duration_seconds=.4)]}
        np.testing.assert_allclose(nominal_boundaries(times, notes), [.2, .5, .8, 1.2])

    def test_real_renderer_resolves_ten_cent_tone_shift(self):
        sr = 22050
        audio = (.2*np.sin(2*np.pi*220*np.arange(sr*2)/sr)).astype('float32')
        shifted, _ = render_local(audio, sr, 10, (.5,1.5))
        before, after = independent_pitch(audio,sr), independent_pitch(shifted,sr)
        times = np.arange(len(before))*.02
        mask = (times>.8)&(times<1.2)
        realized = 1200*np.log2(after[mask]/before[mask])
        self.assertLess(abs(np.median(realized)-10), 3)

    def test_audit_wrong_shift_and_missing_pitch_rejected(self):
        times = np.arange(0,2,.02)
        original = np.full(len(times),220.)
        changed = original.copy()
        changed[(times>=.5)&(times<=1.5)] *= 2**(20/1200)
        base = dict(times_seconds=times,f0_hz=original,voiced_mask=np.ones(len(times),dtype=bool))
        after = dict(times_seconds=times,f0_hz=changed,voiced_mask=np.ones(len(times),dtype=bool))
        reliable = np.ones(len(times),dtype=bool)
        audit,mask,target = audit_shift(base,after,original,changed,reliable,(.5,1.5),20)
        self.assertTrue(audit['passed'])
        self.assertTrue((target[mask & (times>.7)&(times<1.3)] == -20).all())
        self.assertFalse(mask[(times>=.5)&(times<.68)].any())
        failed,mask,_ = audit_shift(base,base,original,original,reliable,(.5,1.5),20)
        self.assertFalse(failed['passed'])
        self.assertFalse(mask.any())
        after['f0_hz'] = np.full(len(times),np.nan)
        self.assertFalse(audit_shift(base,after,original,changed,reliable,(.5,1.5),20)[0]['passed'])

    def test_audit_requires_coverage_and_rejects_tail_outside_drift(self):
        times = np.arange(0, 2, .02)
        original = np.full(len(times), 220.)
        plateau = (times >= .68) & (times <= 1.32)
        changed = original.copy()
        changed[plateau] *= 2**(20/1200)
        base = dict(times_seconds=times, f0_hz=original, voiced_mask=np.ones(len(times), dtype=bool))
        after = dict(times_seconds=times, f0_hz=changed, voiced_mask=np.ones(len(times), dtype=bool))
        reliable = np.ones(len(times), dtype=bool)
        sparse = dict(after, voiced_mask=np.ones(len(times), dtype=bool))
        sparse['voiced_mask'][np.flatnonzero(plateau)[::2]] = False
        audit, _, _ = audit_shift(base, sparse, original, changed, reliable, (.5, 1.5), 20)
        self.assertFalse(audit['passed'])
        self.assertLess(audit['plateau_coverage_fraction'], AuditPolicy().min_plateau_coverage_fraction)
        outside = (times < .4) | (times > 1.6)
        shifted_outside = changed.copy()
        shifted_outside[np.flatnonzero(outside)[:8]] *= 2**(30/1200)
        drifted = dict(after, f0_hz=shifted_outside)
        audit, _, _ = audit_shift(base, drifted, original, shifted_outside, reliable, (.5, 1.5), 20)
        self.assertFalse(audit['passed'])
        self.assertLessEqual(audit['outside_median_drift_cents'], AuditPolicy().outside_pitch_drift_tolerance_cents)
        self.assertGreater(audit['outside_p90_drift_cents'], AuditPolicy().outside_p90_pitch_drift_tolerance_cents)

    def test_exploratory_or_failed_examples_never_training_ready(self):
        for row in [dict(training_eligible=False), dict(training_eligible=True,exploratory=True),
                    dict(training_eligible=True, exploratory=False, audit_passed=False,review_status='approved')]:
            with self.assertRaises(ValueError):
                require_training_example(row)

    def test_loader_rechecks_reviews_and_model_input_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root/'manifest.json'
            phrase = dict(phrase_id='p', source_id='s', singer_id='singer', split='train',
                          sha256='crop-hash', duration_seconds=2.)
            manifest_path.write_text(json.dumps({'phrases': [phrase]}))
            review_path = root/'reviews.csv'
            with review_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['phrase_id', 'crop_sha256', 'reviewer_id',
                    'musical_experience', 'decision', 'approved_intervals_json', 'reason', 'reviewed_at_utc'])
                writer.writeheader()
                for reviewer in ('r1', 'r2'):
                    writer.writerow(dict(phrase_id='p', crop_sha256='crop-hash', reviewer_id=reviewer,
                        musical_experience='singer', decision='pending', approved_intervals_json='[]',
                        reason='', reviewed_at_utc=''))
            policy_path = root/'policy.json'
            policy_path.write_text(json.dumps(AuditPolicy().__dict__))
            audit_path = root/'audit.json'
            audit_path.write_text(json.dumps(dict(passed=True, policy=AuditPolicy().__dict__)))
            inputs = root/'inputs'
            inputs.mkdir()
            times = np.arange(100)*.02
            base = np.zeros((100, 70), dtype=np.float32)
            base[:, 66] = 1
            changed = base.copy()
            changed[:, 0] = 1
            nominal = np.zeros(100, dtype=np.float32)
            corrupt = nominal.copy()
            corrupt[50] = 1
            for name, frame, boundary in [('A', base, nominal), ('B', base, corrupt),
                                          ('C', changed, nominal), ('D', changed, corrupt)]:
                np.savez_compressed(inputs/(name+'_input.npz'), times_seconds=times,
                    frame_features_70=frame, boundary_features_71=np.column_stack([frame, boundary]))
            target_path = root/'targets.npz'
            plateau = np.zeros(100, dtype=bool)
            plateau[20:30] = True
            target = np.zeros(100)
            target[plateau] = -20
            np.savez_compressed(target_path, common_reliable_mask=np.ones(100, dtype=bool),
                anchor_mask=np.ones(100, dtype=bool), detuned_plateau_mask=plateau,
                zero_anchor_mask=~plateau, correction_target_cents=target)
            original = root/'original.wav'
            detuned = root/'detuned.wav'
            sf.write(original, np.zeros(1000), 1000)
            sf.write(detuned, np.ones(1000)*.01, 1000)
            record = dict(training_eligible=True, exploratory=False, audit_passed=True,
                review_status='approved', split='train', phrase_id='p', source_id='s', singer_id='singer',
                crop_sha256='crop-hash', manifest_path=str(manifest_path), manifest_sha256=checksum(manifest_path),
                review_file_path=str(review_path), review_file_sha256=checksum(review_path),
                support_seconds=[.3, 1.2], audit_policy_path=str(policy_path),
                audit_policy_sha256=checksum(policy_path), cents=20,
                waveform_paths={'A': str(original), 'B': str(original), 'C': str(detuned), 'D': str(detuned)},
                waveform_sha256={'A': checksum(original), 'B': checksum(original),
                    'C': checksum(detuned), 'D': checksum(detuned)},
                input_directory=str(inputs), input_sha256={name: checksum(inputs/(name+'_input.npz'))
                    for name in ('A', 'B', 'C', 'D')}, target_path=str(target_path),
                targets_sha256=checksum(target_path), audit_path=str(audit_path), audit_sha256=checksum(audit_path))
            with patch('bpdr_input.laboratory.verify_manifest', return_value=True):
                with self.assertRaisesRegex(ValueError, 'approvals'):
                    require_training_example(record)
                with review_path.open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    row.update(decision='approve', approved_intervals_json='[[0.2, 1.3]]',
                        reason='audited', reviewed_at_utc='2026-09-30T00:00:00Z')
                with review_path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                record['review_file_sha256'] = checksum(review_path)
                self.assertIs(require_training_example(record), record)
                np.savez_compressed(inputs/'A_input.npz', times_seconds=times,
                    frame_features_70=changed, boundary_features_71=np.column_stack([changed, nominal]))
                with self.assertRaisesRegex(ValueError, 'Model inputs changed'):
                    require_training_example(record)
                record['input_sha256']['A'] = checksum(inputs/'A_input.npz')
                record['support_seconds'] = [.1, 1.2]
                with self.assertRaisesRegex(ValueError, 'outside'):
                    require_training_example(record)


if __name__ == '__main__':
    unittest.main()
