import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair import local_review as local
from bpdr_input.repair.repair_controller import decision_hash, plan_fixture


class LocalReviewTests(unittest.TestCase):
    def setUp(self):
        parent = local.WORKSPACE / 'results/demo_sessions'
        self.temp = TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rate = 22050
        original = self.root / 'original.wav'
        preview = self.root / 'preview.wav'
        sf.write(original, .15 * np.sin(np.arange(self.rate * 10) * 2 * np.pi * 220 / self.rate), self.rate, subtype='PCM_24')
        self.source, _ = sf.read(original, dtype='float32')
        self.times = np.arange(len(self.source) // 441 + 1) * .02
        self.curve = np.zeros(len(self.times))
        self.curve[(self.times >= 2.) & (self.times < 3.)] = 20.
        self.curve[(self.times >= 6.) & (self.times < 7.)] = -20.
        reliable = np.ones(len(self.times), dtype=bool)
        protected = np.zeros(len(self.times), dtype=bool)
        plan = plan_fixture(self.source, self.rate, self.times, self.curve, self.curve,
                            np.zeros_like(self.curve), reliable, protected)
        self.assertTrue(plan['permitted'])
        candidate = self.source.copy()
        sample_times = np.arange(len(candidate)) / self.rate
        candidate[local._mask(sample_times, plan['intervals_seconds'])] *= .8
        sf.write(preview, candidate, self.rate, subtype='PCM_24')
        self.candidate, _ = sf.read(preview, dtype='float32')
        np.savez_compressed(self.root / 'model_predictions.npz', times=self.times,
                            proposed_correction=self.curve, reliable=reliable, protected=protected)
        self.report = dict(outcome='VERIFIED_FIXTURE_CANDIDATE', permission=plan,
            candidate=dict(path=str(preview), sha256=checksum(preview)),
            transaction=dict(post_render_audit=dict(passed=True)),
            handoff=dict(request_id='synthetic_combination', source_id='synthetic_combination', status='ACCEPT_UNCHANGED',
                accepted_audio=dict(path=str(original), checksum=checksum(original), sample_rate_hz=self.rate,
                                    sample_count=len(self.source), channels=1)))
        self.review = dict(intent='singing_humming_chant',
            original=dict(decision='PASS_EVIDENCE', waveform_sha256=checksum(original)),
            preview=dict(decision='PASS_EVIDENCE', waveform_sha256=checksum(preview)))

    def audit_fixture(self, path, original, rate, times, curve, reliable, intervals, intent):
        # Tests transport/state only; real acoustic audits are exercised separately.
        samples, sr = sf.read(path, dtype='float32')
        self.assertEqual(sr, rate)
        inside = local._mask(np.arange(len(original)) / rate, intervals)
        np.testing.assert_array_equal(samples[~inside], original[~inside])
        self.last_curve, self.last_intervals = curve.copy(), intervals
        return dict(passed=True, sha256=checksum(path), acoustic_audit=dict(passed=True,
            scope='mocked synthetic unit fixture')), dict(decision='PASS_EVIDENCE', feedback='Synthetic fixture')

    def test_subset_and_combination_use_only_selected_samples_and_fresh_audit(self):
        for selected in [[0], [1], [0, 1]]:
            with self.subTest(selected=selected), patch.object(local, '_audit_and_assess', side_effect=self.audit_fixture) as audit:
                report, review = local.select_intervals(self.root, self.report, self.review, selected)
                audit.assert_called_once()
            intervals = [self.report['permission']['intervals_seconds'][i] for i in selected]
            inside = local._mask(np.arange(len(self.source)) / self.rate, intervals)
            samples, _ = sf.read(report['candidate']['path'], dtype='float32')
            np.testing.assert_array_equal(samples[inside], self.candidate[inside])
            np.testing.assert_array_equal(samples[~inside], self.source[~inside])
            self.assertTrue(report['local_selection']['final_decoded_audit']['passed'])
            self.assertEqual(report['permission']['decision_sha256'], decision_hash(report['permission']))
            self.assertEqual(review['preview']['waveform_sha256'], checksum(report['candidate']['path']))
            np.testing.assert_array_equal(self.last_curve[~local._mask(self.times, intervals)], 0.)

    def test_reject_all_has_no_candidate_or_acoustic_audit(self):
        with patch.object(local, '_audit_and_assess') as audit:
            report, review = local.select_intervals(self.root, self.report, self.review, [])
        audit.assert_not_called()
        self.assertNotIn('candidate', report)
        self.assertNotIn('preview', review)
        self.assertFalse(report['permission']['permitted'])
        self.assertEqual(report['local_selection']['rejected_edit_indices'], [0, 1])

    def test_failed_final_audit_leaves_previous_version_unchanged(self):
        before = copy.deepcopy(self.report)
        digest = checksum(self.report['candidate']['path'])
        with patch.object(local, '_audit_and_assess', side_effect=RuntimeError('fixture audit failure')):
            with self.assertRaisesRegex(ValueError, 'not accepted'):
                local.select_intervals(self.root, self.report, self.review, [0])
        self.assertEqual(self.report, before)
        self.assertEqual(checksum(self.report['candidate']['path']), digest)
        receipts = list(self.root.glob('review_combinations/*/review_selection.json'))
        self.assertEqual(len(receipts), 1)
        self.assertFalse(json.loads(receipts[0].read_text())['success'])

    def test_invalid_ids_and_changed_prediction_binding_are_refused(self):
        for selected in [[-1], [2], [True], [0, 0]]:
            with self.assertRaises(ValueError):
                local.select_intervals(self.root, self.report, self.review, selected)
        np.savez_compressed(self.root / 'model_predictions.npz', times=self.times,
            proposed_correction=self.curve + 1, reliable=np.ones(len(self.times), dtype=bool))
        with self.assertRaisesRegex(ValueError, 'prediction grid'):
            local.select_intervals(self.root, self.report, self.review, [0])

    def test_protected_or_outside_permission_changes_cannot_be_selected(self):
        self.report['user_protection'] = dict(intervals_seconds=[[2., 2.5]])
        with self.assertRaisesRegex(ValueError, 'protected samples'):
            local.select_intervals(self.root, self.report, self.review, [0])
        self.report.pop('user_protection')
        preview = Path(self.report['candidate']['path'])
        changed = self.candidate.copy()
        changed[100] *= .5
        sf.write(preview, changed, self.rate, subtype='PCM_24')
        self.report['candidate']['sha256'] = checksum(preview)
        self.review['preview']['waveform_sha256'] = checksum(preview)
        with self.assertRaisesRegex(ValueError, 'outside its permission'):
            local.select_intervals(self.root, self.report, self.review, [0])


if __name__ == '__main__':
    unittest.main()
