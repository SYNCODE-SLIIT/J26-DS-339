import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

import numpy as np
import soundfile as sf

from bpdr_input.application.request_delivery import summarize
from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.repair_controller import plan_fixture, decision_hash, waveform_hash
from bpdr_input.repair.review_export import export_reviewed


class RequestDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original, self.preview = self.root / 'original.wav', self.root / 'preview.wav'
        self.rate = 22050
        sf.write(self.original, .15 * np.sin(np.arange(self.rate * 10) * 2 * np.pi * 220 / self.rate), self.rate, subtype='PCM_24')
        self.source, _ = sf.read(self.original, dtype='float32')
        self.times = np.arange(len(self.source) // 441 + 1) * .02
        self.curve = np.zeros(len(self.times))
        self.curve[(self.times >= 1.) & (self.times < 2.)] = 20.
        self.curve[(self.times >= 6.) & (self.times < 7.)] = -15.
        plan = plan_fixture(self.source, self.rate, self.times, self.curve, self.curve,
            np.zeros_like(self.curve), np.ones(len(self.times), dtype=bool), np.zeros(len(self.times), dtype=bool))
        self.assertTrue(plan['permitted'])
        self.candidate = self.source.copy()
        sample_times = np.arange(len(self.source)) / self.rate
        self.candidate[((sample_times >= 1.) & (sample_times < 2.)) | ((sample_times >= 6.) & (sample_times < 7.))] *= .6
        sf.write(self.preview, self.candidate, self.rate, subtype='PCM_24')
        raw = np.where(self.curve != 0, 30., 0.)
        np.savez_compressed(self.root / 'model_predictions.npz', times=self.times,
            proposed_correction=self.curve, predictions=np.stack([raw, raw, raw]))
        self.report = dict(outcome='VERIFIED_FIXTURE_CANDIDATE', permission=plan,
            candidate=dict(path=str(self.preview), sha256=checksum(self.preview)),
            transaction=dict(post_render_audit=dict(passed=True)),
            handoff=dict(request_id='synthetic_request', source_id='synthetic_request', status='ACCEPT_UNCHANGED',
                accepted_audio=dict(path=str(self.original), checksum=checksum(self.original),
                    sample_rate_hz=self.rate, sample_count=len(self.source), channels=1)))
        self.review = {role: dict(decision='PASS_EVIDENCE', waveform_sha256=checksum(path))
                       for role, path in [('original', self.original), ('preview', self.preview)]}

    def test_original_selection_separates_request_permission_and_actual_zero_changes(self):
        before = copy.deepcopy(self.report)
        result = summarize(self.root, self.report, 'original')
        self.assertEqual(result['model_request_max_abs_cents'], 30.)
        self.assertEqual(result['operating_request_max_abs_cents'], 20.)
        self.assertEqual(result['permitted_preview_request_max_abs_cents'], 20.)
        self.assertEqual(result['selected_request_max_abs_cents'], 0.)
        self.assertEqual(result['actual_changed_samples'], 0)
        self.assertEqual(result['selected_sha256'], checksum(self.original))
        self.assertEqual(self.report, before)

    def test_preview_counts_are_measured_from_decoded_wav(self):
        result = summarize(self.root, self.report, 'preview')
        candidate, _ = sf.read(self.preview, dtype='float64')
        original, _ = sf.read(self.original, dtype='float64')
        self.assertEqual(result['actual_changed_samples'], int(np.count_nonzero(candidate != original)))
        self.assertEqual(result['selected_request_max_abs_cents'], 20.)
        self.assertEqual(result['selected_sha256'], checksum(self.preview))

    def test_withheld_preview_does_not_gain_permission_from_existing_file(self):
        self.report['outcome'] = 'ROLLED_BACK'
        result = summarize(self.root, self.report, 'original')
        self.assertEqual(result['model_request_max_abs_cents'], 30.)
        self.assertEqual(result['permitted_preview_request_max_abs_cents'], 0.)
        self.assertEqual(result['actual_changed_samples'], 0)
        with self.assertRaisesRegex(ValueError, 'No permitted preview'):
            summarize(self.root, self.report, 'preview')

    def test_subset_permission_uses_selected_curve_not_full_proposal_maximum(self):
        candidate = self.source.copy()
        selected = (np.arange(len(candidate)) / self.rate >= 6.) & (np.arange(len(candidate)) / self.rate < 7.)
        candidate[selected] *= .6
        sf.write(self.preview, candidate, self.rate, subtype='PCM_24')
        self.report['candidate']['sha256'] = checksum(self.preview)
        plan = self.report['permission']
        plan['intervals_seconds'] = [(6., 7.)]
        curve = self.curve.copy()
        curve[(self.times < 6.) | (self.times >= 7.)] = 0
        plan['proposal_sha256'] = waveform_hash(curve)
        plan['decision_sha256'] = decision_hash(plan)
        result = summarize(self.root, self.report, 'preview')
        self.assertEqual(result['operating_request_max_abs_cents'], 20.)
        self.assertEqual(result['permitted_preview_request_max_abs_cents'], 15.)
        self.assertEqual(result['selected_request_max_abs_cents'], 15.)

    def test_cleanup_is_not_presented_as_a_pitch_request(self):
        self.report.update(outcome='NOISE_PREVIEW_PASSED', noise_candidate=self.report.pop('candidate'),
                           noise=dict(audit=dict(passes=True)))
        result = summarize(self.root, self.report, 'preview')
        self.assertEqual(result['preview_operation'], 'whole_clip_cleanup')
        self.assertIsNone(result['model_request_max_abs_cents'])
        self.assertIsNone(result['permitted_preview_request_max_abs_cents'])
        self.assertGreater(result['actual_changed_samples'], 0)

    def test_unregistered_preview_does_not_invent_a_zero_pitch_request(self):
        self.report.pop('permission')
        result = summarize(self.root, self.report, 'preview')
        self.assertEqual(result['preview_operation'], 'verified_preview')
        self.assertIsNone(result['permitted_preview_request_max_abs_cents'])
        self.assertIsNone(result['selected_request_max_abs_cents'])

    def test_changed_hash_grid_and_rejected_sources_are_refused(self):
        self.report['candidate']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'checksum'):
            summarize(self.root, self.report, 'preview')
        self.report['candidate']['sha256'] = checksum(self.preview)
        sf.write(self.preview, self.candidate, 16000, subtype='PCM_24')
        self.report['candidate']['sha256'] = checksum(self.preview)
        with self.assertRaisesRegex(ValueError, 'working source grid'):
            summarize(self.root, self.report, 'preview')
        with self.assertRaisesRegex(ValueError, 'No accepted source'):
            summarize(self.root, dict(handoff=dict(accepted_audio=None)), 'original')

    def test_export_rejects_stale_selection_summary_and_retains_correct_metadata(self):
        self.report['delivery_summary'] = summarize(self.root, self.report, 'original')
        with self.assertRaisesRegex(ValueError, 'different selection'):
            export_reviewed(self.root, self.report, self.review, use_preview=True,
                            confirmed=True, confirmation_kind='automated_test_fixture')
        self.report['delivery_summary'] = summarize(self.root, self.report, 'preview')
        package, handoff = export_reviewed(self.root, self.report, self.review, use_preview=True,
            confirmed=True, confirmation_kind='automated_test_fixture')
        with zipfile.ZipFile(package / 'accepted_package.zip') as archive:
            diagnostics = json.loads(archive.read('diagnostics.json'))
        self.assertEqual(diagnostics['delivery_summary']['selected_role'], 'preview')
        self.assertEqual(diagnostics['delivery_summary']['selected_sha256'], handoff['accepted_audio']['checksum'])
        self.assertFalse(diagnostics['user_confirmed'])


if __name__ == '__main__':
    unittest.main()
