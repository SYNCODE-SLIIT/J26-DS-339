from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf

from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.review_protection import protect


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.original, self.preview = root / 'original.wav', root / 'preview.wav'
        rate = 22050
        samples = .2 * np.sin(np.arange(rate * 5) * 2 * np.pi * 220 / rate)
        sf.write(self.original, samples, rate, subtype='PCM_24')
        prepared, _ = sf.read(self.original)
        prepared[rate:rate + 2205] *= .5
        sf.write(self.preview, prepared, rate, subtype='PCM_24')
        self.report = dict(outcome='VERIFIED_FIXTURE_CANDIDATE',
            transaction=dict(post_render_audit=dict(passed=True)),
            handoff=dict(request_id='synthetic_fixture', source_id='synthetic_fixture', status='ACCEPT_UNCHANGED',
                accepted_audio=dict(path=str(self.original), checksum=checksum(self.original), sample_rate_hz=rate,
                                    sample_count=len(samples), channels=1)),
            candidate=dict(path=str(self.preview), sha256=checksum(self.preview)))
        self.review = {role: dict(decision='PASS_EVIDENCE', waveform_sha256=checksum(path))
                       for role, path in [('original', self.original), ('preview', self.preview)]}

    def test_actual_changed_samples_veto_whole_preview_and_preserve_original(self):
        digest = checksum(self.original)
        report, review = protect(self.report, self.review, [[1., 1.1]])
        self.assertTrue(report['user_protection']['preview_vetoed'])
        self.assertGreater(report['user_protection']['changed_protected_samples'], 0)
        self.assertEqual(report['outcome'], 'USER_PROTECTED_UNCHANGED')
        self.assertNotIn('candidate', report)
        self.assertNotIn('preview', review)
        self.assertEqual(checksum(self.original), digest)
        self.assertIn('candidate', self.report)  # Caller retains an undo snapshot.
        self.assertIn('preview', self.review)

    def test_half_open_end_and_nonoverlap_keep_existing_preview(self):
        for ranges in [[[0., 1.]], [[1.1, 2.]], [[0., .5], [2., 3.]]]:
            report, review = protect(self.report, self.review, ranges)
            self.assertFalse(report['user_protection']['preview_vetoed'])
            self.assertEqual(report['candidate'], self.report['candidate'])
            self.assertIn('preview', review)

    def test_invalid_and_sample_free_ranges_are_rejected(self):
        for ranges in [[], [[-1., 1.]], [[2., 1.]], [[0., 6.]], [[float('nan'), 1.]], [[.000001, .000002]]]:
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                protect(self.report, self.review, ranges)

    def test_changed_preview_cannot_reuse_review(self):
        self.preview.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Preview changed'):
            protect(self.report, self.review, [[0., 1.]])


if __name__ == '__main__':
    unittest.main()
