"""The advertised learned preview must pass checks after delivery encoding."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from bpdr_input.audio.contracts import AcceptedAudio, ProcessingResult, Status
from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.demo_runtime import eligible_preview
from bpdr_input.repair.learned_runtime import audit_delivered_wav, run, run_candidate


class LearnedDeliveryTests(unittest.TestCase):
    def source(self, directory):
        sr = 22050
        samples = (.1 * np.sin(2 * np.pi * 220 * np.arange(2 * sr) / sr)).astype(np.float32)
        path = Path(directory) / 'accepted.wav'
        sf.write(path, samples, sr, subtype='PCM_24')
        original, _ = sf.read(path, dtype='float32')
        return path, original, sr

    def test_auditor_receives_decoded_delivery_and_digest_binds_to_file(self):
        with TemporaryDirectory() as directory:
            _, original, sr = self.source(directory)
            candidate = original.copy()
            candidate[sr // 2:sr] += np.float32(.00000003)
            path = Path(directory) / 'candidate.wav'
            sf.write(path, candidate, sr, subtype='PCM_24')
            decoded, _ = sf.read(path, dtype='float32')
            self.assertFalse(np.array_equal(candidate, decoded))
            auditor = Mock(return_value={'passed': True, 'measurement': 'decoded fixture'})
            audit = audit_delivered_wav(path, original, sr, ((.5, 1.),), auditor)
            np.testing.assert_array_equal(auditor.call_args.args[1], decoded)
            self.assertEqual(audit['sha256'], checksum(path))
            self.assertTrue(audit['outside_support_unchanged'])
            self.assertEqual(audit['subtype'], 'PCM_24')

    def test_outside_sample_change_stops_before_acoustic_approval(self):
        with TemporaryDirectory() as directory:
            _, original, sr = self.source(directory)
            candidate = original.copy()
            candidate[0] += .001
            path = Path(directory) / 'candidate.wav'
            sf.write(path, candidate, sr, subtype='PCM_24')
            auditor = Mock(return_value={'passed': True})
            with self.assertRaisesRegex(ValueError, 'OUTSIDE_SUPPORT_CHANGED'):
                audit_delivered_wav(path, original, sr, ((.5, 1.),), auditor)
            auditor.assert_not_called()

    def test_changed_rate_count_and_channels_are_rejected(self):
        with TemporaryDirectory() as directory:
            _, original, sr = self.source(directory)
            path = Path(directory) / 'candidate.wav'
            for waveform, rate in ((original, 16000), (original[:-1], sr),
                                   (np.column_stack((original, original)), sr)):
                sf.write(path, waveform, rate, subtype='PCM_24')
                with self.subTest(rate=rate, shape=waveform.shape):
                    with self.assertRaisesRegex(ValueError, 'SAMPLE_GRID'):
                        audit_delivered_wav(path, original, sr, ((.5, 1.),), Mock())

    def test_delivery_acoustic_failure_is_not_a_pass(self):
        with TemporaryDirectory() as directory:
            path, original, sr = self.source(directory)
            with self.assertRaisesRegex(ValueError, 'ACOUSTIC_AUDIT_FAILED'):
                audit_delivered_wav(path, original, sr, (), lambda *_: {'passed': False})

    def test_file_replacement_during_audit_is_rejected(self):
        with TemporaryDirectory() as directory:
            path, original, sr = self.source(directory)
            def replace(*_):
                sf.write(path, original * .5, sr, subtype='PCM_24')
                return {'passed': True}
            with self.assertRaisesRegex(ValueError, 'CHANGED_DURING_AUDIT'):
                audit_delivered_wav(path, original, sr, (), replace)

    def test_public_run_always_uses_guarded_loader(self):
        with patch('bpdr_input.repair.learned_runtime.load_model', side_effect=ValueError('ineligible')):
            with patch('bpdr_input.repair.learned_runtime.run_candidate') as internal:
                with self.assertRaisesRegex(ValueError, 'ineligible'):
                    run('unused.wav')
                internal.assert_not_called()

    def test_failed_delivery_leaves_no_eligible_candidate_even_if_wav_remains(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path, original, sr = self.source(root)
            times = np.arange(len(original) // 441 + 1) * .02
            np.savez_compressed(root / 'features.npz', times_seconds=times)
            accepted = AcceptedAudio(str(path), checksum(path), sr, len(original))
            package = ProcessingResult('fixture', 'source', Status.ACCEPT_UNCHANGED, accepted)
            raw = np.zeros((len(times), 71), dtype=np.float32)
            reliable = np.ones(len(times), dtype=bool)
            proposal = dict(requested_curve=np.zeros(len(times)), operating_predictions=np.zeros((3, len(times))),
                            reliable=reliable, disagreement=np.zeros(len(times)))
            transaction = dict(outcome='VERIFIED_FIXTURE_CANDIDATE', post_render_audit={'passed': True})
            card = dict(checkpoint_sha256='0' * 64, checkpoint='fixture-model')
            with patch('bpdr_input.repair.learned_runtime.process', return_value=(package, root)), \
                 patch('bpdr_input.repair.model_runtime.runtime_views', return_value=([raw] * 3, reliable, ~reliable, {})), \
                 patch('bpdr_input.repair.rule_correction.fine_pitch', return_value=np.zeros(len(times))), \
                 patch('bpdr_input.modeling.recovery_model.propose', return_value=proposal), \
                 patch('bpdr_input.repair.repair_controller.plan_fixture', return_value={'intervals_seconds': []}), \
                 patch('bpdr_input.repair.repair_controller.transact_fixture', return_value=(original, transaction)), \
                 patch('bpdr_input.repair.learned_runtime.audit_delivered_wav', side_effect=ValueError('encoded audit rejected')):
                _, report = run_candidate(path, object(), {}, card, output_root=root)
            self.assertTrue((root / 'experimental_learned_candidate.wav').exists())
            self.assertEqual(report['outcome'], 'LEARNED_PREVIEW_FAILED_UNCHANGED')
            self.assertNotIn('candidate', report)
            self.assertIsNone(eligible_preview(report))
            self.assertEqual(report['transaction']['outcome'], 'ROLLED_BACK')
            self.assertFalse(report['transaction']['delivered_wav_audit']['passed'])
            self.assertEqual(report['handoff']['status'], 'ACCEPT_UNCHANGED')


if __name__ == '__main__':
    unittest.main()
