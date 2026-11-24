import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import soundfile as sf

from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.demo_review import export_reviewed
from bpdr_input.repair.demo_extraction import run as extract
from bpdr_input.repair.demo_runtime import selected_output


class DemoReviewTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        path = root / 'original.wav'
        sf.write(path, .1 * np.sin(np.arange(22050) * 2 * np.pi * 220 / 22050), 22050, subtype='PCM_24')
        accepted = dict(path=str(path), checksum=checksum(path), sample_rate_hz=22050,
                        sample_count=22050, channels=1)
        report = dict(source=str(path), outcome='IDENTITY',
                      handoff=dict(source_id='sample', status='ACCEPT_UNCHANGED', accepted_audio=accepted),
                      prepared_handoff_enabled=False)
        assessment = dict(original=dict(decision='PASS_EVIDENCE', waveform_sha256=checksum(path)))
        return root, report, assessment

    def test_confirmation_is_required_before_any_export(self):
        with self.assertRaisesRegex(ValueError, 'Confirm'):
            export_reviewed('unused', {}, {}, confirmed=False)

    def test_rejected_input_cannot_export_even_with_a_claimed_pass(self):
        with self.assertRaisesRegex(ValueError, 'technically accepted'):
            export_reviewed('unused', {'handoff': {'accepted_audio': None}}, {}, confirmed=True)

    def test_valid_original_exports_portable_audio_and_keeps_shared_policy(self):
        with TemporaryDirectory() as folder:
            root, report, assessment = self.fixture(folder)
            destination, package = export_reviewed(root, report, assessment, confirmed=True)
            self.assertEqual(package['status'], 'ACCEPT_UNCHANGED')
            self.assertFalse(report['prepared_handoff_enabled'])
            with zipfile.ZipFile(destination / 'accepted_package.zip') as archive:
                portable = json.loads(archive.read('handoff.json'))
                self.assertEqual(portable['accepted_audio']['path'], 'accepted.wav')
                self.assertIn('accepted.wav', archive.namelist())
            diagnostics = json.loads((destination / 'diagnostics.json').read_text())
            self.assertFalse(diagnostics['research_approval'])
            self.assertFalse(diagnostics['shared_handoff_changed'])

    def test_failed_candidate_is_never_exported(self):
        with TemporaryDirectory() as folder:
            root, report, assessment = self.fixture(folder)
            item = report['handoff']['accepted_audio']
            report['candidate'] = dict(path=item['path'], sha256=item['checksum'])
            report['outcome'] = 'ROLLED_BACK'
            assessment['preview'] = assessment['original'].copy()
            with self.assertRaisesRegex(ValueError, 'does not authorize'):
                export_reviewed(root, report, assessment, use_preview=True, confirmed=True)

    def test_review_must_match_selected_audio_and_never_accept_unknown_intent(self):
        with TemporaryDirectory() as folder:
            root, report, assessment = self.fixture(folder)
            assessment['original']['waveform_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'does not belong'):
                export_reviewed(root, report, assessment, confirmed=True)
            assessment['original']['decision'] = 'NEED_INTENT_CONFIRMATION'
            with self.assertRaisesRegex(ValueError, 'sufficient'):
                export_reviewed(root, report, assessment, confirmed=True)

    def test_changed_waveform_cannot_reuse_previous_review(self):
        with TemporaryDirectory() as folder:
            root, report, assessment = self.fixture(folder)
            Path(report['source']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                export_reviewed(root, report, assessment, confirmed=True)

    def test_extractor_refuses_paths_outside_workspace_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'inside BPDR_Workspace'):
            extract(Path('C:/Windows'))

    def test_copy_must_retain_reviewed_bytes(self):
        with TemporaryDirectory() as folder:
            root, report, assessment = self.fixture(folder)
            def replaced_source(source, destination):
                Path(destination).write_bytes(b'replaced during copying')
            with patch('bpdr_input.repair.demo_review.shutil.copyfile', side_effect=replaced_source):
                with self.assertRaisesRegex(ValueError, 'Copied waveform'):
                    export_reviewed(root, report, assessment, confirmed=True)
            self.assertFalse(list(root.glob('reviewed_handoffs/*/handoff.json')))

    def test_failed_noise_file_is_not_presented_as_success(self):
        with TemporaryDirectory() as folder:
            root, report, assessment = self.fixture(folder)
            item = report['handoff']['accepted_audio']
            report.update(outcome='NOISE_PREVIEW_FAILED_UNCHANGED', noise_candidate=dict(path=item['path'], sha256=item['checksum']))
            selected, label = selected_output(report)
            self.assertEqual(label, 'Original retained')
            self.assertEqual(selected, Path(item['path']))


if __name__ == '__main__':
    unittest.main()
