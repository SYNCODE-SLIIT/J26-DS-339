import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.demo_review import export_reviewed
from bpdr_input.repair import primary_handoff as primary


class PrimaryHandoffTests(unittest.TestCase):
    def setUp(self):
        parent = primary.WORKSPACE / 'results/demo_sessions'
        parent.mkdir(exist_ok=True)
        self.temp = TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = self.root / 'original.wav'
        sf.write(path, .1 * np.sin(np.arange(22050) * 2 * np.pi * 220 / 22050),
                 22050, subtype='PCM_24')
        self.report = dict(outcome='IDENTITY', handoff=dict(source_id='synthetic_fixture',
            request_id='synthetic_fixture',
            status='ACCEPT_UNCHANGED', accepted_audio=dict(path=str(path), checksum=checksum(path),
            sample_rate_hz=22050, sample_count=22050, channels=1)))
        self.review = dict(original=dict(decision='PASS_EVIDENCE', waveform_sha256=checksum(path)))
        self.package, _ = export_reviewed(self.root, self.report, self.review,
            confirmed=True, confirmation_kind='automated_test_fixture')

    def test_fixture_never_impersonates_user_confirmation(self):
        with patch.object(primary.subprocess, 'run') as child:
            with self.assertRaisesRegex(ValueError, 'fixture mode'):
                primary.run(self.package)
            child.assert_not_called()

    def test_rejected_package_cannot_reach_consumer(self):
        path = self.package / 'handoff.json'
        data = json.loads(path.read_text())
        data.update(status='RERECORD_REQUIRED', accepted_audio=None)
        path.write_text(json.dumps(data))
        with patch.object(primary.subprocess, 'run') as child:
            with self.assertRaisesRegex(ValueError, 'Only an accepted'):
                primary.run(self.package, allow_test_fixture=True)
            child.assert_not_called()

    def test_changed_audio_and_wrong_review_are_rejected(self):
        diagnostics = self.package / 'diagnostics.json'
        data = json.loads(diagnostics.read_text())
        data['usability']['waveform_sha256'] = '0' * 64
        diagnostics.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'Review does not belong'):
            primary.run(self.package, allow_test_fixture=True)
        (self.package / 'accepted.wav').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            primary.run(self.package, allow_test_fixture=True)

    def test_audio_cannot_leave_reviewed_package(self):
        path = self.package / 'handoff.json'
        data = json.loads(path.read_text())
        data['accepted_audio']['path'] = self.report['handoff']['accepted_audio']['path']
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'leaves its reviewed package'):
            primary.run(self.package, allow_test_fixture=True)

    def test_timeout_preserves_accepted_audio_without_success(self):
        digest = checksum(self.package / 'accepted.wav')
        with patch.object(primary, '_freeze', return_value='f' * 64), patch.object(
                primary.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['fixture'], 1)):
            result = primary.run(self.package, timeout_seconds=1, allow_test_fixture=True)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'CONSUMER_TIMEOUT')
        self.assertFalse(result['human_review_claimed'])
        self.assertEqual(checksum(self.package / 'accepted.wav'), digest)
        self.assertNotIn('artifacts', result)

    def test_changed_copy_cannot_launch_consumer(self):
        def replace(source, destination):
            Path(destination).write_bytes(b'wrong copy')
        with patch.object(primary, '_freeze', return_value='f' * 64), patch.object(
                primary.shutil, 'copyfile', side_effect=replace), patch.object(primary.subprocess, 'run') as child:
            with self.assertRaisesRegex(ValueError, 'Consumer copy differs'):
                primary.run(self.package, allow_test_fixture=True)
            child.assert_not_called()

    def test_ui_selection_change_discards_old_export_and_consumer_result(self):
        from streamlit.testing.v1 import AppTest
        original = Path(self.report['handoff']['accepted_audio']['path'])
        prepared = self.root / 'prepared.wav'
        samples, rate = sf.read(original)
        sf.write(prepared, samples * .6, rate, subtype='PCM_24')
        self.report.update(source=str(original), candidate=dict(path=str(prepared), sha256=checksum(prepared)),
            outcome='VERIFIED_FIXTURE_CANDIDATE', transaction=dict(post_render_audit=dict(passed=True)))
        self.review['preview'] = dict(decision='PASS_EVIDENCE', waveform_sha256=checksum(prepared))
        for item in self.review.values():
            item['feedback'] = 'Synthetic UI fixture, no human assessment.'
        # The UI state contract does not depend on loading any legacy model.
        from bpdr_input.repair.demo_runtime import METHODS
        self.enterContext(patch('bpdr_input.repair.demo_runtime.available_methods', return_value=METHODS))
        self.enterContext(patch('bpdr_input.application.demo_examples.examples', return_value=[dict(
            id='synthetic_ui', label='Synthetic state fixture', input=original,
            reference=None, source_kind='Synthetic UI fixture only', method=METHODS[3])]))
        app = AppTest.from_file(str(primary.WORKSPACE / 'source/bpdr_input/application/review_demo_app.py'), default_timeout=120)
        app.session_state['demo_result'] = dict(directory=str(self.root), report=self.report,
            review=self.review, method='Synthetic UI fixture', reference=None, known_case=None)
        app.run()
        self.assertFalse(app.exception)
        choose = lambda: next(r for r in app.radio if r.label == 'Version to hand off')
        # Exercise checkbox state only; never click export or create user-review evidence.
        next(c for c in app.checkbox if c.label.startswith('I listened')).check().run()
        selection = [str(self.root.resolve()), 'original', checksum(original)]
        handoff = json.loads((self.package / 'handoff.json').read_text())
        app.session_state['demo_export'] = dict(directory=str(self.package), handoff=handoff, selection=selection)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(next(s for s in app.selectbox if s.label == 'Melody extractor').value, 'Shared melody component')
        app.session_state['demo_extraction'] = dict(success=False, consumer='frozen_shared_pyin')
        app.session_state['confirm_' + self.root.name + '_preview_' + checksum(prepared)] = True
        choose().set_value('preview').run()
        self.assertFalse(app.exception)
        with self.assertRaises(KeyError):
            app.session_state['demo_export']
        with self.assertRaises(KeyError):
            app.session_state['demo_extraction']
        next(b for b in app.button if b.label == 'Protect this range').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['demo_result']['report']['outcome'], 'USER_PROTECTED_UNCHANGED')
        self.assertEqual(next(r for r in app.radio if r.label == 'Version to hand off').value, 'original')
        next(b for b in app.button if b.label == 'Undo last review change').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['demo_result']['report']['outcome'], 'VERIFIED_FIXTURE_CANDIDATE')
        self.assertIn('preview', app.session_state['demo_result']['review'])
        self.assertFalse(next(c for c in app.checkbox if c.label.startswith('I listened')).value)


if __name__ == '__main__':
    unittest.main()
