from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from bpdr_input.repair import local_review as local
from bpdr_input.repair.demo_runtime import METHODS
from test_local_review import LocalReviewTests


class LocalReviewUITests(unittest.TestCase):
    def setUp(self):
        self.fixture = LocalReviewTests('test_reject_all_has_no_candidate_or_acoustic_audit')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        original = self.fixture.report['handoff']['accepted_audio']['path']
        self.fixture.report['source'] = original
        for row in (self.fixture.review['original'], self.fixture.review['preview']):
            row['feedback'] = 'Synthetic UI fixture; no musical judgment.'
        self.enterContext(patch('bpdr_input.repair.demo_runtime.available_methods', return_value=METHODS))
        self.enterContext(patch('bpdr_input.application.demo_examples.examples', return_value=[dict(
            id='synthetic_ui', label='Synthetic local selection', input=Path(original), reference=None,
            source_kind='Synthetic state fixture only', method=METHODS[3])]))
        self.audit = self.enterContext(patch.object(local, '_audit_and_assess', side_effect=self.fixture.audit_fixture))
        self.app = AppTest.from_file(str(local.WORKSPACE / 'source/bpdr_input/application/review_demo_app.py'), default_timeout=120)
        self.app.session_state['demo_result'] = dict(directory=str(self.root), report=self.fixture.report,
            review=self.fixture.review, method='Synthetic fixture', reference=None, known_case=None)
        self.app.run()
        self.assertFalse(self.app.exception)

    def choose(self, indices):
        next(m for m in self.app.multiselect if m.label == 'Edits to keep').set_value(indices)
        next(b for b in self.app.button if b.label == 'Apply selected edits').click().run()
        self.assertFalse(self.app.exception)

    def test_subset_combination_and_undo_keep_original_base_available(self):
        self.choose([0])
        self.assertEqual(self.app.session_state['demo_result']['report']['local_selection']['selected_edit_indices'], [0])
        self.choose([0, 1])
        self.assertEqual(self.app.session_state['demo_result']['report']['local_selection']['selected_edit_indices'], [0, 1])
        next(b for b in self.app.button if b.label == 'Undo last review change').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state['demo_result']['report']['local_selection']['selected_edit_indices'], [0])
        self.assertFalse(next(c for c in self.app.checkbox if c.label.startswith('I listened')).value)

    def test_reject_all_and_failed_audit_retain_last_usable_version(self):
        self.choose([0])
        previous = self.app.session_state['demo_result']['report']['candidate']['sha256']
        self.audit.side_effect = RuntimeError('synthetic final-audit failure')
        self.choose([0, 1])
        self.assertTrue(any('previous version is retained' in e.value for e in self.app.error))
        self.assertEqual(self.app.session_state['demo_result']['report']['candidate']['sha256'], previous)
        self.choose([])
        report = self.app.session_state['demo_result']['report']
        self.assertEqual(report['outcome'], 'USER_REJECTED_EDITS_UNCHANGED')
        self.assertNotIn('candidate', report)
        self.assertEqual(report['local_selection']['selected_edit_indices'], [])
        self.assertEqual(len(next(m for m in self.app.multiselect if m.label == 'Edits to keep').options), 2)


if __name__ == '__main__':
    unittest.main()
