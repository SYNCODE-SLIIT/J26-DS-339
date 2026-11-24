import copy
import unittest
from bpdr_input.boundary_comparison import advancement


class AdvancementTests(unittest.TestCase):
    def setUp(self):
        self.summary=dict(physical_passes=32,identity_passes=8,candidate_fine_passes=32,lost_baseline_passes=[],
            transitions={w:dict(baseline_above30=5,candidate_above30=4,baseline_max_peak=42.,candidate_max_peak=40.) for w in ('2048','1024')})

    def test_aggregate_gain_cannot_hide_a_lost_baseline_pass(self):
        summary=copy.deepcopy(self.summary);summary['lost_baseline_passes']=['passing_fixture']
        result=advancement(summary)
        self.assertFalse(result['eligible_for_bounded_vocal_comparison'])
        self.assertFalse(result['checks']['no_baseline_pass_lost'])

    def test_shorter_window_regression_cannot_be_ignored(self):
        summary=copy.deepcopy(self.summary);summary['transitions']['1024']['candidate_max_peak']=50.
        self.assertFalse(advancement(summary)['eligible_for_bounded_vocal_comparison'])

    def test_meeting_engineering_filter_never_adopts_or_accepts_audio(self):
        result=advancement(self.summary)
        self.assertTrue(result['eligible_for_bounded_vocal_comparison']);self.assertFalse(result['adopted'])
        self.assertIn('not audio acceptance',result['meaning'])
