"""Safety properties of training-only coverage probes."""
import unittest
import numpy as np
from bpdr_input.coverage import candidate_policies, support_record, training_phrases
from bpdr_input.interventions import AuditPolicy


class CoverageTests(unittest.TestCase):
    def test_held_out_paths_are_not_selected(self):
        rows = [dict(phrase_id=split, split=split, audio_path='unreadable')
                for split in ('train', 'development', 'pp1_evaluation', 'final_holdout')]
        self.assertEqual(training_phrases(dict(phrases=rows)), [rows[0]])

    def test_candidate_geometry_preserves_quality_thresholds(self):
        baseline = AuditPolicy()
        changed = {'version', 'analysis_guard_seconds', 'ramp_seconds', 'min_plateau_seconds'}
        for candidate in candidate_policies().values():
            for key, value in baseline.__dict__.items():
                if key not in changed:
                    self.assertEqual(getattr(candidate, key), value)
            self.assertGreaterEqual(candidate.analysis_guard_seconds, 2048/(2*22050))
        self.assertEqual(AuditPolicy(), baseline)

    def test_reference_only_eligibility_is_not_enough_without_outside_anchors(self):
        times = np.arange(0, .6, .02)
        mask = np.ones(len(times), dtype=bool)
        result = support_record(times, mask, candidate_policies()['candidate_050'])
        self.assertIsNotNone(result['support_seconds'])
        self.assertFalse(result['eligible'])


if __name__ == '__main__':
    unittest.main()
