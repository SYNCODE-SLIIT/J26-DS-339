"""Partition safety and paired renderer reporting, independent of model execution."""
import unittest

from bpdr_input.interventions import AuditPolicy
from bpdr_input.renderer_calibration import fine_pass, select_vocadito, summarize


class CalibrationTests(unittest.TestCase):
    def test_geometry_cannot_select_held_out_audio(self):
        rows = [dict(phrase_id=split, split=split, audio_path='must-not-be-opened')
                for split in ('train', 'development', 'pp1_evaluation', 'final_holdout')]
        coverage = dict(geometry=[dict(phrase_id=r['phrase_id'], policies={'baseline_086': {'eligible': True}})
                                  for r in rows])
        self.assertEqual(select_vocadito(dict(phrases=rows), coverage), [rows[0]])

    def test_more_passes_still_report_lost_baseline_cases_and_one_singer(self):
        rows = []
        for condition, passing in [('active_fft2048_hop512', {-30., -20.}),
                                   ('candidate_fft1024_hop128', {-30., -10., 10.}),
                                   ('candidate_fft1024_hop256', set())]:
            for cents in AuditPolicy().shifts_cents:
                rows.append(dict(record_id='crop_1', source_id='song_1', group_id='singer_1', dataset='vocadito',
                                 condition=condition, cents=cents, audit={'passed': cents in passing},
                                 reasons=[] if cents in passing else ['FINE_P90_ERROR']))
        result = summarize(rows)['vocadito/candidate_fft1024_hop128']
        self.assertEqual(result['passing_variants'], 3)
        self.assertEqual(result['retained_active_passes'], 1)
        self.assertEqual(result['lost_active_passes'], 1)
        self.assertEqual(result['added_passes'], 2)
        self.assertEqual(result['tested_groups'], 1)
        self.assertEqual(result['all_six_crops'], 0)

    def test_missing_or_insufficient_control_measurements_cannot_pass(self):
        self.assertFalse(fine_pass(dict(frames=0, median_error_cents=None, p90_error_cents=None), AuditPolicy()))
        self.assertFalse(fine_pass(dict(frames=9, median_error_cents=0., p90_error_cents=0.), AuditPolicy()))
        self.assertFalse(fine_pass(dict(frames=20, median_error_cents=1., p90_error_cents=11.), AuditPolicy()))


if __name__ == '__main__':
    unittest.main()
