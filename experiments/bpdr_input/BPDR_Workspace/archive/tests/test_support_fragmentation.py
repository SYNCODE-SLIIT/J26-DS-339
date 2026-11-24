import unittest
import numpy as np
from bpdr_input.support_fragmentation import diagnose, intervals
from bpdr_input.source_estimator import nearest_indices, guarded_support


class SupportFragmentationTests(unittest.TestCase):
    def fixture(self):
        t = np.arange(6)*.1
        f = np.array([np.nan, 220., 220., 220., 220., 220.])
        y = np.array([220., np.nan, 440., 220., 220., 220.])
        c = np.array([.2, .9, .9, .9, .9, .9])
        reliable = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
        st = np.arange(13230)/22050
        mapped = reliable[nearest_indices(t, st)]; guard = guarded_support(mapped, 22050)
        return dict(frame_times=t, frame_pesto_hz=f, frame_yin_hz=y, frame_confidence=c,
                    frame_reliable=reliable, sample_times=st, sample_reliable=mapped,
                    guarded_support=guard, retained_support=guard.copy(), estimated_amplitude=np.ones(len(st)))

    def test_exclusive_partition_does_not_double_count_invalid_frames(self):
        r = diagnose(self.fixture())
        self.assertEqual(list(r['frame_partition'].values()), [1, 1, 1, 3])
        self.assertEqual(r['reliable_samples']-r['guard_removed_samples'], r['guarded_samples'])

    def test_zero_support_is_observed_and_no_runs_are_invented(self):
        a = self.fixture(); a['frame_confidence'][:] = 0; a['frame_reliable'][:] = False
        for k in ('sample_reliable', 'guarded_support', 'retained_support'): a[k][:] = False
        r = diagnose(a)
        self.assertEqual(r['raw_runs'], 0); self.assertEqual(r['reliable_yin_band_denominator'], 0)

    def test_mask_tampering_is_rejected(self):
        a = self.fixture(); a['retained_support'][0] = True
        with self.assertRaises(ValueError): diagnose(a)

    def test_amplitude_loss_and_holes_are_counted(self):
        a = self.fixture(); a['estimated_amplitude'][8000:8100] = 0
        a['retained_support'][8000:8100] = False
        self.assertEqual(diagnose(a)['amplitude_removed_samples'], 100)

    def test_half_open_intervals_preserve_single_sample_and_terminal_runs(self):
        r = intervals(np.array([1, 0, 1, 1], dtype=bool))
        self.assertEqual([(v['start_sample'], v['end_sample']) for v in r], [(0, 1), (2, 4)])
        self.assertEqual(intervals(np.zeros(0, dtype=bool)), [])

    def test_diagnosis_does_not_mutate_frozen_arrays(self):
        a = self.fixture(); saved = {k: v.copy() for k, v in a.items()}; diagnose(a)
        for k in a: np.testing.assert_array_equal(a[k], saved[k])
