from types import SimpleNamespace
import unittest
from bpdr_input.training_reference_completion_v2 import expected_accepted_count


class ExactAcceptedCountTests(unittest.TestCase):
    def phrase(self,count):
        return dict(split='train',start_sample=877568,end_sample=877568+count,
            original_sample_rate_hz=44100,duration_seconds=count/44100)

    def test_real_half_tie_does_not_round_duration_twice(self):
        p=self.phrase(226409)
        self.assertEqual(round(p['duration_seconds']*22050),113205)
        self.assertEqual(expected_accepted_count(p,SimpleNamespace(frames=226409,samplerate=44100)),113204)

    def test_both_half_ties_and_even_counts_follow_original_pipeline(self):
        for n in [226409,226411,226410,44100]:
            self.assertEqual(expected_accepted_count(self.phrase(n),SimpleNamespace(frames=n,samplerate=44100)),round(n*22050/44100))

    def test_wrong_source_count_rate_or_duration_refused(self):
        p=self.phrase(226409)
        for info in [SimpleNamespace(frames=226410,samplerate=44100),SimpleNamespace(frames=226409,samplerate=48000)]:
            with self.assertRaises(ValueError):expected_accepted_count(p,info)
        with self.assertRaises(ValueError):expected_accepted_count(dict(p,duration_seconds=5.),SimpleNamespace(frames=226409,samplerate=44100))

    def test_reserved_partition_refused_before_read(self):
        with self.assertRaises(ValueError):expected_accepted_count(dict(self.phrase(44100),split='final_holdout'),None)
