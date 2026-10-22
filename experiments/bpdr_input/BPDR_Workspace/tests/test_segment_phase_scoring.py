import unittest
import numpy as np
from bpdr_input.segment_phase_scoring import score_frequencies


class SegmentPhaseScoringTests(unittest.TestCase):
    def test_invalid_predictions_stay_in_full_denominator_and_common_domain_is_paired(self):
        b=np.array([220.,-1.,np.nan]);c=np.array([220.,220.,220.]);mask=np.ones(3,dtype=bool)
        r=score_frequencies(b,c,mask,np.full(3,220.),mask)
        self.assertEqual(r['retained_denominator'],3);self.assertEqual(r['common_valid_voiced_samples'],1)
        self.assertEqual(r['arms']['B']['invalid_on_retained_samples'],2)
        self.assertEqual(r['arms']['C']['scored_voiced_samples'],3)
        self.assertEqual(r['arms']['C']['common_valid_absolute_error_cents']['samples'],1)

    def test_silent_retained_samples_are_reported_without_pitch_accuracy(self):
        f=np.full(3,220.);mask=np.ones(3,dtype=bool);voiced=np.array([1,0,0],dtype=bool)
        r=score_frequencies(f,f,mask,f,voiced)
        self.assertEqual(r['retained_silent_samples'],2);self.assertEqual(r['arms']['B']['scored_voiced_samples'],1)

    def test_empty_common_domain_keeps_null_error_and_nonzero_invalid_count(self):
        b=np.array([np.nan]);c=np.array([-1.]);mask=np.ones(1,dtype=bool)
        r=score_frequencies(b,c,mask,np.array([220.]),mask)
        self.assertIsNone(r['arms']['B']['absolute_error_cents'])
        self.assertEqual(r['arms']['C']['invalid_on_retained_samples'],1)
