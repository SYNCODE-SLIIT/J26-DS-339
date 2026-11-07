import unittest
import numpy as np
from bpdr_input.experiments.learning.original_coverage import covered_original_trace


class OriginalCoverageTests(unittest.TestCase):
    def fixture(self):
        records=[dict(kind='corrupted',target=np.array([0.,20.,20.,0.]),mask=np.ones(4,bool)),dict(kind='original',target=np.zeros(10),mask=np.ones(10,bool)),dict(kind='original',target=np.zeros(4),mask=np.ones(4,bool)),dict(kind='original',target=np.zeros(3),mask=np.zeros(3,bool))]
        trace=np.array([[[(0,1)]*12+[(0,0)]*12+[(1,0)]*12][0]]*2,np.int32)
        return records,trace

    def test_complete_pass_precedes_repeats_and_other_classes_unchanged(self):
        records,trace=self.fixture();actual,stats=covered_original_trace(records,trace,340)
        np.testing.assert_array_equal(actual[:,:24],trace[:,:24])
        draws=actual[:,24:].reshape(-1,2)
        self.assertEqual(len(set(map(tuple,draws[:14]))),14)
        self.assertEqual(set(map(tuple,draws)),{(1,j) for j in range(10)}|{(2,j) for j in range(4)})
        self.assertEqual(stats['repeated_draws_after_first_full_pass'],10)
        self.assertFalse(np.any(draws[:,0]==3))
        again,_=covered_original_trace(records,trace,340);np.testing.assert_array_equal(actual,again)

    def test_budget_exhaustion_refuses_false_complete_coverage(self):
        records,trace=self.fixture()
        with self.assertRaisesRegex(ValueError,'exceeds fixed query budget'):covered_original_trace(records,trace[:1],340)

    def test_nonzero_original_or_unlabelled_trace_is_refused(self):
        records,trace=self.fixture();records[1]['target'][0]=1.
        with self.assertRaisesRegex(ValueError,'Original target'):covered_original_trace(records,trace,340)
        records,trace=self.fixture();trace[0,24]=(3,0)
        with self.assertRaisesRegex(ValueError,'Unlabelled sampled frame'):covered_original_trace(records,trace,340)


if __name__=='__main__':unittest.main()
