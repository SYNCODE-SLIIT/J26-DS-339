import unittest
import numpy as np
from bpdr_input.experiments.learning.preservation_mining import mined_trace


class PreservationMiningTests(unittest.TestCase):
    def fixture(self):
        target=np.zeros(8,np.float32);target[[1,4]]=30
        rows=[dict(kind='corrupted',target=target,mask=np.ones(8,bool)),
              dict(kind='original',target=np.zeros(8,np.float32),mask=np.ones(8,bool))]
        pred=[np.array([0,30,10,0,20,30,14,0],np.float32)]*2
        guard=np.zeros(8,bool);guard[5]=True
        trace=np.array([[(0,1)]*12+[(0,2)]*12+[(1,3)]*12]*2,np.int32)
        return rows,trace,pred,[guard,guard.copy()]

    def test_positive_records_and_group_draws_unchanged_negative_labels_safe(self):
        rows,trace,pred,guard=self.fixture()
        actual,stats=mined_trace(rows,trace,pred,guard,339)
        np.testing.assert_array_equal(actual[:,:12],trace[:,:12])
        np.testing.assert_array_equal(actual[:,:,0],trace[:,:,0])
        np.testing.assert_array_equal(actual[:,12:18,1],np.full((2,6),6))
        self.assertTrue(np.isin(actual[:,24:30,1],[1,4]).all())
        np.testing.assert_array_equal(actual[:,18:24],trace[:,18:24])
        np.testing.assert_array_equal(actual[:,30:],trace[:,30:])
        self.assertEqual(stats['hard_draws'],24)
        again,_=mined_trace(rows,trace,pred,guard,339)
        np.testing.assert_array_equal(actual,again)

    def test_no_high_request_evidence_keeps_uniform_trace(self):
        rows,trace,pred,guard=self.fixture()
        actual,stats=mined_trace(rows,trace,[np.zeros_like(p) for p in pred],guard,339)
        np.testing.assert_array_equal(actual,trace)
        self.assertEqual(stats['hard_draws'],0)
        self.assertEqual(stats['uniform_fallbacks'],24)

    def test_invalid_scores_and_positive_class_are_refused(self):
        rows,trace,pred,guard=self.fixture();pred[0]=pred[0].copy();pred[0][0]=np.nan
        with self.assertRaisesRegex(ValueError,'Invalid mining evidence'):
            mined_trace(rows,trace,pred,guard,339)
        rows,trace,pred,guard=self.fixture();trace[0,0,1]=0
        with self.assertRaisesRegex(ValueError,'positive'):
            mined_trace(rows,trace,pred,guard,339)


if __name__=='__main__':unittest.main()
