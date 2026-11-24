import unittest
import numpy as np
from bpdr_input.experiments.learning.nearby_zero_training import nearby_zero_indices,resample_index

class NearbyZeroTests(unittest.TestCase):
    def test_unknown_and_positive_never_enter_pool(self):
        target=np.zeros(150); target[70:80]=20
        mask=np.ones(150,dtype=bool); mask[60:70]=False; mask[80:90]=False
        pool=nearby_zero_indices(mask,target)
        self.assertTrue(np.array_equal(pool,np.r_[27:60,90:123]))
        self.assertTrue(np.all(mask[pool] & (target[pool]==0)))
        self.assertEqual(len(nearby_zero_indices(mask,np.zeros(150))),0)

    def test_positive_and_empty_pool_preserve_random_stream(self):
        a=np.random.default_rng(9); b=np.random.default_rng(9)
        self.assertEqual(resample_index(5,np.array([1,2]),20,a),5)
        self.assertEqual(resample_index(5,np.array([],dtype=int),0,a),5)
        self.assertEqual(a.random(),b.random())

    def test_sample_stays_in_trusted_pool_or_legacy_position(self):
        rng=np.random.default_rng(9); pool=np.array([20,25,30])
        draws=[resample_index(100,pool,0,rng) for _ in range(200)]
        self.assertTrue(set(draws)<=set([100,20,25,30]))
        self.assertTrue(60<sum(x!=100 for x in draws)<140)

if __name__=='__main__': unittest.main()
