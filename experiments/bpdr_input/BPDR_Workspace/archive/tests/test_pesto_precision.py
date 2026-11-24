import unittest
import numpy as np
from bpdr_input.modeling.pesto_precision import decode_float64

class PestoPrecisionTests(unittest.TestCase):
    def test_single_bins_and_repeat_identity(self):
        a=np.zeros((3,384),np.float32);a[np.arange(3),[207,210,204]]=1
        f=decode_float64(a)
        np.testing.assert_allclose(f,[440,440*2**(1/12),440*2**(-1/12)],rtol=1e-14)
        np.testing.assert_array_equal(f,decode_float64(a.copy()))
        self.assertEqual(f.dtype,np.float64)

    def test_fractional_weighted_mean_and_invalid_inputs(self):
        a=np.zeros((1,384),np.float32);a[0,207]=.75;a[0,208]=.25
        self.assertAlmostEqual(float(decode_float64(a)[0]),440*2**((1/12)/12),places=10)
        for a in (np.zeros((1,384)),np.ones((1,383)),np.full((1,384),np.nan)):
            with self.assertRaises(ValueError):decode_float64(a)

if __name__=='__main__':unittest.main()
