import unittest
from decimal import Decimal,localcontext
import numpy as np
from bpdr_input.modeling.pesto_centered_precision import decode_centered

class CenteredPrecisionTests(unittest.TestCase):
    def test_local_peak_mass_cannot_move_constant_pitch(self):
        masses=np.linspace(.7,1.,101,dtype=np.float32)
        a=np.zeros((len(masses),384),np.float32);a[:,196]=masses
        result=decode_centered(a)
        np.testing.assert_array_equal(result,np.full(len(masses),440*np.exp2((196.-207)/36)))

    def test_matches_high_precision_exact_binary_weight_reference(self):
        rng=np.random.default_rng(339);a=rng.uniform(0,.03,(40,384)).astype(np.float32)
        for j,peak in enumerate([0,383,196,207]*10):a[j,peak]=rng.uniform(.7,1.)
        reference=[]
        with localcontext() as ctx:
            ctx.prec=90
            for row in a:
                center=int(np.argmax(row));idx=np.clip(center+np.arange(-2,3),0,383)
                weights=[Decimal.from_float(float(row[k])) for k in idx]
                mean=sum(w*int(k) for w,k in zip(weights,idx))/sum(weights)
                reference.append(float(Decimal(440)*(Decimal(2)**((mean-207)/36))))
        np.testing.assert_allclose(decode_centered(a),reference,rtol=4*np.finfo(float).eps,atol=0)

    def test_window_edges_and_invalid_inputs(self):
        a=np.zeros((2,384),np.float32);a[0,0]=.9;a[1,383]=.9
        np.testing.assert_allclose(decode_centered(a),440*np.exp2((np.array([0,383])-207)/36),rtol=0,atol=0)
        for bad in (np.zeros((1,384)),np.ones((1,383)),np.full((1,384),-1),np.full((1,384),np.nan)):
            with self.assertRaises(ValueError):decode_centered(bad)

if __name__=='__main__':unittest.main()
